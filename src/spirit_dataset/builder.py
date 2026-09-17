import json
from collections import Counter
from contextlib import ExitStack, closing
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from common.paths import DATASETS_DIR, TBL_DIR, ensure_artifact_directories
from game_data.localization import StringResolver
from game_data.references import StringTableReferences
from game_data.story import StoryRepository
from sft_dataset.dialogue import SpeakerRole, TurnClassification, classify_dialogue_turn
from sft_dataset.manifest import compute_file_sha256
from sft_dataset.records import SFTRecordTurn
from sft_dataset.split import SplitConfig
from sft_dataset.storage import persona_dataset_dir, sft_split_path, write_records_jsonl
from spirit_dataset.judgment import (
    SELF_MEMORY_CUE,
    apply_source_judgments,
    spirit_judgment_path,
    split_with_judgments,
)
from spirit_dataset.memory import (
    MIN_LOVE_LEVEL,
    PastMemory,
    PastMemoryRepository,
    compose_system_memory,
    spirit_memory_path,
)
from spirit_dataset.profile import SpiritProfile, SpiritProfileRepository, core_identity_memory
from spirit_dataset.records import (
    ExclusionReason,
    JudgmentTrace,
    MemoryEvidence,
    SourceClass,
    SourceKind,
    SourceReference,
    SpiritExclusionRecord,
    SpiritTrainingRecord,
    TrainingTask,
)
from spirit_dataset.roster import SpiritIdentity, SpiritRoster, load_spirit_roster
from spirit_dataset.sources import CanonicalExchange, SpiritSourceReader

SPIRIT_DATASET_VERSION = "1.5.0"
SOURCE_DATABASES: tuple[str, ...] = (
    "evertalk",
    "hero",
    "lobby",
    "localization",
    "meta",
    "story",
    "town",
    "trip",
)
ROSTER_FILE_NAME = "spirit_roster.json"
SPIRIT_FILE_NAME = "spirit.json"
EXCLUSIONS_SPLIT = "exclusions"
CONTENT_SOURCE_TYPE = "tbl"


@dataclass(frozen=True)
class SpiritDatasetSummary:
    identity: SpiritIdentity
    record_count: int
    exclusion_count: int
    train_count: int
    validation_count: int
    test_count: int
    episodic_memory_count: int


@dataclass
class SpiritDatasetBuildResult:
    roster: SpiritRoster
    summaries: list[SpiritDatasetSummary] = field(default_factory=list)
    spirits_without_records: list[SpiritIdentity] = field(default_factory=list)


def render_prompt(
    record_memory: tuple[str, ...], exchange: CanonicalExchange
) -> list[SFTRecordTurn]:
    turns = [SFTRecordTurn(role=SpeakerRole.SYSTEM.value, content="\n".join(record_memory))]
    if exchange.previous_spirit_text is not None:
        turns.append(
            SFTRecordTurn(role=SpeakerRole.ASSISTANT.value, content=exchange.previous_spirit_text)
        )
    user_content = "\n".join(exchange.user_items) if exchange.user_items else exchange.situation
    turns.append(SFTRecordTurn(role=SpeakerRole.USER.value, content=user_content))
    return turns


class SpiritDatasetBuilder:
    def __init__(self, split_config: SplitConfig | None = None) -> None:
        self._split_config = split_config if split_config is not None else SplitConfig()
        with ExitStack() as resources:
            self._resolver = resources.enter_context(closing(StringResolver()))
            self._references = StringTableReferences()
            self._profiles = resources.enter_context(closing(
                SpiritProfileRepository(self._resolver, self._references)
            ))
            story = resources.enter_context(closing(StoryRepository(self._resolver)))
            self._memories = PastMemoryRepository(story)
            self._sources = SpiritSourceReader(self._resolver, self._references, story)
            self._source_hashes = {
                name: compute_file_sha256(TBL_DIR / f"{name}.db") for name in SOURCE_DATABASES
            }
            self._resources = resources.pop_all()

    def close(self) -> None:
        self._resources.close()

    @staticmethod
    def _memory_records(
        profile: SpiritProfile, past_memories: list[PastMemory],
    ) -> list[SpiritTrainingRecord]:
        memories: list[tuple[str, str, tuple[MemoryEvidence, ...], int | None, int]] = [
            (memory.text, memory.cue, memory.evidence, None, MIN_LOVE_LEVEL)
            for memory in profile.self_memory
        ]
        memories.extend(
            (memory.text, "네가 예전에 겪은 일 하나 들려줄래?", (
                MemoryEvidence(SourceClass.CANON_STORY, f"StoryInfo.No={memory.story_no}"),
                *(MemoryEvidence(SourceClass.CANON_DIALOGUE, f"Talk.No={key}")
                  for key in memory.talk_keys),
            ), memory.story_no, memory.love_level_min)
            for memory in past_memories
        )
        records: list[SpiritTrainingRecord] = []
        for index, (text, cue, evidence, story_no, love_level) in enumerate(memories):
            source = SourceReference(
                SourceKind.SELF_MEMORY, "self_memory", (profile.identity.hero_no, index),
                story_no=story_no, source_class=SourceClass.DERIVED_MEMORY,
            )
            records.append(SpiritTrainingRecord(
                id=f"self_memory:{index:05d}", source=source, love_level=love_level,
                judgment=JudgmentTrace(situation=cue, activated_memory=(text,)),
                prompt=[
                    SFTRecordTurn("system", "\n".join(compose_system_memory(
                        core_identity_memory(profile.identity_memory), [], love_level,
                    ))),
                    SFTRecordTurn("user", f"{SELF_MEMORY_CUE}\n{cue}"),
                ],
                completion=[SFTRecordTurn("assistant", text)],
                source_class=SourceClass.DERIVED_MEMORY, evidence=evidence,
                task=TrainingTask.SELF_MEMORY,
            ))
        return records

    def _records(
        self, profile: SpiritProfile
    ) -> tuple[
        list[SpiritTrainingRecord], list[SpiritExclusionRecord], Counter[str], list[PastMemory]
    ]:
        identity = profile.identity
        past_memories = self._memories.load(identity)
        material = self._sources.read(profile)
        records: list[SpiritTrainingRecord] = []
        exclusions = [
            SpiritExclusionRecord(
                reason=skip.reason,
                detail=skip.detail,
                source=skip.source,
                text_preview=skip.text[:80],
            )
            for skip in material.skips
        ]
        seen: set[tuple[tuple[str, str], ...]] = set()
        duplicates: Counter[str] = Counter()
        for exchange in material.exchanges:
            classification = classify_dialogue_turn(
                SpeakerRole.ASSISTANT, identity.name, exchange.spirit_text, CONTENT_SOURCE_TYPE
            )
            if classification is not TurnClassification.ACCEPTED:
                exclusions.append(
                    SpiritExclusionRecord(
                        reason=ExclusionReason.CONTENT_CLASSIFICATION,
                        detail=classification.value,
                        source=exchange.source,
                        text_preview=exchange.spirit_text[:80],
                    )
                )
                continue
            prompt = render_prompt(
                compose_system_memory(profile.identity_memory, past_memories, exchange.love_level),
                exchange,
            )
            completion = [
                SFTRecordTurn(role=SpeakerRole.ASSISTANT.value, content=exchange.spirit_text)
            ]
            signature = tuple((turn.role, turn.content) for turn in (*prompt, *completion))
            if signature in seen:
                duplicates[exchange.source.kind.value] += 1
                continue
            seen.add(signature)
            records.append(
                SpiritTrainingRecord(
                    id=f"{exchange.source.kind.value}:{len(records):05d}",
                    source=exchange.source,
                    love_level=exchange.love_level,
                    judgment=JudgmentTrace(situation=exchange.situation, emotion=exchange.emotion),
                    prompt=prompt,
                    completion=completion,
                )
            )
        records.extend(self._memory_records(profile, past_memories))
        records, speaker_exclusions = apply_source_judgments(identity.slug, records)
        exclusions.extend(speaker_exclusions)
        # Validate annotations against all available memories first. Normal dialogue
        # then learns to speak from its weights without the answer in its system input.
        records = [replace(record, prompt=[
            SFTRecordTurn("system", "\n".join(compose_system_memory(
                core_identity_memory(profile.identity_memory), [], record.love_level,
            ))),
            *record.prompt[1:],
        ]) for record in records]
        return records, exclusions, duplicates, past_memories

    def _manifest(
        self,
        profile: SpiritProfile,
        records: list[SpiritTrainingRecord],
        exclusions: list[SpiritExclusionRecord],
        duplicates: Counter[str],
        split_counts: dict[str, int],
        episodic_memory_count: int,
    ) -> dict[str, Any]:
        memory_path = spirit_memory_path(profile.identity.slug)
        judgment_path = spirit_judgment_path(profile.identity.slug)
        return {
            "dataset_version": SPIRIT_DATASET_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "hero_no": profile.identity.hero_no,
            "slug": profile.identity.slug,
            "name": profile.identity.name,
            "source_databases_sha256": self._source_hashes,
            "episodic_memory_file": memory_path.name if memory_path.exists() else None,
            "episodic_memory_sha256": (
                compute_file_sha256(memory_path) if memory_path.exists() else None
            ),
            "episodic_memory_count": episodic_memory_count,
            "judgment_file": judgment_path.name if judgment_path.exists() else None,
            "judgment_sha256": (
                compute_file_sha256(judgment_path) if judgment_path.exists() else None
            ),
            "factual_memory_count": len(profile.identity_memory),
            "records_by_source": dict(Counter(record.source.kind.value for record in records)),
            "records_by_source_class": dict(Counter(
                record.source_class.value for record in records
            )),
            "records_by_task": dict(Counter(record.task.value for record in records)),
            "judgment_complete_count": sum(
                record.task is TrainingTask.SPEECH and record.judgment.is_complete
                for record in records
            ),
            "exclusions_by_reason": dict(Counter(item.reason.value for item in exclusions)),
            "duplicates_removed_by_source": dict(duplicates),
            "splits": split_counts,
        }

    def build_spirit(self, identity: SpiritIdentity) -> SpiritDatasetSummary:
        profile = self._profiles.load(identity)
        records, exclusions, duplicates, past_memories = self._records(profile)
        episodic_count = len(past_memories)
        split = split_with_judgments(records, self._split_config)
        records = [*split.train, *split.validation, *split.test]
        split_counts = {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        }
        write_records_jsonl(sft_split_path(identity.slug, "train"), split.train)
        write_records_jsonl(sft_split_path(identity.slug, "validation"), split.validation)
        write_records_jsonl(sft_split_path(identity.slug, "test"), split.test)
        write_records_jsonl(sft_split_path(identity.slug, EXCLUSIONS_SPLIT), exclusions)
        (persona_dataset_dir(identity.slug) / SPIRIT_FILE_NAME).write_text(
            json.dumps(
                {
                    "manifest": self._manifest(
                        profile, records, exclusions, duplicates, split_counts, episodic_count
                    ),
                    "profile": profile.to_dict(),
                    "episodic_memory": [memory.to_dict() for memory in past_memories],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return SpiritDatasetSummary(
            identity=identity,
            record_count=len(records),
            exclusion_count=len(exclusions),
            train_count=split_counts["train"],
            validation_count=split_counts["validation"],
            test_count=split_counts["test"],
            episodic_memory_count=episodic_count,
        )

    def build_all(self, slugs: set[str] | None = None) -> SpiritDatasetBuildResult:
        ensure_artifact_directories()
        roster = load_spirit_roster(self._resolver, self._references)
        result = SpiritDatasetBuildResult(roster=roster)
        for identity in roster.spirits:
            if slugs is not None and identity.slug not in slugs:
                continue
            summary = self.build_spirit(identity)
            result.summaries.append(summary)
            if summary.record_count == 0:
                result.spirits_without_records.append(identity)
        (DATASETS_DIR / ROSTER_FILE_NAME).write_text(
            json.dumps(
                {
                    "dataset_version": SPIRIT_DATASET_VERSION,
                    "spirits": [
                        {
                            "hero_no": identity.hero_no,
                            "slug": identity.slug,
                            "name": identity.name,
                            "name_en": identity.name_en,
                            "legacy_persona_file": (
                                None
                                if identity.legacy_persona_file is None
                                else identity.legacy_persona_file.name
                            ),
                        }
                        for identity in roster.spirits
                    ],
                    "unmatched_legacy_persona_files": [
                        path.name for path in roster.unmatched_legacy_files
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result
