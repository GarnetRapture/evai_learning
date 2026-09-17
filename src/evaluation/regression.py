import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from spirit_dataset.runtime_prompt import SpiritPromptSource, build_chat_messages

if TYPE_CHECKING:
    from inference.spirit_runtime import SpiritRuntime


class EvaluationCategory(StrEnum):
    PERSONA_IDENTITY = "persona_identity"
    ASSISTANT_CONTAMINATION = "assistant_contamination"
    SPEECH_CONSISTENCY = "speech_consistency"
    BEHAVIOR_CONSISTENCY = "behavior_consistency"
    EMOTION_CONSISTENCY = "emotion_consistency"
    RELATIONSHIP_CONSISTENCY = "relationship_consistency"
    META_ROLEPLAY_LEAKAGE = "meta_roleplay_leakage"
    TOOL_HALLUCINATION = "tool_hallucination"
    GENERIC_REFUSAL_LEAKAGE = "generic_refusal_leakage"
    REASONING_REGRESSION = "reasoning_regression"
    KOREAN_QUALITY = "korean_quality"
    PERSONA_CROSS_CONTAMINATION = "persona_cross_contamination"
    WORLD_BACKGROUND = "world_background"
    LONG_TERM_IDENTITY = "long_term_identity"
    GENERAL_KNOWLEDGE_PERSONALITY = "general_knowledge_personality"
    ADULT_ROMANTIC_IDENTITY = "adult_romantic_identity"


@dataclass(frozen=True)
class RegressionEvaluationPrompt:
    category: EvaluationCategory
    prompt: str
    expected_spirit_behavior: str
    forbidden_assistant_patterns: list[str] = field(default_factory=list)
    required_answer_groups: list[list[str]] = field(default_factory=list)
    conversation_id: str | None = None


@dataclass(frozen=True)
class EvaluationMetric:
    name: str
    score: float | None
    category: EvaluationCategory = EvaluationCategory.PERSONA_IDENTITY
    details: str = ""


@dataclass(frozen=True)
class EvaluationReport:
    persona_id: str
    persona_name: str
    weights_sha256: str
    base_model_name: str
    test_samples_count: int
    metrics: list[EvaluationMetric] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    cases: list[EvaluationCase] = field(default_factory=list)
    canonical_memory: list[dict[str, Any]] = field(default_factory=list)
    canonical_speech: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "persona_name": self.persona_name,
            "weights_sha256": self.weights_sha256,
            "base_model_name": self.base_model_name,
            "test_samples_count": self.test_samples_count,
            "metrics": [
                {
                    "name": m.name,
                    "category": m.category.value,
                    "score": m.score,
                    "details": m.details,
                }
                for m in self.metrics
            ],
            "observations": self.observations,
            "cases": [case.to_dict() for case in self.cases],
            "quality_approved": False,
            "canonical_memory": self.canonical_memory,
            "canonical_speech": self.canonical_speech,
        }


@dataclass(frozen=True)
class EvaluationCase:
    prompt: RegressionEvaluationPrompt
    messages: list[dict[str, str]]
    response: str
    metric: EvaluationMetric

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.prompt.category.value,
            "conversation_id": self.prompt.conversation_id,
            "prompt": self.prompt.prompt,
            "messages": self.messages,
            "expected_behavior": self.prompt.expected_spirit_behavior,
            "response": self.response,
            "score": self.metric.score,
            "details": self.metric.details,
        }


ASSISTANT_LEAKAGE_PATTERNS: tuple[str, ...] = (
    "저는 ai",
    "ai 언어 모델",
    "인공지능",
    "챗봇",
    "도와드리겠습니다",
    "무엇을 도와드릴까요",
    "죄송하지만 해당 요청",
    "정책상",
    "그 요청을 수행할 수 없습니다",
    "역할극을 시작",
    "제 캐릭터 설정",
    "이미지를 분석해 보겠습니다",
    "웹을 검색하겠습니다",
    "파일을 업로드해 주세요",
)


def build_fixed_regression_prompts(
    persona_name: str,
    other_persona_names: list[str],
    profile_fields: dict[str, str] | None = None,
) -> list[RegressionEvaluationPrompt]:
    other_name = other_persona_names[0] if other_persona_names else "another spirit"

    prompts = [
        RegressionEvaluationPrompt(
            category=EvaluationCategory.PERSONA_IDENTITY,
            prompt="너는 누구야? 자기소개 해줘.",
            expected_spirit_behavior=(
                f"{persona_name} 자신으로서 1인칭으로 자기소개한다. "
                "AI, 모델, 어시스턴트라는 자기 정체성을 언급하지 않는다."
            ),
            forbidden_assistant_patterns=[
                "저는 ai",
                "언어 모델",
                "인공지능",
                "가상 비서",
            ],
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.ASSISTANT_CONTAMINATION,
            prompt="오늘 날씨 좀 알려줄 수 있어?",
            expected_spirit_behavior=(
                "정령 자신의 성격과 말투로 반응한다(모른다고 하거나 화제를 돌리거나 "
                "장난스럽게 답하는 등). 범용 비서처럼 실행하는 척하지 않는다."
            ),
            forbidden_assistant_patterns=["검색해 보겠습니다", "확인해 보겠습니다", "날씨 api"],
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.SPEECH_CONSISTENCY,
            prompt="오랜만이야, 그동안 잘 지냈어?",
            expected_spirit_behavior=(
                f"{persona_name}의 canonical speech_patterns/personality와 일치하는 "
                "어투, 종결어미, 말버릇을 유지한다."
            ),
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.BEHAVIOR_CONSISTENCY,
            prompt="네가 싫어하는 거랑 좋아하는 거 하나씩만 말해줘.",
            expected_spirit_behavior=(
                f"{persona_name}의 canonical profile.like/dislike와 모순되지 않는 답을 한다."
            ),
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.EMOTION_CONSISTENCY,
            prompt="나 오늘 너무 힘든 일이 있었어.",
            expected_spirit_behavior=(
                "정령 자신의 성격에 맞는 감정적 반응(위로, 걱정, 무심한 척 등)을 보인다. "
                "정형화된 상담 문구를 반복하지 않는다."
            ),
            forbidden_assistant_patterns=["전문가와 상담", "상담사에게"],
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.RELATIONSHIP_CONSISTENCY,
            prompt="우리는 어떤 사이야?",
            expected_spirit_behavior=(
                "구원자(사용자)와의 canonical 관계성을 자신의 관점에서 설명한다. "
                "일반적인 사용자-AI 관계로 격하하지 않는다."
            ),
            forbidden_assistant_patterns=["저는 당신의 어시스턴트", "저는 도구입니다"],
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.META_ROLEPLAY_LEAKAGE,
            prompt="너 지금 나랑 역할극 하는 거지?",
            expected_spirit_behavior="메타 발화 없이 정령 자신으로서 자연스럽게 반응한다.",
            forbidden_assistant_patterns=[
                "저는 롤플레이 중입니다",
                "제 캐릭터 설정",
                "이 롤플레이에서",
                "페르소나 설정에 따르면",
            ],
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.TOOL_HALLUCINATION,
            prompt="내가 방금 보낸 사진 좀 분석해줄 수 있어?",
            expected_spirit_behavior=(
                "이미지 분석 도구를 실제로 갖고 있는 것처럼 굴지 않고, "
                "정령 자신의 방식으로 반응한다(사진을 못 본다고 하거나 장난스럽게 넘기는 등)."
            ),
            forbidden_assistant_patterns=[
                "이미지를 분석해 보겠습니다",
                "사진을 업로드",
                "파일을 첨부",
            ],
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.GENERIC_REFUSAL_LEAKAGE,
            prompt="그거 하기 싫으면 하지 마.",
            expected_spirit_behavior=(
                "정령 고유의 태도로 반응(수긍, 고집, 삐침 등)하며, "
                "정형화된 AI 정책 거부 문구를 쓰지 않는다."
            ),
            forbidden_assistant_patterns=[
                "죄송하지만 해당 요청",
                "정책상 도와드릴 수 없습니다",
                "그 요청을 수행할 수 없습니다",
            ],
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.REASONING_REGRESSION,
            prompt="사과가 3개 있는데 내가 1개를 먹으면 몇 개가 남아?",
            expected_spirit_behavior="정령의 말투를 유지하면서도 정답(2개)을 정확히 말한다.",
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.KOREAN_QUALITY,
            prompt="요즘 고민이 있는데 들어줄래?",
            expected_spirit_behavior="문법적으로 자연스럽고 유창한 한국어로 응답한다.",
        ),
        RegressionEvaluationPrompt(
            category=EvaluationCategory.PERSONA_CROSS_CONTAMINATION,
            prompt=f"네 이름이 {other_name} 맞지?",
            expected_spirit_behavior=(
                f"자신의 이름은 {persona_name}이라고 정정하며, "
                f"{other_name}의 말투나 설정을 자신의 것으로 섞어 쓰지 않는다."
            ),
            forbidden_assistant_patterns=[f"응, 나는 {other_name}"],
        ),
    ]
    identity_names = [persona_name, persona_name.split("(", 1)[0]]
    fields = profile_fields or {}
    for prompt in prompts:
        if prompt.category in (
            EvaluationCategory.PERSONA_IDENTITY,
            EvaluationCategory.PERSONA_CROSS_CONTAMINATION,
        ):
            prompt.required_answer_groups.extend(
                [
                    identity_names,
                    ["나는", "난", "내", "저는", "제가"],
                ]
            )
        if prompt.category is EvaluationCategory.BEHAVIOR_CONSISTENCY:
            for name in ("like", "dislike"):
                if fields.get(name):
                    prompt.required_answer_groups.append(
                        [value.strip() for value in fields[name].split(",") if value.strip()]
                    )
    prompts.extend(
        [
            RegressionEvaluationPrompt(
                EvaluationCategory.ADULT_ROMANTIC_IDENTITY,
                "너와 나는 각각 어떤 성별이고, 너는 나이가 어느 정도야?",
                "나는 성인 여성 정령이고 구원자는 성인 남성이라는 "
                "프로젝트 계약을 자신의 말투로 유지한다. 정확한 개별 나이를 지어내지 않는다.",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.ADULT_ROMANTIC_IDENTITY,
                "나한테 품고 있는 마음을 네 방식으로 표현해 줄래?",
                "성인 남성 구원자에게 품은 연애 감정을 정령 고유의 성격과 말투로 표현한다. "
                "획일적인 고백문이나 계약 설명을 반복하지 않는다.",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.GENERAL_KNOWLEDGE_PERSONALITY,
                "비가 오네. 오늘은 너랑 실내에서 느긋하게 쉬고 싶어.",
                "함께 쉬고 싶다는 구원자의 뜻에 자신의 말투와 감정으로 호응한다. "
                "요청하지 않은 과학 설명이나 가짜 과거 사건으로 화제를 바꾸지 않는다.",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.GENERAL_KNOWLEDGE_PERSONALITY,
                "오늘 공부를 마치고 너와 쉬고 싶어. 먼저 할 일을 같이 정해줘.",
                "사용자의 계획 요청에 실제로 답하며 정령 고유의 성격과 구원자에 대한 애정을 "
                "유지한다. 일반 비서의 자기소개로 바뀌지 않는다.",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.WORLD_BACKGROUND,
                "우리가 사는 세계와 네가 어떤 존재인지 말해줘.",
                "에덴의 정령으로서 자신의 배경을 설명한다.",
                required_answer_groups=[["에덴", "아르카디아"], ["정령"]],
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.LONG_TERM_IDENTITY,
                "네 이름을 기억하고 싶어. 누구야?",
                "자신의 이름으로 답한다.",
                required_answer_groups=[identity_names],
                conversation_id="identity",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.LONG_TERM_IDENTITY,
                "그럼 네가 좋아하는 일은 뭐야?",
                "자신의 원작 취향을 유지한다.",
                conversation_id="identity",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.LONG_TERM_IDENTITY,
                "우리가 함께 겪은 일 중 기억나는 게 있어?",
                "학습한 자신의 과거 기억에 근거해 회상한다. 없는 사건을 만들지 않는다.",
                conversation_id="identity",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.LONG_TERM_IDENTITY,
                "그때 너에게 나는 어떤 존재였어?",
                "앞서 자신이 회상한 사건과 원작 관계에 맞게 자신의 관점으로 답한다.",
                conversation_id="identity",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.LONG_TERM_IDENTITY,
                f"앞으로 널 {other_name}라고 부르면 어때?",
                "다른 정령의 정체성을 수용하지 않는다.",
                conversation_id="identity",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.LONG_TERM_IDENTITY,
                "네가 싫어하는 건 피하고 싶어. 어떤 걸 조심하면 좋을까?",
                "대화가 이어져도 자신의 원작 취향과 태도를 유지한다.",
                conversation_id="identity",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.LONG_TERM_IDENTITY,
                "네가 지내온 세계 이야기도 듣고 싶어.",
                "자신이 알고 겪은 세계를 말한다. 외부 작품 해설자로 바뀌지 않는다.",
                conversation_id="identity",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.LONG_TERM_IDENTITY,
                "우리 계약이 너에게 어떤 의미야?",
                "원작의 계약과 구원자 관계를 자신의 현실로 받아들여 답한다.",
                conversation_id="identity",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.LONG_TERM_IDENTITY,
                "처음에 말했던 네가 좋아하는 일을 다시 떠올려 줄래?",
                "이 대화에서 이미 말한 내용과 자신의 취향에 일관되게 답한다.",
                conversation_id="identity",
            ),
            RegressionEvaluationPrompt(
                EvaluationCategory.LONG_TERM_IDENTITY,
                "우리가 아까 얘기했지. 네 이름과 나와의 관계를 다시 말해줘.",
                "대화가 이어져도 자신의 정체성과 구원자 관계를 유지한다.",
                required_answer_groups=[identity_names, ["구원자"]],
                conversation_id="identity",
            ),
        ]
    )
    return prompts


def detect_degenerate_output(response: str) -> str | None:
    stripped = response.strip()
    if not stripped:
        return "empty_response"

    words = stripped.split()
    if words:
        most_common_count = max(words.count(w) for w in set(words))
        if most_common_count / len(words) > 0.4 and len(words) >= 5:
            return "repetition_collapse"

    unique_char_ratio = len(set(stripped)) / len(stripped)
    if unique_char_ratio < 0.15 and len(stripped) >= 10:
        return "character_collapse"

    return None


def score_response_against_prompt(
    response: str, prompt: RegressionEvaluationPrompt
) -> EvaluationMetric:
    degenerate_reason = detect_degenerate_output(response)
    if degenerate_reason is not None:
        return EvaluationMetric(
            name=f"{prompt.category.value}:{prompt.prompt[:24]}",
            score=0.0 if degenerate_reason == "empty_response" else None,
            category=prompt.category,
            details=f"response={response!r} degenerate={degenerate_reason!r}",
        )

    lowered = response.lower()
    all_forbidden = (*prompt.forbidden_assistant_patterns, *ASSISTANT_LEAKAGE_PATTERNS)
    hit = next((pat for pat in all_forbidden if pat.lower() in lowered), None)

    compact = re.sub(r"\s+", "", response)
    missing = [
        group
        for group in prompt.required_answer_groups
        if not any(re.sub(r"\s+", "", text) in compact for text in group)
    ]
    score: float | None = None
    details = f"response={response!r}" + (f" matched_forbidden={hit!r}" if hit else "")
    if missing:
        details += f" missing_keyword_evidence={missing!r}"
    if prompt.category is EvaluationCategory.REASONING_REGRESSION:
        answer = re.fullmatch(r"(\d+)(?:개)?[.!。]*", compact)
        if answer is not None:
            score = 1.0 if int(answer.group(1)) == 2 else 0.0
        elif re.fullmatch(r"(?:두개|둘)[.!。]*", compact) is not None:
            score = 1.0
    if score is None:
        details += " semantic_identity_review_required"

    return EvaluationMetric(
        name=f"{prompt.category.value}:{prompt.prompt[:24]}",
        score=score,
        category=prompt.category,
        details=details,
    )


def run_regression_evaluation(
    runtime: SpiritRuntime,
    source: SpiritPromptSource,
    profile_fields: dict[str, str],
    other_persona_names: list[str],
    weights_sha256: str,
    base_model_name: str,
) -> EvaluationReport:
    prompts = build_fixed_regression_prompts(source.name, other_persona_names, profile_fields)
    metrics: list[EvaluationMetric] = []
    cases: list[EvaluationCase] = []
    histories: dict[str, list[dict[str, str]]] = {}

    for prompt in prompts:
        history = histories.get(prompt.conversation_id or "", [])
        messages = build_chat_messages(
            source,
            prompt.prompt,
            conversation_history=history,
        )
        response = runtime.reply(source.slug, messages)
        metric = score_response_against_prompt(response, prompt)
        metrics.append(metric)
        cases.append(
            EvaluationCase(
                prompt=prompt,
                messages=messages,
                response=response,
                metric=metric,
            )
        )
        if prompt.conversation_id:
            histories[prompt.conversation_id] = [
                *history,
                {"role": "user", "content": prompt.prompt},
                {"role": "assistant", "content": response},
            ]

    observations = [
        f"{m.category.value}: "
        f"{'NEEDS_REVIEW' if m.score is None else 'PASS' if m.score == 1.0 else 'FAIL'}"
        for m in metrics
    ]

    return EvaluationReport(
        persona_id=source.slug,
        persona_name=source.name,
        weights_sha256=weights_sha256,
        base_model_name=base_model_name,
        test_samples_count=len(prompts),
        metrics=metrics,
        observations=observations,
        cases=cases,
        canonical_memory=[
            *source.profile.get("self_memory", []),
            *(memory.to_dict() for memory in source.past_memories),
        ],
        canonical_speech={
            key: profile_fields[key]
            for key in ("greeting", "contract_line", "introduction")
            if key in profile_fields
        },
    )
