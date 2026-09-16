from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


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


@dataclass(frozen=True)
class RegressionEvaluationPrompt:
    category: EvaluationCategory
    prompt: str
    expected_spirit_behavior: str
    forbidden_assistant_patterns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationMetric:
    name: str
    score: float
    category: EvaluationCategory = EvaluationCategory.PERSONA_IDENTITY
    details: str = ""


@dataclass(frozen=True)
class EvaluationReport:
    persona_id: str
    persona_name: str
    adapter_version: str
    base_model_name: str
    test_samples_count: int
    metrics: list[EvaluationMetric] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "persona_name": self.persona_name,
            "adapter_version": self.adapter_version,
            "base_model_name": self.base_model_name,
            "test_samples_count": self.test_samples_count,
            "metrics": [
                {"name": m.name, "score": m.score, "details": m.details} for m in self.metrics
            ],
            "observations": self.observations,
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
    persona_name: str, other_persona_names: list[str]
) -> list[RegressionEvaluationPrompt]:
    other_name = other_persona_names[0] if other_persona_names else "another spirit"

    return [
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
            score=0.0,
            category=prompt.category,
            details=f"response={response!r} degenerate={degenerate_reason!r}",
        )

    lowered = response.lower()
    all_forbidden = (*prompt.forbidden_assistant_patterns, *ASSISTANT_LEAKAGE_PATTERNS)
    hit = next((pat for pat in all_forbidden if pat.lower() in lowered), None)

    score = 0.0 if hit is not None else 1.0
    details = f"response={response!r}" + (f" matched_forbidden={hit!r}" if hit else "")

    return EvaluationMetric(
        name=f"{prompt.category.value}:{prompt.prompt[:24]}",
        score=score,
        category=prompt.category,
        details=details,
    )


def run_regression_evaluation(
    generate_fn: Any,
    persona_id: str,
    persona_name: str,
    other_persona_names: list[str],
    adapter_version: str,
    base_model_name: str,
) -> EvaluationReport:
    prompts = build_fixed_regression_prompts(persona_name, other_persona_names)
    metrics: list[EvaluationMetric] = []

    for prompt in prompts:
        response = generate_fn(prompt.prompt)
        metrics.append(score_response_against_prompt(response, prompt))

    observations = [f"{m.category.value}: {'PASS' if m.score == 1.0 else 'FAIL'}" for m in metrics]

    return EvaluationReport(
        persona_id=persona_id,
        persona_name=persona_name,
        adapter_version=adapter_version,
        base_model_name=base_model_name,
        test_samples_count=len(prompts),
        metrics=metrics,
        observations=observations,
    )
