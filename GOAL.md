# ASTRA 6 — GarnetRapture_evai RECOVERY DIRECTIVE

이 프로젝트를 처음부터 재설계하지 말고 현재 결과물·소스·아티팩트를 전부 이어받아 **잘못된 Full-FT 중심 구조를 사용자가 확정한 EVAI 구조로 완전히 복구한다.**

## 0. SOURCE OF TRUTH

충돌 시 우선순위는 다음과 같다.

**사용자의 최신 명시 지시 → 프로젝트 실측 사실 → 현재 소스 코드 → 문서/메모리**

기존 HANDOFF/memory/plan/docs에 있는 다음 주장은 폐기 대상이다.

`정령당 Full-FT 모델 1개가 올바른 최종 구조`
`직접 학습 = Full-parameter fine-tuning`
`LoRA는 persona skin`
`Android/GGUF/hotswap이 현재 주목적`

이 내용은 사용자 결정이 아니라 이전 Claude의 오판으로 기록된 것이다. 사실처럼 재사용하지 않는다.

------

# 1. 최종 목표

EVAI의 현재 1차 목표는 **한국어 정령 AI 학습**이다.

기준 모델:

```
LiquidAI/LFM2.5-230M-Base
```

최종 구조:

```text
LFM2.5-230M-Base
        │
        ├── Garnet adapter
        ├── Beleth adapter
        ├── Chloe adapter
        └── ...
```

런타임은 **LFM backbone 객체 하나**를 유지하고 선택된 정령의 weight delta만 활성화한다.

```text
SpiritId = garnet
→ Base + ΔW_garnet
→ Garnet만 활성
```

다른 정령 adapter는 연산에 개입하지 않는다.

여러 adapter를 동시에 merge하지 않는다.

정령의 목표 학습 대상은 말투만이 아니다.

**성격, 감정 표현, 행동양식, 판단방식, 사고방식, 응답방식, 프로필, 배경, 세계관, 관계, 구원자와의 관계 및 해당 정령의 1인칭 정체성**을 학습 데이터에 실제로 반영한다.

결과는 `AI Assistant가 Garnet처럼 말함`이 아니라 **Garnet 자체로 응답하는 모델 상태**여야 한다.

한국어를 먼저 완성한다. 다른 언어 확장은 그 이후다.

성우/CV 정보는 학습 대상이 아니다.

------

# 2. 환경 게이트

작업 시작 즉시 실제 환경을 다시 실측한다.

현재 마지막 검증값:

```text
Python       3.14.7
PyTorch      2.14.0+cu130
GPU          RTX 3070 / CUDA 사용 가능
Transformers 5.17.0
PEFT         0.21.0
TRL          1.13.0
```

Python 3.14.7은 현재 프로젝트가 실제 사용했던 버전이며 현재 Python 안정판이기도 하다.

오래된 예제를 이유로 Python을 임의 다운그레이드하지 않는다.

실제 import/training/runtime incompatibility가 재현될 때만 원인을 수정한다.

모델은 반드시:

```
LiquidAI/LFM2.5-230M-Base
```

를 기준으로 한다.

이 모델은 230M Base checkpoint이며 한국어를 포함하고, 언어·도메인 특화 fine-tuning 용도로 제공된 모델이다. 일반 Transformer라고 가정하지 말고 실제 `named_modules`, config, weight shape을 읽는다.

현재 구조는 14-layer hybrid LFM2 계열이므로 단순히:

```python
target_modules=["q_proj", "v_proj"]
```

처럼 다른 Transformer의 관습을 복사하지 않는다.

**근거: 실제 모델 구조를 먼저 측정하고 대상 weight를 결정한다.**

------

# 3. PROJECT AUDIT GATE

수정 전에 다음을 프로젝트 내부에서 완전히 파악한다.

`CLAUDE.md`
`HANDOFF.md`
`plan.md`
`configs/*`
`pyproject.toml`
`src/GarnetRapture_evai/**/*.py`
`tests/*`
`docs/*`
`data/*`
`artifacts/*`

외부 웹조사를 작업 대신 하지 않는다.

특히 현재 다음 실측 결함을 다시 확인한다.

```text
train.py        → 아직 Full-parameter SFT 중심
evaluate.py     → merged Full-FT 중심
web/server.py   → TrainedModelStore가 Full-FT 모델 교체 로드
export.py       → merged Full-FT를 전제로 함
adapters        → Garnet/Beleth 중심의 일부만 존재
95 Full-FT      → 약 41GB
기존 adapter    → 약 367MB 수준으로 지나치게 큼
dataset         → profile/world/relationship가 학습 입력에 충분히 반영되지 않음
prompt-less     → completion-only 레코드 존재
loss            → assistant-only 학습이 보장되지 않음
dialogue        → trailing user block 무기록 소실
tests           → 마지막 실측 2 failures
docs/memory     → Full-FT/LoRA 설명 상충
```

확인 없이 과거 HANDOFF 내용을 사실로 복사하지 않는다.

------

# 4. 미래 학습 경로 수정

기존 checkpoint 변환과 별개로 앞으로의 학습 자체도 최종 아키텍처와 일치시킨다.

`training.yaml`, `train.py`, 관련 CLI를 **Base + 정령별 LoRA SFT** 경로로 맞춘다.

학습 시 실제 모델 구조를 기준으로 LoRA target을 결정한다.

가넷 하나를 예로 들면:

```text
Base load
→ Garnet dataset
→ Garnet adapter만 trainable
→ Base weight 공유
→ Garnet adapter 저장
```

정령마다 Base 전체 복사본을 만들지 않는다.

새 학습체크포인트니뭐니지랄하지않는다. 병목, 디스크낭비, 애미뒤진 중복, 순환, 병신구조 금지다.

------

# 5. DATASET FIX

학습은 단순 대사 암기가 아니다.

현재 로드만 하고 학습 입력에 빠진 다음 정보를 실제 레코드 구성에 반영한다.

```text
profile
personality
background
world knowledge
relationships
speech patterns
dialogues
좋아하는 것/싫어하는 것 등 실제 등록 데이터
```

Sno는 숫자만 보고 다른 table에서 찾지 않는다.

**field가 속한 실제 TBL/JSON 종류에서 해당 Sno를 해석한다.**

`구원자`는 user 역할이다.

등록되지 않은 캐릭터를 임의로 `정령`으로 만들지 않는다.

source에 없는 설정을 창작하여 학습 데이터로 채우지 않는다.

현재 dataset이 비어 있는 persona도 먼저 원본 source와 builder를 재검증하고, 실제 source가 없다는 것이 확인된 경우에만 정확히 blocker로 기록한다.

### prompt-less record

speech pattern/greeting 등이 assistant completion만 던져지는 구조라면 그대로 학습시키지 않는다.

그 데이터가 어떤 상황·질문·상태에 대한 정령 응답인지 실제 source 의미를 보존하는 학습 record로 만든다.

### Loss

user/system context 자체를 답변처럼 학습하지 않는다.

SFT에서 **assistant completion이 학습 목표가 되도록 실제 token/loss mask를 검증**한다.

설정 이름만 존재하는 것으로 통과시키지 않는다.

------

# 6. DPO GATE

DPO라는 이름만 추가하지 않는다.

현재 실제 preference pair 데이터가 존재하는지 먼저 확인한다.

```text
prompt
chosen
rejected
```

구조와 품질이 실제로 존재해야 DPO를 수행한다.

없으면 SFT 결과를 DPO라고 부르지 않는다.

사용자가 요구한 architecture 복구를 DPO 연구로 우회하지 않는다.

------

# 7. RUNTIME FIX

`web/server.py`의 persona별 Full-FT 모델 교체 로딩을 최종 구조로 바꾼다.

목표:

```text
Base model 1개
+
여러 정령 adapter 보유
+
현재 선택된 adapter 1개만 활성
```

정령 변경 시 전체 400MB대 Base 모델을 다시 읽는 구조가 아니어야 한다.

`evaluate.py`도 실제 서비스와 다른 Full-FT 경로를 평가해서는 안 된다.

**최종 runtime과 동일한 Base + selected adapter 경로를 평가한다.**

web UI의 정령 선택 역시 동일한 runtime 상태를 사용한다.

------

# 8. 평가 게이트

기존 `12/12 PASS`만으로 완료 판정하지 않는다.

그 평가는 붕괴/금지 패턴 탐지 성격이 강하므로 정령 identity 평가와 구분한다.

최소 검증 대상:

```text
정령 혼합 여부
다른 정령 이름/성격 침범
generic AI assistant 성향
1인칭 identity
말투
감정
관계 인식
세계관/배경
행동양식
질문 대응 방식
동일 정령의 장기 일관성
Base 기능의 과도한 붕괴 여부
```

adapter 변환에서는 Full-FT를 reference로 사용한다.

새 LoRA 학습에서는 dataset holdout과 identity regression을 사용한다.

------

# 9. 작업 행동 규칙

작업은 다음 형태로 진행한다.

**근거 확인 → 수정 → 실제 실행 → 검증 → 다음 수정**

각 결론에는 실제 근거를 남긴다.

예:

```text
근거: 파일/함수/설정/실측값
수정: 무엇을 왜 바꿨는지
검증: 실행한 명령과 결과
```

추측을 문서화해서 다음 세션의 확정 사실로 만들지 않는다.

사용자가 정하지 않은 것을 `사용자가 결정했다`고 기록하지 않는다.

문제가 확인됐으면 선택지만 나열하고 멈추지 말고 직접 수정하고 검증한다.

부분 작업 후 나머지를 별도 과제로 밀어내지 않는다.

------

# 10. 프로젝트 작업 규칙

기존 프로젝트 규칙을 그대로 따른다.

```text
PowerShell 고정
Bash 사용 금지
기존 파일은 Edit
새 파일만 Write
temp는 tmp-claude
작업 종료 전 temp 정리
subagent 사용 금지
.py의 사용자 출력 메시지는 영어
사용자 응답은 한국어
일본어 응답 금지
작성 코드에 주석을 추가하지 않음
warning suppression으로 통과시키지 않음
```

ruff / pyright / pytest를 실제 실행한다.

마지막 결과는 warning/error/test failure를 숨기지 않는다.

------

# 11. 금지된 작업 방식

다음 행동을 반복하지 않는다.

- `"직접 학습"`을 자의적으로 Full-FT 허가로 해석
- 정령 95개를 전체 모델 95개로 만드는 것
- LoRA를 단순 말투 skin으로 취급
- 모든 정령 adapter를 동시에 merge
- 선택되지 않은 정령 delta를 추론에 참여시킴
- rank를 근거 없이 전체 정령에 고정
- q_proj/v_proj 같은 범용 예제만 복사
- 기존 Full-FT를 adapter 검증 전에 삭제


- epoch checkpoint 대량 생성
- 프로젝트 수정 대신 Android/GGUF/llama.cpp/hotswap 연구로 이탈
- 배포 플랫폼을 임의 결정
- 웹검색으로 현재 프로젝트 분석을 대체
- 사용자가 요구하지 않은 새 서비스/프레임워크/가드 추가
- source에 없는 정령 설정/대사 생성
- shallow 12/12 평가를 최종 품질 보증으로 사용
- 깨진 테스트를 기존 문제라고 밀어냄
- 잘못된 memory/HANDOFF를 근거로 같은 오류를 반복

------

# 12. 완료 기준

아래 상태가 실제로 성립해야 이 복구가 끝난다.

```text
LiquidAI/LFM2.5-230M-Base 단일 backbone

각 정령 = 독립 adapter/delta

선택한 정령 하나만 활성

기존 Full-FT → 검증된 adapter 변환 완료

새 학습 경로 = LoRA SFT

한국어 persona 데이터가 profile/world/relationship/dialogue까지 실제 반영

assistant target loss 검증

train/evaluate/web가 같은 adapter architecture 사용

Full-FT 중심 stale config/docs/memory 제거 또는 정정

불필요한 400MB급 adapter 복제 원인 해결

ruff PASS
pyright 0 error
pytest 0 fail

HANDOFF.md에는 실제 완료 결과와 남은 blocker만 기록
```

한국어 1차 목표가 위 상태로 완료된 뒤에만 사용자가 이전에 말한 영어/기타 언어 확장과 추가 서비스 작업으로 진행한다.

**현재 할 일은 설명문 작성이 아니라 이 구조로 프로젝트를 실제 복구하고, 각 단계의 근거를 실측으로 증명하면서 끝까지 수행하는 것이다.**