# EVAI 정령 학습

`LiquidAI/LFM2.5-230M-Base` 백본 하나와 정령별 독립 LoRA를 사용한다. 선택한 정령의 adapter 하나만 활성화한다. 학습 대상은 정령 자신의 1인칭 정체성·성격·감정·행동·판단·말투·프로필·세계관·관계다. 한국어를 먼저 완성하고, 이후 영어·중국어로 확장한다.

원천은 `data/tbl/*.db`다. `src/spirit_dataset`이 수집 가능 정령 roster, `spirit.json`, prompt/completion과 제외 기록을 만든다. 시스템 기억과 사용자 문맥은 loss에서 제외하고 assistant completion만 학습한다. 문자열 Sno는 필드의 실제 문자열 테이블에서 해석하며 성우 정보는 학습하지 않는다.

`configs/training.yaml`이 학습 설정의 단일 소유자다. `train`과 `train-spirit`는 동일한 `SpiritLoraTrainer`를 사용한다. 정령별 rank 후보를 학습해 validation loss로 선택하고, test는 선택 이후에만 사용한다. validation이 없는 소량 데이터의 training-loss 선택은 일반화 검증을 뜻하지 않는다. 후보 adapter는 메모리에 보관하고 선택한 adapter 하나만 저장한다. 정령마다 백본 복사본이나 epoch checkpoint를 만들지 않는다.

```powershell
python -m cli build-dataset
python -m cli train garnet_rapture
python -m cli train-spirit
python -m cli evaluate garnet_rapture --love-level 5
python -m cli export garnet_rapture
python -m cli serve
```

웹·평가는 `SpiritRuntime`과 같은 기억 프롬프트 생성기를 사용한다. 정령 변경은 adapter 선택이며 전체 모델을 교체하지 않는다. 학습된 adapter가 없으면 API가 이를 오류로 알린다. 규칙 기반 문장을 학습 모델 응답으로 대체하지 않는다.

`export`는 공유 백본·선택 adapter·프로필의 경로와 SHA-256을 담는 native manifest를 `tmp-codex`에 기록한다. `export.spirit_bundle.load_exported_spirit`가 이를 검증하고 같은 runtime으로 읽는다. GGUF/Ollama 등 배포 플랫폼은 현재 목표로 확정하지 않는다.

기존 Full-FT는 변환의 reference일 뿐 새 학습 출력이 아니다. 원본을 검증 전에 삭제하지 않는다. 현재 파일 존재 여부와 실행 결과는 `HANDOFF.md`에만 기록한다. 금지어·붕괴 탐지 통과는 정체성 품질 합격이 아니다. 정체성·말투·감정·관계·세계관·연속 대화와 base 기능 보존을 별도로 확인한다. preference pair가 없으면 DPO를 실행하거나 완료했다고 표기하지 않는다.

소스는 `src/<domain>/`, 데이터셋은 `artifacts/datasets/<slug>/`, adapter는 `artifacts/adapters/<slug>/`에 둔다. 경로는 `common.paths`, 모델 로드는 `inference.model_loader`, 생성은 `inference.generation`, 학습 설정은 `training.spirit_lora`가 소유한다. PowerShell·설치된 Python 환경을 사용하고 임의 설치·다운그레이드를 하지 않는다.

원작 TBL·Story·대사, 프로젝트 계약값, 파생 기억, teacher 분석은 출처를 구분한다. Story 선택지와 실제 정령 응답은 함께 보존하며, 다른 화자의 말과 지문은 상황 입력이다. 메인 스토리의 파생 기억에는 `StoryInfo.No`와 근거 `Talk.No`를 남긴다. 완전한 출처 검증을 통과한 teacher 판단은 별도 `self_judgment` 학습 목표로 사용하며, 같은 사건의 발화와 동일한 split에 둔다. 일반 채팅 출력에 `<think>`를 붙이지 않는다.

`speech`는 원작 발화, `self_memory`는 짧은 자기 기억 회상, `self_judgment`는 관련 기억·상황 해석·감정·의도·행동·발화의 연결을 학습한다. 회상 정답을 system 입력에 미리 넣지 않는다. 일반 채팅에는 이름과 성인 여성 정령(200~600세), 성인 남성 구원자, 구원자를 향한 연애 감정이라는 프로젝트 계약 및 관계 수준을 전달한다. 나머지 세계관과 경험은 학습 대상으로 보존한다. 각 과제의 실제 학습 건수와 supervised token 수는 학습 보고서에 기록한다.

필수 검사: `ruff check src tests`, `pyright`, `python -m pytest`. 실제 실행 범위·진단과 제품 품질 상태를 구분한다. [아키텍처](docs/architecture.md), [원작 분석](docs/game_data_analysis.md), [현재 작업 상태](HANDOFF.md), [목표 원문](GOAL.md).
