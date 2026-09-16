# evai_learning: EVAI Persona Fine-Tuning Pipeline

EVAI 공개 페르소나 데이터(`data/*.json`)와 로컬 `LiquidAI/LFM2.5-230M-Base` 베이스 모델(`models/lfm2-230m`)을 활용한 **로컬 페르소나 전체 파라미터(full-parameter) 파인튜닝 파이프라인**입니다.

---

## 1. 프로젝트 목적

본 프로젝트는 정령(페르소나)이 어시스턴트 위에 얹힌 스킨이 아니라 **모델의 기본 출력 정체성 그 자체**가 되도록, LFM2.5-230M-Base의 모든 파라미터(LIV 컨볼루션 블록과 GQA 블록 전 레이어)를 직접 파인튜닝하는 결정론적 파이프라인입니다. 프로즌 베이스에 얹는 LoRA 어댑터 방식이 아니라, 베이스 모델 자체의 가중치를 정령의 성격·말투·감정·사고방식·응답방식으로 직접 갱신합니다.

파이프라인의 전체 라이프사이클:
```
data/*.json
  → 검증 (Schema Validation)
  → 보수적 정규화 (Normalization)
  → 대화 추출 및 오염/콘텐츠 안전 분류 (Dialogue Extraction + Classification)
  → 결정론적 SFT 데이터셋 (Deterministic Dataset)
  → 누수 방지 분할 (Leakage-safe Split)
  → 전체 파라미터 파인튜닝 (Full-Parameter Fine-Tuning, LFM2.5-230M-Base)
  → 학습된 정령 모델 (artifacts/merged/<persona_id>)
  → 고정 회귀 평가 (Fixed Identity-Regression Evaluation)
  → GGUF 변환 및 Ollama 등록
  → 로컬 웹챗 UI로 대화 테스트 (garnet-evai serve)
```

기반 구조(경로 SSOT, 에러 계층, 스키마/로더, 텍스트 정규화, 환경/모델 검사)뿐 아니라, Dataset Builder(대화 추출·오염 분류·SFT 레코드 생성), 전체 파라미터 학습 실행, 고정 회귀 평가, GGUF/Ollama 익스포트, 로컬 웹챗 UI까지 실제 동작하는 파이프라인으로 구현되어 있습니다.

---

## 2. 프로젝트 디렉터리 구조

```text
evai_learning/                   (프로젝트 루트)
├─ data/
│  └─ *.json                   # EVAI 원본 페르소나 JSON (99개, 절대 수정/이동 금지)
│
├─ models/
│  └─ lfm2-230m/               # 로컬 LiquidAI/LFM2.5-230M-Base (safetensors, tokenizer 등)
│
├─ configs/
│  ├─ model.yaml               # 베이스 모델 로컬 경로 및 오프라인 로드 설정
│  └─ training.yaml            # 전체 파라미터 파인튜닝 하이퍼파라미터 (bf16, AdamW, cosine)
│
├─ src/                        # src/<도메인>/<이름>.py, import 루트는 src
│  ├─ common/                  # paths.py(경로), errors.py(예외), device.py(디바이스 선택)
│  ├─ persona/                 # schema.py(페르소나 스키마), loader.py(data/*.json 로더)
│  ├─ sft_dataset/             # normalize, dialogue, records, split, manifest, storage(JSONL 입출력)
│  ├─ inspection/              # runtime_environment.py(CUDA/라이브러리), base_model_assets.py(모델 자산)
│  ├─ inference/               # model_loader.py(모델·토크나이저 로드), generation.py(생성 설정·응답 생성)
│  ├─ training/                # trainer.py(학습 설정 로드·SFT 실행)
│  ├─ adapter/                 # spirit_adapter.py(정령 어댑터 추출·런타임)
│  ├─ evaluation/              # regression.py(고정 회귀 평가)
│  ├─ export/                  # gguf_ollama.py(GGUF 변환·Ollama 등록)
│  ├─ web/                     # server.py, index.html, assets/ (로컬 웹챗)
│  └─ cli/                     # entrypoint.py(garnet-evai, python -m cli), parser.py, *_commands.py(도메인별 명령)
│
├─ tests/                      # test_paths/test_schema/test_loader/test_normalize/
│                               # test_environment/test_model/test_dataset/test_web
│
├─ artifacts/                  # 생성물 저장소 (Git 제외)
│  ├─ datasets/                # 변환된 JSONL 데이터셋 및 manifest
│  ├─ merged/                  # 전체 파라미터 학습 결과 모델(어댑터 병합 단계 없음)
│  ├─ gguf/                    # 양자화 GGUF 파일 및 Ollama Modelfile
│  └─ reports/                 # 평가 리포트
│
├─ pyproject.toml              # 프로젝트 패키지 메타데이터 및 린터/테스터 설정
├─ README.md                   # 프로젝트 문서
└─ .gitignore                  # 로컬 헤비 자산 및 임시 파일 제외 설정
```

---

## 3. 원본 데이터 규약 (`data/*.json`)

1. **단일 원본 경로**:
   - 정규 페르소나 데이터는 오직 `data/*.json`에만 위치합니다.
   - `data/personas/` 같은 하위 폴더는 생성하지 않으며 허용되지 않습니다.
2. **비재귀적 탐색 및 결정론적 정렬**:
   - `src/persona/loader.py`는 `data/` 직하위의 `*.json` 파일만 수집합니다.
   - 데이터셋 입력 처리는 파일명 기준 오름차순으로 엄격하게 정렬됩니다.
3. **원본 불변성 (Immutability)**:
   - 원본 JSON 파일을 수정, 재포맷팅, 이동, 삭제하지 않습니다.
4. **미지 필드 보존**:
   - Pydantic v2 `extra="allow"` 설정을 통해 `i18n` 및 미래에 추가될 수 있는 확장 메타데이터를 누락 없이 보존합니다.
5. **화자 변환 규약**:
   - `구원자` / `Savior` $\rightarrow$ `user`
   - `현재 학습 페르소나` $\rightarrow$ `assistant`
   - 타 캐릭터의 대사나 `comments`는 어시스턴트 completion으로 학습하지 않습니다.

---

## 4. 로컬 베이스 모델 규약 (`models/lfm2-230m`)

1. **로컬 전용 로딩**:
   - 베이스 모델은 사전 다운로드된 `models/lfm2-230m`(LiquidAI/LFM2.5-230M-Base)을 사용합니다.
   - Hugging Face 네트워크 다운로드를 차단하기 위해 `local_files_only=True`를 강제합니다.
2. **오프라인 경량 검사**:
   - 기반 검사 단계에서는 전체 모델 가중치를 VRAM에 불필요하게 인스턴스화하지 않습니다.
   - `AutoConfig`, `AutoTokenizer` 및 `model.safetensors`(약 0.43GB) 파일의 존재와 헤더 무결성만을 점검합니다.
3. **전체 파라미터 학습**:
   - 프로즌 베이스에 얹는 LoRA 어댑터가 아니라, `model_type=lfm2`의 8개 LIV 컨볼루션 블록과 6개 GQA 블록 전체 파라미터를 직접 갱신합니다. 230M 파라미터는 8GB급 GPU에서 bf16 전체 파인튜닝이 가능한 크기입니다.

---

## 5. 시스템 Python 및 환경 정책

- 본 프로젝트는 Windows 11 시스템 Python을 직접 활용합니다.
- `uv`, `venv`, `virtualenv`, `conda`, `Docker` 가상화 환경을 의도적으로 도입하지 않습니다.
- 설치된 PyTorch 런타임이 CUDA GPU를 정상적으로 인식하고 사용할 수 있는지가 유효성 기준입니다(`garnet-evai env`로 확인).

---

## 6. CLI 실행 명령어

인라인 파이썬(`python -c` 등)을 일체 사용하지 않으며, 모든 검사는 정식 CLI 모듈로 실행됩니다.

### 런타임 및 GPU 환경 검사
```powershell
garnet-evai env
```
Python/PyTorch 버전, CUDA 런타임, GPU 감지 상태 및 필수 패키지(transformers, datasets, accelerate, trl, safetensors) 설치 상태를 점검합니다.

### 페르소나 데이터 전수 검사
```powershell
garnet-evai data
```
`data/*.json` 내의 99개 페르소나 파일을 전수 로드하고, 스키마 유효성, 스토리/에버톡/화법 개수를 사실에 기반하여 집계합니다.

### 로컬 베이스 모델 자산 검사
```powershell
garnet-evai model
```
VRAM 가중치 로드 없이 `models/lfm2-230m`의 설정, 어휘 사전(65,536), 히든 레이어 크기(1024), 레이어 수(14), 토크나이저 및 `model.safetensors` 가중치 파일의 유효성을 검증합니다.

### 종합 기반 구조 검증
```powershell
garnet-evai check
```

### 결정론적 SFT 데이터셋 생성
```powershell
garnet-evai build-dataset
```
99개 페르소나 전체에 대해 화자 매핑, 어시스턴트 오염/콘텐츠 안전 분류, leakage-safe split을 거쳐 `artifacts/datasets/<persona_id>/{train,validation,test,exclusions}.jsonl`과 `manifest.json`을 생성합니다.

### 전체 파라미터 학습
```powershell
garnet-evai train <persona_id>
```
해당 정령의 SFT 데이터셋으로 베이스 모델 전체 파라미터를 직접 파인튜닝하고, 결과를 `artifacts/merged/<persona_id>`에 저장합니다.

### 고정 회귀 평가
```powershell
garnet-evai evaluate <persona_id>
```
plan.md §14 기반 12개 카테고리(정체성/어시스턴트 오염/말투/행동/감정/관계/메타롤플레이/도구환각/일반거부/추론/한국어품질/정령간교차오염) 고정 프롬프트로 학습된 모델을 채점하고 `artifacts/reports/<persona_id>_evaluation.json`을 생성합니다.

### GGUF 변환 및 Ollama 등록
```powershell
garnet-evai export <persona_id>
```
llama.cpp(`convert_hf_to_gguf.py`, `llama-quantize`)와 Ollama가 PATH에 있어야 합니다(외부 도구, 자동 설치하지 않음).

### 로컬 웹챗 UI
```powershell
garnet-evai serve
```
브라우저에서 정령별 대화 테스트가 가능한 로컬 HTTP 서버를 실행합니다.

---

## 7. 테스트 및 정적 품질 분석

### 단위 및 통합 테스트 실행
```powershell
python -m pytest
```

### 정적 린트 분석 (Ruff)
```powershell
ruff check .
```

### 타입 검사 (Pyright)
```powershell
pyright
```

### 패키지 의존성 무결성 검사
```powershell
python -m pip check
```

---

## 8. 엄격한 엔지니어링 및 경고 정책

1. **경고 억제 절대 금지 (ZERO Suppressions)**:
   - `warnings.filterwarnings("ignore")`, `-W ignore`, `# noqa`, `# type: ignore`, `# pyright: ignore` 사용을 전면 금지합니다.
   - 모든 경고와 진단은 결함으로 취급하며 근본 원인을 추적하여 수정합니다.
2. **근본 원인 해결 (Root-Cause Fixes Only)**:
   - try-except 빈값 리턴, 임의의 기본값 주입, 테스트 기댓값 하향 조정 등 증상 은폐성 패치를 금지합니다.
3. **가짜 데이터/가짜 구현 금지**:
   - TODO, stub, mock, placeholder 및 인위적인 가짜 페르소나 데이터를 생성하지 않습니다.
