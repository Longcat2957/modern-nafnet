# AGENTS.md — NAFNet Deblurring

## 목표와 범위

- NAFNet-width32 기반의 독립적인 Deblurring 리포를 만든다. 공식 모델 코드·학습 설정을 참조 기준으로 삼고, 현재 안정판 PyTorch의 표준 연산/API로 모델을 재이식한다. 공식 체크포인트를 재사용하며 학습·평가·배포 코드는 새로 구성한다.
- 프로젝트 자체 코드는 **AGPL-3.0-only 공개**를 기준으로 설계한다. 초기화 시 `LICENSE`와 `pyproject.toml`의 license metadata에 반영하며, 외부 원저작물의 MIT/Apache 2.0 라이선스·저작권 고지는 별도로 보존한다. 외부 체크포인트의 출처·라이선스도 프로젝트 코드와 구분해 기록한다.
- 최종 목표는 Jetson AGX Orin의 TensorRT W8A8 추론이다. 목표 해상도·FPS·전력 모드는 아직 미정이며 설정으로 둔다. 실기기 측정 전에는 실시간 성능을 확정하지 않는다.
- 간단한 구성과 CLI를 우선한다. 다른 복원 리포와 공통 패키지나 상호 의존성을 만들지 않는다.
- 아래 스캐폴드는 단계별로 구현할 목표 구조다. 필요한 모듈만 만들고 빈 구현·추상 base class·registry·DI framework를 미리 만들지 않는다.

## 도구와 환경

- **Python 패키지·가상환경·실행은 반드시 [uv](https://docs.astral.sh/uv/guides/projects/)로 관리한다.** 의존성 변경은 `uv add`/`uv remove`, 환경 재현은 `uv sync --locked`, 실행·테스트는 `uv run --locked`를 사용한다. `pyproject.toml`, `uv.lock`, `.python-version`을 커밋하고 lockfile은 직접 편집하지 않는다.
- PyTorch CUDA wheel과 NVIDIA 공식 Python 패키지의 인덱스·소스·플랫폼 조건은 `pyproject.toml`에 명시한다. 개별 `pip install`로 우회하지 않는다. Orin의 JetPack CUDA·TensorRT 네이티브 SDK는 시스템 설치를 사용한다. Orin Python runner가 필요한 경우에만 uv 환경에서 호환 바인딩과 CUDA 동작을 확인한다.
- 구현 시점의 **최신 안정판 PyTorch와 NVIDIA ModelOpt**를 우선한다. nightly는 사용하지 않는다. CUDA·ModelOpt·ONNX exporter 호환성을 확인한 뒤 검증된 버전을 고정한다.
- 기본 배포 경로는 **x86 CUDA 서버/워크스테이션에서 학습·calibration·QAT → ONNX export → ONNX를 Orin으로 전달 → 대상 Orin에서 TensorRT 엔진 build → 엔진 infer/bench**다. x86의 `export`와 Orin의 `build/infer/bench`를 분리한다. Jetson PyTorch는 추가 비교가 필요한 경우에만 NVIDIA 공식 호환 패키지·컨테이너를 검토한다.
- 전달 산출물은 ONNX(외부 가중치 파일이 있으면 함께), 입력 shape/dtype·RGB/range·padding/crop 등의 전후처리 설정, hash·버전 manifest다. INT8 calibration/QAT의 scale은 Q/DQ ONNX에 포함해 전달한다. ONNX opset·연산·정밀도는 대상 Orin의 TensorRT 지원 범위에 맞추며, x86에서 빌드한 engine의 Orin 이식성을 가정하지 않는다. [TensorRT 엔진 이식성](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html)
- 이 리포의 모델 재이식·학습·calibration·QAT·ONNX export와 공식 품질 평가는 x86에서 실행한다. 전체 소스·Python 패키지의 Orin 설치·실행 호환성은 필수 범위가 아니다. Orin에는 ONNX·전후처리 설정·manifest를 전달하고, 초기 엔진 build·레이어 분석은 시스템 `trtexec`를 우선 사용한다. 성능 측정은 모델 학습/QAT 이후의 별도 단계로 둔다. [TensorRT 명령행 도구](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/command-line-programs.html)
- 실제 이미지 출력 비교·end-to-end 측정·제품 연동에 필요한 경우에만 최소 TensorRT runner를 추가한다. 기존 제품 runtime이 있으면 재사용하며, Python을 선택한 경우에만 `trt` extra와 aarch64 바인딩·platform marker·source를 검증한다. 이 runner는 PyTorch·ModelOpt·Albumentations·TorchMetrics를 요구하지 않아야 한다.
- `uv.lock`은 Python 의존성을 관리한다. JetPack/L4T·CUDA·cuDNN·TensorRT·Python ABI·GPU SKU·드라이버는 별도 환경 manifest에 기록한다. `.python-version`의 학습 호스트 기본값을 Orin에 강제하지 않으며, 시스템 바인딩을 노출하는 것만으로 lock 재현이 완료됐다고 판단하지 않는다.
- 2026-10-12 확인 기준: 최신 JetPack 7.2.1은 TensorRT 10.16.2를 포함한다. 최신 TensorRT 11.4는 JetPack과 Jetson DLA를 지원하지 않는다. 배포 시 [공식 JetPack 구성](https://developer.nvidia.com/embedded/jetpack/downloads)과 [TensorRT 지원 안내](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html)를 다시 확인한다.
- 275 TOPS는 AGX Orin 64GB MAXN에서 GPU와 두 DLA의 sparse INT8 피크 합계다. QAT만으로 달성한다고 가정하지 않는다. GPU INT8을 먼저 검증하고 2:4 sparsity·DLA는 효과가 확인될 때 추가한다.

## 계층화한 최소 프로젝트 스캐폴드

리포별로 아래 구조를 만들고 필요한 모듈만 구현한다. `uv init --package --name drone-deblur`를 출발점으로 삼으며, 기존 파일은 보존한다.

```text
.
├── AGENTS.md
├── README.md
├── LICENSE                   # 프로젝트 자체 코드: AGPL-3.0-only
├── THIRD_PARTY_NOTICES.md
├── licenses/                 # 참조·수정·포함한 upstream의 원래 라이선스
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── configs/default.json
├── src/drone_deblur/
│   ├── __init__.py
│   ├── cli.py                # Typer app·얇은 명령·main()
│   ├── config.py             # stdlib JSON 설정·검증·경로 처리
│   ├── artifacts.py          # stdlib 실행 metadata·hash·결과 JSON
│   ├── models/
│   │   ├── __init__.py
│   │   ├── nafnet.py         # 구조·가중치 호환 NAFNet/NAFNetLocal 재이식
│   │   ├── layers.py         # 표준 PyTorch 연산의 동등 LayerNorm2d
│   │   └── tlc.py            # 원본 동작을 보존하는 local pooling 재이식
│   ├── data/
│   │   ├── __init__.py
│   │   ├── images.py         # NumPy/OpenCV RGB 입출력·전후처리
│   │   ├── augment.py        # Albumentations paired Compose·seed 설정
│   │   └── paired.py         # PyTorch Dataset·transform 호출·tensor 변환
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loop.py           # FP32 fine-tune/QAT 공통 루프·학습 상태
│   │   ├── losses.py         # 원본 수식을 유지한 PSNRLoss 재구현
│   │   └── quantize.py       # ModelOpt calibration·정책·저장/복원
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── export.py         # PyTorch → FP/QDQ ONNX
│   │   └── tensorrt.py       # 필요 시 최소 Python runner; 초기 build는 trtexec
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py        # 공식 PSNR/SSIM·선택적 보조 지표
│       ├── quality.py        # 동일 데이터·전처리의 backend 품질 평가
│       └── benchmark.py      # engine/end-to-end 측정
└── tests/
```

- 각 폴더는 책임과 import 경계다. 모델·데이터 계층은 CLI·학습·배포 workflow를 import하지 않는다. 평가의 backend 연결은 `quality.py`에 두고 모델을 수정하지 않는다. 공통 코드를 거대한 `utils/`에 모으지 않는다.
- `cli.py`는 Typer app·설정·명령 dispatch만 담당하고, 선택한 명령의 구현을 지연 import한다. 모든 `__init__.py`는 가볍게 유지하며 모델/SDK를 재노출하지 않는다. 패키지 import, 루트와 각 하위 명령의 `--help`가 CUDA·TensorRT·학습용 extra를 요구하지 않아야 한다.
- `images.py`, 설정·metadata, TensorRT 실행 경로에는 torch import를 넣지 않는다. PyTorch tensor 변환은 `paired.py` 또는 PyTorch workflow에서 수행한다. `deployment/export.py`의 torch 의존성이 TensorRT runtime으로 전파되지 않게 한다.
- Albumentations import와 학습용 Compose 구성은 `data/augment.py`에 둔다. `images.py`, 패키지 import, CLI help, TensorRT runtime은 증강 패키지를 요구하지 않는다.
- `training/loop.py`는 준비된 모델을 학습하며 ModelOpt를 직접 import하지 않는다. calibration·양자화 변환은 `quantize.py`에서 준비해 FP32 학습이 QAT extra를 요구하지 않게 한다.
- `pyproject.toml`의 CLI entry point는 `drone-deblur = "drone_deblur.cli:main"`으로 둔다. `main()`은 Typer app을 호출하며 `@app.command()`로 x86의 `train/qat/export/infer/eval`을 필요한 순서대로 등록한다. Orin은 우선 네이티브 `trtexec` 명령을 README에 남기며, 자체 runner가 필요한 경우에만 `build/infer/bench`를 확장한다. 루트 callback을 두어 명령이 하나인 초기 단계에도 `drone-deblur infer` 형태를 유지한다. `uv run --locked drone-deblur --help`로 진입한다.
- Typer의 type hint·`Annotated`·`Path` 등으로 인자/옵션과 기본 경로 검증을 선언한다. 명령 함수는 CLI 타입을 실행 설정으로 연결하는 얇은 경계로 유지하며 모델·학습·평가 함수에는 Typer 의존성을 전파하지 않는다. `no_args_is_help=True`, `pretty_exceptions_show_locals=False`를 사용한다. SDK import·설정 읽기·데이터 확인은 실제 명령 실행에서 수행한다.
- `configs/default.json`에 데이터·모델·학습·양자화·export·runtime·평가·benchmark 설정을 둔다. JSON을 override하는 CLI 옵션의 미지정 기본값은 `None`으로 두고 `None`이 아닌 값만 병합한다. `bool | None`으로 미지정과 명시적 `False`를 구분하고 `0`/`False`를 truthiness로 누락하지 않는다. 상대 경로의 기준을 문서화하며 해상도·FPS·전력 모드의 미정 값은 `null`로 둔다.
- 개발 도구는 uv의 `dev` dependency group으로 관리한다. `.gitignore`에는 `/.venv/`, `/data/`, `/artifacts/`로 저장소 루트의 작업 산출물만 제외한다. `data/` 패턴으로 `src/drone_deblur/data/` 패키지까지 제외하지 않는다. 테스트 파일은 해당 단계의 검증에 필요한 경우에만 추가한다.

## 라이브러리 선정과 의존성 프로필

2026-10-13 공식 release/PyPI metadata 확인 기준이며, 아래 버전은 후보이지 실행 검증을 마친 pin이 아니다. 구현 시 안정판·wheel·CUDA/export 호환성을 다시 확인하고 uv로 고정한다.

| 라이브러리 | 확인한 후보 | 초기 선택 |
|---|---|---|
| `numpy` + `opencv-python-headless` | OpenCV headless 5.0.0.93, Python >=3.9에서 NumPy >=2 | 기본 이미지 입출력·전후처리; 학습 증강은 Albumentations 사용 |
| `typer` | 0.27.3, MIT, Python >=3.10 | 기본 CLI command·typed option·help; Rich 통합 |
| `rich` | 15.0.0, MIT, Python >=3.9 | 기본 CLI progress·logging·결과 표 |
| `torchmetrics` | 1.9.0, Apache-2.0, Python >=3.10 | 학습 호스트의 선택적 보조 지표; 공식 SSIM 대체 금지 |
| legacy `albumentations` | 2.0.8, MIT, 공식 리포 archived | 유지보수되는 X를 선택하며 같은 import namespace의 동시 설치 금지 |
| `albumentationsx[headless]` | 2.5.1, AGPL-3.0-only, Python >=3.11·NumPy >=2.2.6 | **학습/QAT 필수**; import는 `albumentations as A` |

- 기본 실행 의존성은 NumPy·OpenCV headless·Typer·Rich로 둔다. 선택 프로필은 `train`(torch·albumentationsx[headless]), `qat`(torch·nvidia-modelopt·albumentationsx[headless]), `export`(torch·ONNX 및 선택 exporter가 요구하는 패키지), `metrics`(torchmetrics)로 구분한다. `trt`(대상 SDK 호환 TensorRT/CUDA 바인딩)는 Python runner를 구현할 때만 추가하고, 초기에는 전체 프로젝트의 aarch64 의존성 해결을 요구하지 않는다. Albumentations는 train/QAT 내부에서 필수이며 NumPy 자체 증강으로 우회하지 않는다. `pytest`·`ruff`는 `dev` group에 둔다. 예: `uv run --locked --extra train drone-deblur train ...`.
- Typer와 직접 사용하는 Rich를 프로젝트 의존성으로 선언한다. Typer 자체 command/testing API를 사용하고 외부 Click이나 Typer 내부 `_click`을 직접 사용하는 의존성을 추가하지 않는다. 선택 extra 누락은 실행 시 필요한 프로필과 명령을 안내하고 nonzero 종료한다. [Typer metadata](https://pypi.org/project/typer/), [command/callback](https://typer.tiangolo.com/tutorial/commands/callback/)
- 부동소수점 ONNX export는 `--extra export`, ModelOpt 구조·quantizer 상태를 복원하는 Q/DQ export는 `--extra export --extra qat`로 실행한다. 부동소수점 export가 ModelOpt를 import하지 않게 한다.
- TorchMetrics의 PSNR/SSIM에는 PyTorch Lightning과 `torchmetrics[image]`/torchvision을 자동 도입하지 않는다. 설정 처리는 stdlib로 시작한다. ModelOpt 자체의 전이 의존성과 프로젝트가 직접 사용하는 의존성을 구분하고, 필요 없이 ModelOpt `[all]`/`[hf]`를 설치하지 않는다.
- OpenCV 배포 패키지는 `opencv-python-headless` 하나만 사용한다. 다른 `opencv-python*`/`opencv-contrib-python*`와 `cv2` namespace를 공유하므로 중복 설치를 금지한다. Orin Python runner에서 OpenCV가 필요하면 aarch64 wheel의 ABI/glibc와 NumPy 호환성을 확인하고, 예상치 못한 source build는 허용하지 않는다. PyPI headless wheel을 CUDA OpenCV 빌드로 가정하지 않는다.
- `albumentationsx[headless]`와 직접 사용하는 OpenCV headless의 요구 버전을 uv가 하나의 배포 패키지로 해결하게 한다. legacy `albumentations`를 함께 설치하지 않는다. 설치명은 `albumentationsx`, 사용 API는 `import albumentations as A`다. [공식 설치](https://albumentations.ai/docs/1-introduction/installation/), [2.5.1 의존성](https://github.com/albumentations-team/AlbumentationsX/blob/2.5.1/pyproject.toml)
- 이미지 읽기는 BGR → RGB를 명시한다. 학습 paired geometry는 `A.Compose(..., additional_targets={"sharp": "image"})` 한 번에 `image=blur, sharp=sharp`를 전달한다. 기본은 square `RandomCrop` 뒤 `A.D4(p=1)`로 원본 flip/transpose의 8개 방향 균등 분포를 유지한다. 별도 호출로 blur/sharp에 각각 난수를 적용하지 않는다.
- 공통 geometry, 공통 photometric, blur 입력에만 적용하는 degradation을 구분한다. 실제 paired 데이터는 원본의 기하 증강만 사용한다. `data.synthetic_blur.enabled`/`--synthetic-blur`로 sharp-only 학습 항목에 한해 shared crop/D4 후 별도 AlbumentationsX MotionBlur를 입력에만 적용한다. GT는 blur하지 않으며 실제 paired 항목을 추가로 흐리게 하지 않는다. RGB sharp를 mask로 매핑하지 않으며 calibration/평가에는 학습용 무작위 증강을 적용하지 않는다.
- 합성 blur는 uint8 crop에서 처리해 원해상도 blur 비용을 피하고, 마지막에 float `[0,1]`로 변환한다. sharp-only 이미지는 한 번만 읽으며 쌍을 디스크에 복제하지 않는다. 다중 DataLoader worker에서는 OpenCV 내부 스레드·OpenCL을 꺼 중첩 병렬화를 피한다. worker 수는 설정으로 조절하며 CPU 처리량·합성 분포의 학습 효과는 추후 실측한다. geometry와 blur의 독립 RNG를 저장·복원하고 기존 paired checkpoint 재개 호환성을 유지한다.
- `Compose(seed=...)`와 DataLoader 전용 `torch.Generator`의 seed를 설정하고 worker 수·persistent_workers·sampler·batch/drop_last·데이터 순서를 기록한다. AlbumentationsX의 worker별 seed 처리를 사용하고 매 sample/worker를 같은 seed로 재설정하지 않는다. Compose의 난수는 NumPy 전역 seed만으로 제어되지 않는다. [재현성 지침](https://albumentations.ai/docs/4-advanced-guides/reproducibility/)
- `ToTensorV2`는 HWC → CHW 변환을 담당하며 `[0,1]` 정규화를 대신하지 않는다. RGB float range 변환을 한 번 명시하고 ImageNet mean/std 정규화를 적용하지 않는다. 배열은 contiguous를 보장한다. 이를 위해 torchvision을 추가하지 않는다. [2.5.1 ToTensorV2](https://github.com/albumentations-team/AlbumentationsX/blob/2.5.1/albumentations/pytorch/transforms.py)
- Rich는 stderr에 사람용 progress/logging을 출력하고 정상 실행의 stdout에는 기계가 읽는 결과만 둔다. Typer help는 표준 help 출력으로 유지한다. `Console(stderr=True)`와 `Progress(..., redirect_stdout=False)`를 사용한다. 비 TTY에서는 live progress를 비활성화하고 benchmark 측정 구간 안에서 출력·refresh·추가 GPU 동기화를 하지 않는다. Typer CLI와 Rich를 통합하고 별도 TUI framework를 추가하지 않는다.
- 라이브러리 근거: [OpenCV 패키지·wheel](https://pypi.org/project/opencv-python-headless/), [Rich metadata](https://pypi.org/project/rich/), [Rich Progress](https://rich.readthedocs.io/en/stable/progress.html), [TorchMetrics release](https://github.com/Lightning-AI/torchmetrics/releases/tag/v1.9.0), [legacy Albumentations](https://github.com/albumentations-team/albumentations), [AlbumentationsX metadata](https://pypi.org/project/albumentationsx/), [additional targets](https://albumentations.ai/docs/4-advanced-guides/additional-targets/), [ModelOpt 패키징](https://github.com/NVIDIA/Model-Optimizer/blob/0.47.0/pyproject.toml).

## 데이터 계약과 실행 기록

- calibration에는 blur-only 데이터를 사용할 수 있다. supervised 학습/QAT는 실제 정렬된 blur/sharp 쌍 또는 옵션으로 활성화한 sharp-only 합성 쌍을 사용한다. 합성 manifest는 `sharp` 필수·`blur` 생략으로 표시하며 실제 paired 항목과 혼합할 수 있다. 명시한 blur 경로가 잘못된 경우에는 합성으로 대체하지 않는다. PSNR/SSIM 품질 평가는 실제 정렬된 blur/sharp 쌍을 요구한다. 파일 쌍·크기·RGB/range·읽기 실패를 검증하고 누락 샘플을 조용히 건너뛰지 않는다.
- 촬영 시퀀스·장소·날짜를 기준으로 train/calibration/validation/test manifest를 고정한다. 같은 영상의 인접 프레임을 학습과 최종 평가에 나누지 않으며, 최종 test는 calibration·튜닝에 사용하지 않는다.
- 드론 정답 쌍이 없다면 paired GoPro로 QAT하고 드론 데이터는 calibration·시각·추론 평가에 사용한다. 정답 없는 드론 영상의 PSNR/SSIM을 보고하지 않는다.
- 각 실행은 `artifacts/<run>/`에 적용된 설정, upstream commit·checkpoint SHA-256, 데이터 manifest 식별자, 패키지·장비·SDK 버전, 입력 shape/dtype, 평가 protocol, 결과 JSON을 남긴다. ONNX·엔진 hash와 builder/typing 설정도 기록한다. 외부 실험 추적 서버는 초기 범위에 넣지 않는다.

## 모델 기준

- 출처: [공식 NAFNet 리포](https://github.com/megvii-research/NAFNet). 가져온 upstream commit과 체크포인트 SHA-256을 기록하고 원본 라이선스·저작권 표시를 보존한다. NAFNet은 MIT, 포함된 BasicSR 부분은 Apache 2.0이다.
- 공개 GoPro 기준은 **32.8705 dB / SSIM 0.9606**이다. 이는 원본 부동소수점 모델 수치이며 INT8 결과가 아니다.
- 공식 width32 설정을 유지한다: `img_channel=3`, `width=32`, `enc_blk_nums=[1,1,1,28]`, `middle_blk_num=1`, `dec_blk_nums=[1,1,1,1]`.
- 공개 학습·평가 설정의 `NAFNetLocal`/TLC를 보존한다. `train_size=(1,3,256,256)`, `fast_imp=False`를 기준으로 하고, RGB·NCHW·입력 범위 `[0,1]`, 원본의 16 배수 zero padding과 출력 crop을 유지한다.
- TLC kernel은 원본 생성자의 `train_size` dummy forward로 초기화한 값을 유지한다. 배포 입력으로 최초 kernel을 산정하거나 다시 초기화하지 않는다. 초기화 → checkpoint strict load → 배포 크기 warm-up → calibration/export 순서를 고정하고 pool별 kernel을 기록한다.
- 256 patch의 global pooling과 큰 입력의 local `cumsum`/replicate padding 경로를 각각 검증한다. 작은 patch에서 export된 모델만으로 큰 해상도의 TLC export 성공을 판정하지 않는다.
- LayerNorm2d는 **채널 축 통계**, `eps=1e-6`이다. export를 위한 동등한 PyTorch 연산 치환은 허용하되 출력과 gradient를 비교한다. BatchNorm으로 교체하거나 TLC를 제거하지 않는다.
- LayerNorm 비교에는 입력·affine weight/bias gradient를 포함하고 TLC 치환에는 입력 gradient·padding 경계를 포함한다. dtype/shape별 허용오차와 max absolute/relative error·finite 여부를 기록한다. FP32 기준, near-constant 입력, 배포 크기를 확인한 뒤 FP16 정책을 적용한다.

## 현대 PyTorch 재이식 원칙

- production 모델은 공식 NAFNet/BasicSR 파일을 통째로 복사하거나 BasicSR을 runtime 의존성으로 가져오지 않는다. 구조·연산 순서·학습 수식·TLC 동작을 참조해 현재 안정판 PyTorch의 표준 연산으로 구현한다. 참조 upstream commit과 파일별 변경 이유는 `THIRD_PARTY_NOTICES.md`에 기록하며 재이식에서도 출처·원래 고지를 보존한다.
- 공식 upstream snapshot은 개발 검증용 reference로 분리해 사용하고 production 모델이 이를 import하지 않게 한다. 공식 checkpoint는 새 모델에 `strict=True`로 load한다. 모듈/parameter 이름과 shape를 유지하고 편의를 위한 키 재매핑·`strict=False`로 구조 차이를 숨기지 않는다.
- LayerNorm2d는 채널 mean·biased variance(`correction=0`)의 표준 autograd 연산 또는 NCHW → NHWC → `F.layer_norm(..., (C,), weight, bias, eps=1e-6)` → NCHW로 표현할 수 있다. NCHW에 `nn.LayerNorm(C)`를 직접 적용하지 않는다. 원본 forward와 입력/weight/bias gradient 동등성 및 export를 확인해 최소 구현을 선택한다.
- TLC의 누적·slice·replicate padding은 표준 PyTorch 연산으로 재표현하고 필요한 경우 비제자리 `cumsum`을 사용한다. 원본 kernel 초기화·분기·출력 crop은 유지한다. custom autograd라는 이유만으로 원본이 비호환이라고 단정하지 않으며, 표준 연산 경로가 검증을 통과하면 production에서 별도 backward 구현을 두지 않는다.
- FP32 동등성을 먼저 확보한다. 혼합정밀도 학습은 현재 `torch.amp.autocast(device_type="cuda", ...)`와 `torch.amp.GradScaler("cuda", ...)`를 사용하고 LayerNorm 통계·TLC 누적·손실의 dtype을 확인한다. inference는 `model.eval()`과 `torch.inference_mode()`를 사용한다.
- 부동소수점 ONNX는 현재 `torch.onnx.export(..., dynamo=True)` 경로를 우선 검증한다. ModelOpt INT8 Q/DQ는 해당 버전 quantizer/exporter의 지원을 별도로 확인한다. legacy exporter가 필요한 경우 `dynamo=False`를 명시하고 버전·이유·Q/DQ 및 출력 검증을 기록한다. 제거된 `fallback` 인자를 사용하거나 Q/DQ가 누락된 export를 성공으로 처리하지 않는다.
- `torch.compile`, channels-last, 연산 fusion은 재이식 완료 조건으로 강제하지 않는다. 원본 동등성·QAT·export·Orin 기준선이 확보된 뒤 실측 병목에 대해서만 적용한다. 모델 폭·블록 수·SimpleGate·attention·beta/gamma·TLC 변경은 문법 현대화에 포함하지 않는다.
- 현대 API 근거: [PyTorch LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html), [AMP](https://docs.pytorch.org/docs/2.14/amp.html), [2.14.1 ONNX exporter](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/onnx/__init__.py), [ModelOpt 0.47.0 INT8 symbolic](https://github.com/NVIDIA/Model-Optimizer/blob/0.47.0/modelopt/torch/quantization/tensor_quant.py), [ModelOpt export 검증 helper](https://github.com/NVIDIA/Model-Optimizer/blob/0.47.0/tests/_test_utils/torch/quantization/onnx_export.py).

## INT8/QAT 원칙

- [NVIDIA ModelOpt 공식 QAT 경로](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)를 사용한다: `INT8_DEFAULT_CFG` 기반 calibration → QAT → ONNX Q/DQ → TensorRT.
- static symmetric INT8, Conv weight per-channel·activation per-tensor를 출발점으로 삼는다. 실제 모듈에 quantizer가 삽입되었는지 확인하고 residual add·곱셈 경계는 별도로 검토한다.
- 주요 Conv는 W8A8을 목표로 한다. LayerNorm·SimpleGate·attention·beta/gamma 곱셈은 처음에는 FP16/FP32로 유지하고 품질·시간 측정 후 범위를 조정한다. TLC의 `cumsum` 누적은 우선 FP32로 처리한다.
- attention의 `sca.1` 1×1 Conv도 기본 Conv 변환에 포함될 수 있으므로 초기 quantizer 제외 정책에 `*.sca.*`를 명시한다. 주요 Conv의 input/weight quantizer, add·곱셈 주변 Q/DQ를 확인하며 기본 output quantizer 동작을 실제 summary와 그래프로 검증한다.
- calibration은 실제 배포 전처리·고정 입력 크기에서 수행한다. calibration 후 quantizer scale을 고정하는 QAT를 출발점으로 두며, 다시 calibration한다면 시점·데이터·결과를 기록한다.
- 실제 주간·야간·블러 장면으로 calibration하고 평가 데이터는 분리한다. QAT는 원본 PSNRLoss를 출발점으로 하며, calibration 상태와 학습 가중치를 함께 저장한다.
- 저장은 ModelOpt의 `mto.save/restore` 또는 `modelopt_state + state_dict` 경로로 구조·quantizer·가중치를 함께 복원한다. 학습 재개용 optimizer/scheduler/step/RNG와 TLC/config metadata도 저장하고 저장 전후 출력·상태 및 재개 backward를 검증한다. Albumentations의 독립 난수 상태와 DataLoader generator 상태를 포함해 실제 복원 가능한 재현 범위를 명시한다. prefetch/worker 상태를 복원하지 않고 임의의 중간 step에서 동일한 데이터·증강이 재현된다고 보장하지 않는다.
- 최종 ONNX는 `QuantizeLinear`/`DequantizeLinear`를 포함해야 한다. 초기에는 batch 1과 고정 입력 크기를 사용한다. ONNX의 존재나 INT8 가중치 파일만으로 배포 완료를 판단하지 않는다.
- TensorRT Q/DQ는 explicit quantization 경로를 사용한다. typing 정책을 기록하며 strong typing을 선택한 경우 FP16/INT8 builder flag·layer precision 지정을 혼용하지 않는다. FP16/FP32 구간과 FP32 TLC 누적은 ONNX 타입·Cast 및 출력 비교로 검증한다. [TensorRT 정밀도 정책](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html#strongly-typed-networks)
- 2:4 pruning은 dense INT8 검증 후 진행한다. 해당 조건을 만족하는 일반 Conv에만 적용하고 depthwise Conv의 가속을 가정하지 않는다. DLA는 LayerNorm·TLC 제약과 전환 비용 때문에 후속 실험으로 둔다.

## 구현 순서

1. uv 프로젝트와 필요한 스캐폴드를 구성한다. x86/Orin 환경·데이터 계약을 기록하고 CUDA 및 작은 calibration → backward → Q/DQ export smoke test로 패키지 후보를 검증한다. 대상 TensorRT parser/build 호환성은 Orin에서 확인한 뒤 배포 조합을 확정한다.
2. 공식 모델을 참조해 현대 PyTorch로 재이식하고 원래 출처·라이선스 고지를 남긴다. checkpoint strict load, 원본/재이식 모델의 forward·gradient 동등성을 확인한다. 먼저 `infer/eval`을 구현하고 공식 GoPro FP32 평가 조건과 품질을 재현한다.
3. x86의 `export`와 Orin의 `trtexec`로 ONNX/TensorRT FP16 엔진 빌드와 출력 동등성을 확인한다. 실제 이미지 출력 비교에는 필요한 최소 runner를 사용하며 실제 입력의 TLC/LN·엔진 출력을 검증한다. 이 단계에서는 지연·FPS·전력·end-to-end 성능을 측정하지 않는다. TLC export가 막히면 동등한 연산 표현을 먼저 검증한다.
4. ModelOpt calibration/PTQ의 품질과 Q/DQ 엔진 실행을 확인한다. 엔진 inspector로 INT8 Conv와 floating 구간을 기록한다. 긴 QAT 전에 실제 배포 경로가 실행되는지 확인한다.
5. Albumentations paired pipeline과 공통 학습 루프·`qat`을 구현해 품질을 회복한다. 증강의 정렬·seed·RGB/range를 검증한다. 드론 도메인 추가 학습이 필요하면 FP32 대조군을 확보하고 같은 평가 protocol로 비교한다. `train` 명령은 공통 루프를 통해 필요한 fine-tune에 사용한다.
6. 모델 학습/QAT 이후 별도 평가 단계에서 Orin의 최종 품질·지연·전력·온도와 end-to-end 성능을 측정한다. 해상도·P95 지연/FPS·전력 모드·메모리 목표를 실측에 따라 구체화한다. 병목이 확인된 부분만 최적화하고 실행 명령과 결과를 README에 남긴다. 데이터·Orin이 없는 단계는 미검증으로 두고 독립적인 구현은 계속할 수 있다.

## 검증과 완료 기준

- 같은 GoPro 평가 조건을 사용한다: RGB, border crop 0, Y-channel 평가 아님. PyTorch FP32·TensorRT FP16·QAT INT8의 PSNR/SSIM을 함께 기록한다. 초기 품질 목표는 FP32 대비 PSNR 하락 **0.3 dB 이내**이며 사용자가 조정할 수 있다.
- 공식 재현 평가는 upstream `use_image=True` 경로의 출력·GT `[0,1]` clip → ×255 → round → uint8 변환과 이미지별 지표 산술 평균을 보존한다. 공식 원해상도 평가와 배포 크기 평가를 구분하고, 동일 비교 안에서는 resize/crop·전후처리를 같게 한다.
- 공식 SSIM은 upstream 기본 `ssim3d=True`의 채널축을 포함한 11×11×11 Gaussian Conv3D·replicate padding을 보존한다. TorchMetrics의 일반 RGB 2D/grouped SSIM·reflect padding으로 교체하지 않는다. 보조 지표는 `ssim_torchmetrics` 등 이름·range·clipping을 구분해 기록한다. [공식 metric](https://github.com/megvii-research/NAFNet/blob/main/basicsr/metrics/psnr_ssim.py)
- TorchMetrics PSNR의 기본 `dim=None`은 전체 MSE로 집계하므로 공식 이미지별 dB 평균과 다르다. NCHW `dim=(1,2,3)` 또는 이미지별 functional 계산·sum/count를 사용하고 `data_range`를 명시한다. float range/clipping 설정만으로 uint8 round가 적용됐다고 판단하지 않는다. [TorchMetrics PSNR](https://github.com/Lightning-AI/torchmetrics/blob/v1.9.0/src/torchmetrics/image/psnr.py)
- 원본 SSIM3d는 torch/CUDA에 의존한다. 초기에는 Orin TensorRT 출력과 GT를 보관해 x86 공식 evaluator로 평가할 수 있다. Orin의 torch 없는 로컬 품질 평가가 필요하면 NumPy/OpenCV CPU 동등 구현을 upstream과 비교한 뒤 사용한다. 해당 port의 수치 일치·성능은 별도 검증 항목이며 완료된 것으로 간주하지 않는다.
- 공식 checkpoint FP32, calibration 직후 PTQ, QAT 및 해당 TensorRT 결과의 출처를 구분한다. 도메인 fine-tune이 있다면 양자화 출발 FP32와 같은 데이터/학습량의 FP32 대조군을 명시해 도메인 개선과 양자화 손실을 구분한다.
- TensorRT engine inspector/레이어 프로파일로 실제 INT8 Conv, 부동소수점 구간, 정밀도 변환 비용을 확인한다. Q/DQ 노드 수를 INT8 실행 비율로 대신하지 않는다.
- warm-up과 CUDA 동기화 또는 CUDA event를 사용해 batch 1의 engine 시간과 전처리·전송·후처리를 포함한 end-to-end 시간을 구분한다. P50/P95, FPS, 메모리, 전력 모드·온도·버전을 기록한다.
- `trtexec`의 pipelined throughput·GPU compute time은 엔진 지표로 기록한다. 이를 이미지 전처리·후처리가 포함된 application FPS나 end-to-end 지연으로 보고하지 않는다. 실제 Orin 출력의 품질 검증과 전체 처리 경로의 측정은 별도로 수행한다. [trtexec 측정 정의](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/command-line-programs.html)
- 필요한 작은 검증만 둔다: 선택 extra 없는 package import/루트·하위 CLI help, CLI override와 잘못된 인자, paired 증강 정렬·RGB/stride, 공식 metric fixture, 원본/이식 모델 동등성, 수정한 LayerNorm/TLC, QAT gradient·저장/복원, export 및 엔진 출력 비교. CLI는 `typer.testing.CliRunner`로 확인하고 JSON stdout과 로그 stderr를 구분해 검증한다. 문서 수정만을 위해 테스트·의존성을 추가하지 않는다. 데이터·Orin이 없으면 미검증으로 표시하고 수치를 만들지 않는다. [Typer 테스트](https://typer.tiangolo.com/tutorial/testing/)
- 해상도·FPS/P95·전력 모드가 미정이면 기능 구현과 검증 상태까지 보고하고 실시간 배포 목표 달성을 판정하지 않는다. 가변 입력·타일링·분산학습·추적 서비스·2:4/DLA는 초기 완료 조건에 추가하지 않는다.

참고: [공식 width32 평가 설정](https://github.com/megvii-research/NAFNet/blob/main/options/test/GoPro/NAFNet-width32.yml), [모델 코드](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py), [TLC 코드](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/local_arch.py).
