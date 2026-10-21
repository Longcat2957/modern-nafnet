# Modern NAFNet

공식 NAFNet-width32를 참조해 현재 PyTorch로 재이식한 독립적인 Deblurring 프로젝트다.
학습·calibration·QAT·ONNX export는 x86 CUDA 호스트에서 실행하고, ONNX를 Orin으로
전달해 대상 TensorRT 엔진을 빌드한다. 프로젝트 자체 코드는 AGPL-3.0-only다.
원저작물 고지와 체크포인트 출처는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있다.

## 환경

Python 3.12를 기본으로 사용한다. 패키지·환경·실행은 uv로 관리한다.

```bash
uv sync --locked --all-extras
uv run --locked modern-nafnet --help
uv run --locked --all-extras pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

선택 의존성은 `train`(PyTorch·AlbumentationsX), `qat`(PyTorch·ModelOpt·AlbumentationsX),
`export`(PyTorch·ONNX·ONNXScript), `metrics`(TorchMetrics)다. pytest·Ruff·ONNX Runtime은
개발 group에 있다. 전체 설치가 필요 없으면 `uv sync --locked --extra train`처럼
프로필을 지정한다. 각 명령은 사용하는 프로필을 `uv run --locked --extra ...`에 명시한다.

Linux x86_64의 torch는 명시적인 공식 `cu132` 인덱스를 사용한다. 현재 검증 환경은
PyTorch 2.14.1+cu132 / CUDA 13.2, ModelOpt 0.47.0, Python 3.12.14,
RTX 5000 Ada Generation Laptop GPU다. ModelOpt 0.47의 PyTorch 모듈은
base metadata에 없는 `requests`·`huggingface-hub`를 import하므로 `qat`에 추가했다.
Transformers·ModelOpt `[all]`·BasicSR은 프로젝트 의존성이 아니다.

## 설정과 데이터

실험 설정 파일 **하나**에 데이터 위치와 바꿀 학습 옵션만 적는다.
[`configs/train.json`](configs/train.json)을 수정해 시작하며, 나머지는 기본값을 사용한다.
데이터 입력용 manifest와 `--manifest` 옵션은 제거했다.

```json
{
  "data": {"path": "data/deblur"},
  "training": {"checkpoint": "weights/nafnet-gopro-width32.pt"}
}
```

기본 폴더 구조는 다음과 같다. 하위 시퀀스 폴더가 있어도 같은 상대 경로·파일명을
기준으로 blur/sharp를 자동 매칭한다. 누락된 쌍이나 모호한 배치는 오류로 처리한다.

```text
data/deblur/
├── train/
│   ├── blur/0001.png
│   └── sharp/0001.png
├── val/
│   ├── blur/0002.png
│   └── sharp/0002.png
├── test/                  # 평가할 때 준비
│   ├── blur/0003.png
│   └── sharp/0003.png
└── calibration/0004.png   # QAT 초기화할 때 실제 blur 이미지 준비
```

```bash
uv run --locked --extra train modern-nafnet train --config configs/train.json
```

config 없이 경로를 바로 지정할 수도 있다. GoPro의 `train/<sequence>/{blur,sharp}/`
구조는 그대로 읽으며 JSON 목록을 만들지 않는다.

```bash
uv run --locked --extra train modern-nafnet train \
  --data data/GoPro --checkpoint weights/nafnet-gopro-width32.pt
```

폴더명이 다르면 같은 config 안에서 `data.train/val/test/calibration`만 바꾼다.
[Ultralytics의 데이터 설정](https://docs.ultralytics.com/datasets/detect/)처럼 `data.path`는
루트, split 값은 그 아래의 경로다. `data.path`와 CLI 상대 경로는 **명령을 실행한
작업 디렉터리** 기준이며, split에 절대 경로를 지정할 수도 있다. `eval`은 기본 `val`을
읽고 `--split test`로 test를 선택한다. GoPro의 공개 평가 폴더는 `test`를 사용한다.

sharp 원본만 있다면 `data/Clean/train/`에 이미지를 넣고 합성 blur를 켠다.
하위 폴더 또는 `train/sharp/`만 있는 구조도 사용할 수 있다.

```bash
uv run --locked --extra train modern-nafnet train \
  --data data/Clean --checkpoint weights/nafnet-gopro-width32.pt \
  --synthetic-blur --num-workers 2
```

원본은 깨끗한 sharp GT여야 한다. crop/D4를 먼저 적용한 뒤 입력에만 AlbumentationsX
MotionBlur를 적용하며 GT를 보존한다. 실제 paired 폴더의 누락을 합성 blur로 대체하지 않는다.
`data.synthetic_blur.blur_range`의 기본 kernel 범위는 `[3,15]`다. 이미지를 한 번 읽고
uint8 crop에서 증강한 뒤 contiguous CHW float `[0,1]`로 변환한다. ImageNet 정규화나
합성 쌍의 디스크 복제는 하지 않는다. `qat`에서도 같은 옵션을 사용할 수 있다.

폴더는 시작 시 한 번 탐색하며 sample마다 재탐색하지 않는다. `--num-workers`는 장비에
맞춰 조절한다. worker에서는 OpenCV 내부 스레드·OpenCL을 꺼 중첩 병렬화를 피한다.
crop 우선·uint8 유지는 [공식 성능 지침](https://albumentations.ai/docs/3-basic-usage/performance-tuning/)을 따른다.
원본 이미지 decode 비용은 남으며 CPU 처리량과 합성 blur의 학습 효과는 아직 측정하지 않았다.

calibration은 `data.calibration`의 실제 blur 이미지를 사용하며 `export.shape`에 맞는
고정 H/W로 준비한다. 자동 resize/crop·합성 증강·평가 폴더 fallback을 하지 않는다.
촬영 시퀀스·장소·날짜 단위로 train/calibration/val/test를 분리하고 임의 자동 split은 만들지 않는다.
CLI에 명시한 값만 config를 덮어쓰며 `--no-amp`, `--no-synthetic-blur`, `qat --steps 0`도 보존한다.

## 모델 실행과 학습

[`weights/nafnet-gopro-width32.pt`](weights/nafnet-gopro-width32.pt)는 저장소에 포함되어
clone 후 별도 upstream 다운로드 없이 사용한다. 공식 GoPro width32 `.pth`의 가중치
값·dtype을 FP32 그대로 보존하고 `state_dict`, `config.model`, `tlc_kernels`,
`provenance`를 담은 자체 `.pt`로 포장했다. 재학습·양자화하지 않았다.
원본 `.pth`의 SHA-256은
`19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`이며,
변환본의 checksum과 출처·라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 기록한다.
raw state dict, 공식 `params`, 자체 `state_dict`는 `strict=True`로 읽으며
키 재매핑으로 구조 차이를 숨기지 않는다.
`artifacts/`에는 학습 체크포인트·복원 이미지·ONNX·평가 기록 등 실행으로 생성한 결과만 둔다.

```bash
uv run --locked --extra train modern-nafnet infer \
  --checkpoint weights/nafnet-gopro-width32.pt \
  --input data/blur.png --output artifacts/restored.png

uv run --locked --extra train modern-nafnet train \
  --config configs/train.json \
  --checkpoint weights/nafnet-gopro-width32.pt
```

학습은 원본 PSNRLoss, AdamW(weight decay 0), cosine schedule을 사용한다.
`training.steps/batch_size/lr/seed/num_workers/device/amp/output_dir`로 제어한다.
AMP는 `torch.amp`를 사용하고 손실·LayerNorm 통계·TLC 누적의 정밀도를 관리한다.

ModelOpt INT8는 calibration 후 scale을 고정해 학습한다. Conv weight는 per-channel,
activation은 per-tensor이고 attention의 `sca.*` quantizer는 제외한다.
QAT 설정에는 `data.path/calibration`, `export.shape`,
`quantization.calibration_batches`가 필요하다. 학습에는 paired 데이터 또는
합성 blur를 켠 sharp-only 데이터를 사용한다.

```bash
uv run --locked --extra qat modern-nafnet qat \
  --config configs/train.json \
  --checkpoint weights/nafnet-gopro-width32.pt --steps 0

uv run --locked --extra qat modern-nafnet qat \
  --config configs/train.json --checkpoint artifacts/train/ptq.pt
```

`--steps 0`은 부동소수점 체크포인트에서 calibration만 수행해 `ptq.pt`를 저장한다.
학습 체크포인트에는 가중치·ModelOpt 구조·optimizer/scheduler/scaler·RNG·독립 증강 RNG·
DataLoader generator·TLC kernel·설정·데이터 파일 목록과 fingerprint를 함께 저장한다.
완료한 epoch의 `last.pt`는 `--resume`로 재개한다. 중간 epoch의 `partial.pt`와
`interrupted.pt`는 inference/export에 사용할 수 있으며 임의 step의 정확한 재개는 보장하지 않는다.
정확한 증강 replay는 `num_workers=0`인 epoch 경계로 한정한다.
합성 blur의 독립 RNG도 저장·복원하며 resume에는 기존 synthetic 설정을 유지해야 한다.
데이터 fingerprint는 경로·파일 크기·수정시간을 검사하며, 이미지 내용을 전부 읽어 hash하지 않는다.
데이터 snapshot이 없는 과거 학습 checkpoint는 `--resume` 대신 `--checkpoint`로 가중치를 사용한다.

## ONNX 전달

고정 batch 1·RGB NCHW shape를 지정한다. FP ONNX는 dynamo exporter,
ModelOpt INT8는 symbolic Q/DQ를 지원하는 legacy exporter를 명시적으로 사용한다.

```bash
uv run --locked --extra export modern-nafnet export \
  --checkpoint weights/nafnet-gopro-width32.pt \
  --height 256 --width 256 --output artifacts/nafnet-fp32.onnx

uv run --locked --extra export --extra qat modern-nafnet export \
  --config configs/train.json --checkpoint artifacts/train/last.pt \
  --quantized --height 256 --width 256 --output artifacts/nafnet-int8.onnx
```

ONNX checker로 그래프를 검사하며 INT8 그래프에 QuantizeLinear/DequantizeLinear가
없으면 실패한다. ONNX 옆의 `.onnx.json`에 hash·opset·shape·dtype·TLC kernel·RGB/range·
padding/crop·버전을 기록한다. 기존 ONNX와 manifest를 덮어쓰지 않으므로 새 경로를 지정한다.
export는 TensorRT 실행 성공이나 INT8 실행 비율을 인증하지 않는다.

이 리포는 **TensorRT calibration cache 파일을 생성하지 않는다**. 실제 이미지로
ModelOpt calibration을 수행한 scale은 `ptq.pt`/QAT 체크포인트에 저장하고,
export 시 ONNX의 Q/DQ에 포함한다. 이 explicit quantization 경로에는 별도
`--calib` 파일이 필요 없다. NVIDIA도 Q/DQ 모델에 외부 calibration table을
제공하지 말라고 안내한다. [공식 quantization 설명](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)
`trtexec --int8 --calib=...`는 Q/DQ 없는 FP 모델의 implicit INT8 경로이며,
cache 없이 `--int8`만 사용하면 임의 dynamic range가 설정될 수 있다.
이를 이 리포의 실제 calibration 결과로 취급하지 않는다.

Orin에는 ONNX와 manifest를 전달한다. 전체 Python 프로젝트를 설치할 필요는 없다.
대상 SDK의 `trtexec`로 엔진을 빌드하고 레이어를 분석한다. Q/DQ strong typing을
선택하면 `--int8`/`--fp16` flag를 혼용하지 않으며 floating 정밀도는 ONNX 타입/Cast로 정한다.
실제 이미지 출력 비교·제품 연동에는 필요한 최소 runner를 사용한다.
Orin용 CLI·Python 바인딩·자체 TensorRT runtime은 현재 구현 범위에 추가하지 않았다.

## 평가와 검증 범위

품질 평가 명령은 paired GT가 있는 경우에 사용한다. RGB, border crop 0,
clip→×255→round→uint8, 이미지별 지표 평균과 원본 channel-inclusive SSIM3d를 보존한다.
TorchMetrics의 RGB 2D SSIM과 혼용하지 않는다. PSNR 무한대는 표준 JSON을 위해 `"inf"`로 기록한다.

```bash
uv run --locked --extra train modern-nafnet eval \
  --checkpoint weights/nafnet-gopro-width32.pt \
  --data data/GoPro --split test --output artifacts/evaluation.json
```

각 명령의 정상 stdout은 JSON 결과이고 progress/logging은 stderr로 보낸다.
inference는 PNG 등 이미지와 `.json` 기록을 남기며 입력·체크포인트를 덮어쓰지 않는다.
평가 JSON도 체크포인트·blur/sharp 이미지를 덮어쓰지 않으며 실제 평가 파일 목록을 기록한다.

pytest는 CLI/config, paired 정렬·seed·range, loss·metric, 모델 forward/gradient,
체크포인트 복원, 작은 학습 재개, ModelOpt calibration/backward, FP/QDQ ONNX→ONNX Runtime
출력 동등성을 검증한다. upstream oracle은 고정 commit의
`tests/reference/NAFNet/<UPSTREAM_COMMIT>/`에 Git으로 보존한 개발 snapshot을 사용한다.
이 원본 코드는 테스트 전용이며 production 모델·runtime에서 import하지 않는다.
snapshot 또는 제공 체크포인트가 없는 경우 관련 oracle 비교는 명시적으로 skip하며,
자급 가능한 fixture 검증은 실행한다. snapshot과 가중치는 Git 저장소에 포함하지만
설치 Python 패키지에는 포함하지 않는다.

### GoPro batch 1 시험 결과 (2026-10-13)

`/home/wmin/Downloads/GoProDataset_Kaggle`의 실제 데이터로 pretrained FP32·PTQ와
100-step FP32·QAT를 비교했다. FP32/QAT는 같은 데이터 순서와 crop/D4 증강으로
batch size 1, 256×256 crop, AdamW lr `1e-5`, cosine schedule, seed 42,
worker 0, AMP/TF32 비활성 조건에서 각각 100 step 학습했다.
720×1280 원해상도 test **1,111쌍 전체**를 위의 공식 uint8·RGB·SSIM3d protocol로 평가했다.

| 모델 | PSNR (dB) | SSIM |
|---|---:|---:|
| 공식 pretrained FP32 | 32.8705 | 0.960581 |
| 동일 가중치 INT8 PTQ | 32.1898 | 0.950566 |
| FP32 fine-tune 100 step | 32.8687 | 0.960580 |
| INT8 QAT 100 step | 32.2520 | 0.951568 |

pretrained FP32는 공개 GoPro 기준을 재현했다. PTQ는 pretrained 대비 **−0.6807 dB**,
QAT는 같은 학습량의 FP32 대조군 대비 **−0.6167 dB**다. QAT의 PTQ 대비 회복은
**+0.0622 dB**이며, 두 INT8 비교 모두 초기 0.3 dB 손실 목표를 충족하지 못했다.
PTQ는 일부 이미지에서 큰 열화를 보였으며 최대 PSNR 하락은 **10.7301 dB**였다.
100 step은 짧은 시험이며 학습 수렴을 판정하지 않는다.

원본 데이터는 유지하고 `blur_gamma`를 제외한 파일 symlink 폴더로 실행했다.
calibration에는 `GOPR0372_07_00/01` 전체 촬영본을 학습에서 예약하고 각각 16장의
분산 프레임, 총 32장 실제 blur를 사용했다. 추가 학습에는 test와 영상 ID가 겹치는
train 촬영본도 제외하여 1,028쌍을 사용했다. test는 calibration·튜닝에 사용하지 않았다.
공식 pretrained가 이미 사용한 GoPro 학습 데이터의 노출은 제거할 수 없다.

Conv 226개 중 주요 Conv 190개에 W8A8 fake quantization을 적용하고 attention Conv
36개를 제외했다. PTQ는 원본 664개 가중치 텐서를 그대로 유지했으며,
380개 calibration scale은 QAT 중 bitwise 동일하게 유지됐다.
INT8 수치는 **PyTorch fake quantization 품질**이며 TensorRT INT8 실행·저장 크기·속도 결과가 아니다.

실행 설정·환경·데이터 snapshot·checkpoint hash·이미지별 지표와 보고서는
`artifacts/gopro-b1-100step-20261013/`에 저장했다. `report.txt`는 요약,
`report.json`은 전체 비교·열화 항목·검증 기록이며 실행 산출물은 Git에서 제외한다.
FP32/QAT의 `partial.pt`는 평가·export에 사용할 수 있지만 epoch 중간이므로 정확한
`--resume`은 지원하지 않는다.

**Orin 엔진 빌드와 TensorRT 전체 test FP16/INT8 품질·FPS/지연/전력은 아직 미검증이다.**
이 시험의 720×1280 크기는 최종 Orin 목표 해상도가 아니다. 목표 해상도·FPS·전력 모드는
실측으로 확정하며 engine inspector로 실제 INT8 Conv와 정밀도 변환을 확인해야 한다.

### TensorRT 24.08 Docker 검증 (2026-10-13)

`nvcr.io/nvidia/tensorrt:24.08-py3`의 TensorRT **10.3.0.26 / CUDA 12.6**으로 위 QAT
체크포인트의 고정 `[1,3,720,1280]` Q/DQ ONNX를 빌드했다. 현재 x86 RTX 5000 Ada에서
parser·엔진 build·실제 이미지 3장 추론이 성공했다. `--int8`, `--fp16`, `--calib` 없이
`--stronglyTyped --noTF32`를 사용했다. 확인한 build 명령은 다음과 같다.

```bash
docker run --rm --gpus device=0 --network none \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/artifacts/trt-24.08-check-20261013:/work" \
  nvcr.io/nvidia/tensorrt:24.08-py3 /opt/tensorrt/bin/trtexec \
  --onnx=/work/nafnet-qat-int8-720x1280.onnx \
  --saveEngine=/work/nafnet-qat-int8-720x1280.engine \
  --stronglyTyped --noTF32 --memPoolSize=workspace:4096 \
  --profilingVerbosity=detailed --exportLayerInfo=/work/engine-layers.json \
  --skipInference
```

ONNX에 Q/DQ 각각 **380개**, FP32 TLC `CumSum` **72개**가 있으며 모든 scale은
finite·positive, zero point는 symmetric INT8 0이었다. 체크포인트 `amax/127` 대비
scale 최대 상대 오차는 `1.19e-7`이다. 엔진 inspector에서 **190개 Conv의 INT8
입력·가중치·CUDA tactic**과 attention **36개 Conv의 FP32**를 확인했다.
엔진은 34.91 MiB이며 별도 calibration cache를 제공하지 않았다.

실제 3장에서는 PyTorch QAT 대비 PSNR 차이의 절댓값이 최대 **0.0366 dB**였다.
그러나 픽셀 mean absolute error는 **0.003665–0.004866**, max absolute error는
최대 **0.06805**로, 선언한 `mean < 1/255` 기준을 2/3장에서 넘었다.
**엔진 build·실행은 성공했지만 엄격한 픽셀 동등성 검사는 미달**이다.
PyTorch CPU/GPU 및 ONNX Runtime에서도 유사한 차이가 확인되어 TensorRT나
export만의 오류로 단정하지 않으며, 원인 확정에는 중간 텐서 비교가 필요하다.

ONNX·engine·image digest·hash·layer inspector·추론 이미지·비교 결과·명령 로그는
`artifacts/trt-24.08-check-20261013/`의 `report.txt`/`report.json`과 함께 보관했다.
3장 검증은 전체 test 평가가 아니며 warm-up 없는 1회 추론 로그를 FPS/지연 benchmark로
사용하지 않는다. 이 x86 엔진의 Orin 이식성을 가정하지 않으며 Orin에서 다시 빌드해야 한다.

## TRT-R2R-Light

[`TRT_R2R_LIGHT_IMPLEMENTATION.md`](TRT_R2R_LIGHT_IMPLEMENTATION.md)의 V0 모델을
`architecture: "trt-r2r-light"`로 선택한다. 기존 NAFNet 설정·체크포인트와 별개의
아키텍처이며, 제공된 GoPro 가중치를 새 모델에 로드할 수 없다.
[`configs/trt-r2r-light.json`](configs/trt-r2r-light.json)은 처음부터 학습하는 예제다.
기본 모델은 1,711,761개 파라미터의 Residual CNN과 학습 가능한 고정 크기 Prototype Bank다.
`trt_r2r_light.bank_enabled=false`로 Bank 없는 기준 모델도 학습할 수 있다.

기본 task 순서는 `denoise, derain, dehaze, deblur, lowlight`이며 체크포인트에 고정한다.
Derain/Deblur부터 시작하려면 `trt_r2r_light.tasks`를 `["derain", "deblur"]`로 설정한다.
Clean은 별도 task가 아닌 모두 0인 라벨이다. `derain+deblur` 폴더는 두 라벨이 모두 1이다.
학습·평가에는 아래 구조를 사용한다. calibration은 각 폴더의 `input/`만 필요하다.

```text
data/restoration/
├── train/                 # val/, test/도 같은 구조
│   ├── derain/{input,target}/sequence/0001.png
│   ├── deblur/{input,target}/sequence/0002.png
│   ├── derain+deblur/{input,target}/sequence/0003.png
│   └── clean/target/sequence/0004.png
└── calibration/
    ├── derain/input/0005.png
    ├── deblur/input/0006.png
    ├── derain+deblur/input/0007.png
    └── clean/input/0008.png
```

Input/target의 상대 경로·확장자는 일치해야 하며 자동 resize나 누락 쌍 대체는 하지 않는다.
Clean target은 한 번 읽어 입력으로도 사용한다. 학습에는 paired crop/D4를 적용하고,
기본 `task_sampling.mode="balanced"`는 작은 task 폴더를 반복해 작업별 샘플 수를 맞춘다.
원래 데이터 비율은 `"proportional"`로 선택한다. Calibration은 모든 설정 task와 Clean을
포함해야 하며 존재하는 복합 열화도 제외하지 않는다. 데이터가 부족하거나 최대 장수로
모든 그룹을 포함할 수 없으면 실패한다. 기본 5-task 예제에는 위 두 task 외 세 task도 준비한다.

```bash
uv run --locked --extra train modern-nafnet train --config configs/trt-r2r-light.json

uv run --locked --extra qat modern-nafnet qat --config configs/trt-r2r-light.json \
  --checkpoint artifacts/trt-r2r-light/train/last.pt --steps 0

uv run --locked --extra qat modern-nafnet qat --config configs/trt-r2r-light.json \
  --checkpoint artifacts/trt-r2r-light/train/ptq/ptq.pt

uv run --locked --extra export --extra qat modern-nafnet export \
  --config configs/trt-r2r-light.json \
  --checkpoint artifacts/trt-r2r-light/train/qat/last.pt --quantized \
  --height 256 --width 256 --output artifacts/trt-r2r-light/model-int8.onnx

uv run --locked --extra train modern-nafnet eval --config configs/trt-r2r-light.json \
  --checkpoint artifacts/trt-r2r-light/train/last.pt --split val \
  --output artifacts/trt-r2r-light/quality.json
```

총 step이 epoch 중간이면 `last.pt` 대신 결과 JSON의 `partial.pt`를 사용한다.
PTQ는 `training.output_dir/ptq/ptq.pt`, QAT는 `training.output_dir/qat/`에 저장해
FP32 결과를 보존한다. 손실은 Charbonnier·edge·multiscale, raw-logit BCE와 Clean identity의
합이며 `restoration_loss`로 조정한다. 평가 JSON에는 작업별·Clean·단일/복합 품질,
학습에 존재한/없던 복합 조합, gate confusion·Clean false positive·Bank 진단을 기록한다.
학습 snapshot 없는 체크포인트를 쓰면 복합 조합의 학습 노출은 unknown으로 기록한다.

현재 정밀도는 Stage A다. Backbone/decoder Conv는 W8A8 Q/DQ, Stem/Tail·Query·Bank·Add는
FP32로 유지한다. Prototype Key/Value는 PTQ 전에 고정하며 QAT에서도 유지한다.
FP32 학습과 QAT만 지원하며 이 모델의 `amp=true`는 거부한다. FP export에는
`--extra export`와 `--no-quantized`를 사용한다. Bank fingerprint·task 순서·모델 설정을
엄격히 검사하고, ONNX에서는 단일 RGB 출력·Bank initializer·Q/DQ 경계도 검사한다.

모델 입력은 외부에서 오른쪽/아래를 16배수로 zero-pad한 RGB NCHW `[0,1]`이다.
`infer`/`eval`은 패딩 후 원래 크기로 crop한다. 학습 crop과 export shape는 16배수여야 한다.
Calibration 이미지에도 같은 패딩만 허용하며 export shape와 정확히 일치해야 한다.
1080p는 1088×1920으로 별도 calibration/export하고, ONNX와 `.onnx.json` 및 외부 가중치
파일이 있으면 함께 Orin에 전달한다. Q/DQ manifest에는 calibration snapshot, Bank hash,
정밀도 정책과 `tensorrt_verified=false`를 기록한다.

다중 GPU 학습은 `torchrun`의 환경 변수를 사용한다. `batch_size`는 rank별 크기이며,
rank 0만 체크포인트와 JSON 결과를 쓴다. Balanced sampler의 작업별 샘플 수와
rank별 RNG를 기록하고, resume은 동일 world size·epoch 경계로 제한한다.

```bash
uv run --locked --extra train torchrun --standalone --nproc-per-node=2 \
  --no-python .venv/bin/modern-nafnet train --config configs/trt-r2r-light.json
```

Orin에서 설치된 TensorRT 버전과 `trtexec --help`를 확인한 후 아래처럼 빌드한다.
이 명령은 엔진 build/레이어 분석용이며 성능 측정은 학습 이후 별도 단계다.

```bash
trtexec --onnx=model-int8.onnx --saveEngine=model-int8.plan \
  --stronglyTyped --profilingVerbosity=detailed \
  --exportLayerInfo=layers.json --skipInference
```

강타이핑 Q/DQ에는 `--fp16`/`--int8`을 추가하지 않는다.
실제 INT8 tactic·Reformat·출력 화질은 대상 Orin에서 별도로 확인해야 한다.
GT-guided Teacher Bank, typed FP16 Stage B, stride-stem 비교와 실데이터 학습/품질·Orin
지연/전력 검증은 후속 단계이며, 구현된 기능만으로 복원 품질이나 실시간 성능을 주장하지 않는다.

### TRT-R2R-Light 기능 검증 (2026-10-21)

RTX 5000 Ada에서 합성 fixture로 Bank on/off 각각 FP32 64 step과 QAT 32 step을
실행했다. 공통 초기 backbone·동일 학습량으로 손실 감소와 PTQ→QAT→저장/복원·backward를
확인했으며 48개 calibration `amax`와 고정 Bank, 원본 FP32 체크포인트 hash를 보존했다.
이는 학습 경로 검증이며 실제 Derain/Deblur 품질이나 유용한 routing을 입증하지 않는다.
2-rank DDP는 CPU Gloo로 검증했으며 다중 GPU NCCL 실행은 아직 확인하지 않았다.

기본 5-task 모델의 FP32/QDQ ONNX는 256×256과 1088×1920에서 checker를 통과했다.
캐시된 x86 TensorRT 10.3.0.26 컨테이너에서 두 Q/DQ 엔진을
`--stronglyTyped --noTF32 --skipInference`로 빌드했고, Inspector에서 내부 Conv 24개의
INT8 입력/가중치와 Stem/Tail 2개의 FP32를 확인했다. Orin 엔진·출력·성능 검증은 아니다.

256² FP32의 PyTorch–ONNX Runtime 최대 절대 오차는 `4.77e-7`이었다.
전체 Q/DQ 모델은 ONNX Runtime 최적화를 꺼도 PyTorch fake INT8 대비 최대 약
`0.008–0.010` 차이가 있어 엄격한 수치 동등성 통과로 처리하지 않았다. FP32 Conv의
작은 차이가 activation 양자화 경계를 넘는 현상을 관측했으며 상세 분석은 보고서에 남겼다.
합성 입력의 이 결과를 실제 이미지 PSNR/SSIM이나 TensorRT 출력 동등성으로 해석하지 않는다.

설정·파일 목록·hash·환경·학습/진단 결과는
`artifacts/trt-r2r-light-smoke-20261021/training/report.json`, ONNX·x86 엔진·레이어 분석과
수치 비교는 `artifacts/trt-r2r-light-smoke-20261021/export/report.json` 및
`qdq-diagnosis.json`에 저장했다. 이 실행 산출물은 Git에서 제외한다.
