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
  "training": {"checkpoint": "artifacts/checkpoints/NAFNet-GoPro-width32.pth"}
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
  --data data/GoPro --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth
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
  --data data/Clean --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth \
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

공식 GoPro width32 체크포인트를 upstream 링크에서 받아 `artifacts/checkpoints/`에 둔다.
현재 로컬 검증 파일의 SHA-256은
`19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`다.
raw state dict, 공식 `params`, 자체 `state_dict`는 `strict=True`로 읽으며
키 재매핑으로 구조 차이를 숨기지 않는다.

```bash
uv run --locked --extra train modern-nafnet infer \
  --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth \
  --input data/blur.png --output artifacts/restored.png

uv run --locked --extra train modern-nafnet train \
  --config configs/train.json \
  --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth
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
  --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth --steps 0

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
  --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth \
  --height 256 --width 256 --output artifacts/nafnet-fp32.onnx

uv run --locked --extra export --extra qat modern-nafnet export \
  --config configs/train.json --checkpoint artifacts/train/last.pt \
  --quantized --height 256 --width 256 --output artifacts/nafnet-int8.onnx
```

ONNX checker로 그래프를 검사하며 INT8 그래프에 QuantizeLinear/DequantizeLinear가
없으면 실패한다. ONNX 옆의 `.onnx.json`에 hash·opset·shape·dtype·TLC kernel·RGB/range·
padding/crop·버전을 기록한다. 기존 ONNX와 manifest를 덮어쓰지 않으므로 새 경로를 지정한다.
export는 TensorRT 실행 성공이나 INT8 실행 비율을 인증하지 않는다.

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
  --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth \
  --data data/GoPro --split test --output artifacts/evaluation.json
```

각 명령의 정상 stdout은 JSON 결과이고 progress/logging은 stderr로 보낸다.
inference는 PNG 등 이미지와 `.json` 기록을 남기며 입력·체크포인트를 덮어쓰지 않는다.
평가 JSON도 체크포인트·blur/sharp 이미지를 덮어쓰지 않으며 실제 평가 파일 목록을 기록한다.

pytest는 CLI/config, paired 정렬·seed·range, loss·metric, 모델 forward/gradient,
체크포인트 복원, 작은 학습 재개, ModelOpt calibration/backward, FP/QDQ ONNX→ONNX Runtime
출력 동등성을 검증한다. upstream oracle은 고정 commit의 별도 개발 snapshot을 사용한다.
`artifacts/reference/NAFNet/<commit>/`가 없으면 oracle 비교는 명시적으로 skip하며,
자급 가능한 fixture 검증은 실행한다. 공식 체크포인트도 없는 경우 관련 oracle 테스트는 skip한다.
snapshot과 가중치는 Git 및 설치 패키지에 포함하지 않는다.

**실제 데이터 학습·GoPro 품질 재현·Orin 엔진 빌드·FPS/지연/전력 측정은 아직 수행하지 않았다.**
성능 평가는 모델 학습/QAT 이후 별도 단계로 진행한다. 목표 해상도·FPS·전력 모드는
실측에 따라 확정하며 TensorRT engine inspector로 실제 INT8 Conv와 정밀도 변환을 확인해야 한다.
