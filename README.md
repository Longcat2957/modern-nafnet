# Modern NAFNet

공식 NAFNet-width32를 참조해 현재 PyTorch로 재이식한 독립적인 Deblurring 프로젝트다.
학습·calibration·QAT·ONNX export는 x86 CUDA 호스트에서 실행하고, ONNX를 Orin으로
전달해 대상 TensorRT 엔진을 빌드한다. 프로젝트 자체 코드는 AGPL-3.0-only다.
원저작물 고지와 체크포인트 출처는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있다.

## 환경

Python 3.12를 기본으로 사용한다. 패키지·환경·실행은 uv로 관리한다.

```bash
uv sync --locked --all-extras
uv run --locked drone-deblur --help
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

`configs/default.json`을 실험용 설정으로 복사해 실제 경로를 지정한다. 설정 JSON과
CLI의 상대 경로는 **명령을 실행한 작업 디렉터리**를 기준으로 한다. CLI에서 명시한
옵션만 JSON을 덮어쓰며 `--no-amp`, `--no-quantized`, `qat --steps 0`도 보존한다.
미정인 배포 shape·FPS·전력 모드는 `null`로 둔다.

paired manifest는 비어 있지 않은 JSON 배열이다. **manifest 내부 이미지 경로는
manifest 파일의 디렉터리**를 기준으로 한다. `name`은 선택 사항이며 지정하면 고유해야 한다.

```json
[
  {"name": "sequence-001/frame-0001", "blur": "blur/0001.png", "sharp": "sharp/0001.png"}
]
```

학습에는 `data.train_manifest`와 `data.crop_size`를 지정한다. paired RandomCrop와
균등 D4 변환을 AlbumentationsX Compose 한 번으로 적용한다. 8-bit HWC RGB에서
contiguous CHW float `[0,1]`로 변환하며 ImageNet 정규화를 적용하지 않는다.
파일 누락·읽기 실패·쌍의 크기 불일치는 오류로 처리한다.

calibration manifest는 같은 배열 형식에서 `blur`만 있어도 된다. 이미지는
`export.shape`의 고정 H/W로 미리 준비해야 하며 자동 resize/crop이나 무작위 증강을 하지 않는다.
촬영 시퀀스·장소·날짜 단위로 학습·calibration·validation·test를 분리하고
최종 test를 calibration이나 튜닝에 사용하지 않는다.

## 모델 실행과 학습

공식 GoPro width32 체크포인트를 upstream 링크에서 받아 `artifacts/checkpoints/`에 둔다.
현재 로컬 검증 파일의 SHA-256은
`19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`다.
raw state dict, 공식 `params`, 자체 `state_dict`는 `strict=True`로 읽으며
키 재매핑으로 구조 차이를 숨기지 않는다.

```bash
uv run --locked --extra train drone-deblur infer \
  --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth \
  --input data/blur.png --output artifacts/restored.png

uv run --locked --extra train drone-deblur train \
  --config configs/experiment.json \
  --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth
```

학습은 원본 PSNRLoss, AdamW(weight decay 0), cosine schedule을 사용한다.
`training.steps/batch_size/lr/seed/num_workers/device/amp/output_dir`로 제어한다.
AMP는 `torch.amp`를 사용하고 손실·LayerNorm 통계·TLC 누적의 정밀도를 관리한다.

ModelOpt INT8는 calibration 후 scale을 고정해 학습한다. Conv weight는 per-channel,
activation은 per-tensor이고 attention의 `sca.*` quantizer는 제외한다.
QAT 설정에는 `data.calibration_manifest`, `export.shape`,
`quantization.calibration_batches`와 paired 학습 데이터가 필요하다.

```bash
uv run --locked --extra qat drone-deblur qat \
  --config configs/experiment.json \
  --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth --steps 0

uv run --locked --extra qat drone-deblur qat \
  --config configs/experiment.json --checkpoint artifacts/train/ptq.pt
```

`--steps 0`은 부동소수점 체크포인트에서 calibration만 수행해 `ptq.pt`를 저장한다.
학습 체크포인트에는 가중치·ModelOpt 구조·optimizer/scheduler/scaler·RNG·독립 증강 RNG·
DataLoader generator·TLC kernel·설정·manifest hash를 함께 저장한다.
완료한 epoch의 `last.pt`는 `--resume`로 재개한다. 중간 epoch의 `partial.pt`와
`interrupted.pt`는 inference/export에 사용할 수 있으며 임의 step의 정확한 재개는 보장하지 않는다.
정확한 증강 replay는 `num_workers=0`인 epoch 경계로 한정한다.

## ONNX 전달

고정 batch 1·RGB NCHW shape를 지정한다. FP ONNX는 dynamo exporter,
ModelOpt INT8는 symbolic Q/DQ를 지원하는 legacy exporter를 명시적으로 사용한다.

```bash
uv run --locked --extra export drone-deblur export \
  --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth \
  --height 256 --width 256 --output artifacts/nafnet-fp32.onnx

uv run --locked --extra export --extra qat drone-deblur export \
  --config configs/experiment.json --checkpoint artifacts/train/last.pt \
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
uv run --locked --extra train drone-deblur eval \
  --checkpoint artifacts/checkpoints/NAFNet-GoPro-width32.pth \
  --manifest data/test.json --output artifacts/evaluation.json
```

각 명령의 정상 stdout은 JSON 결과이고 progress/logging은 stderr로 보낸다.
inference는 PNG 등 이미지와 `.json` 기록을 남기며 입력·체크포인트를 덮어쓰지 않는다.
평가 JSON도 manifest·체크포인트·blur/sharp 이미지를 덮어쓰지 않는다.

pytest는 CLI/config, paired 정렬·seed·range, loss·metric, 모델 forward/gradient,
체크포인트 복원, 작은 학습 재개, ModelOpt calibration/backward, FP/QDQ ONNX→ONNX Runtime
출력 동등성을 검증한다. upstream oracle은 고정 commit의 별도 개발 snapshot을 사용한다.
`artifacts/reference/NAFNet/<commit>/`가 없으면 oracle 비교는 명시적으로 skip하며,
자급 가능한 fixture 검증은 실행한다. 공식 체크포인트도 없는 경우 관련 oracle 테스트는 skip한다.
snapshot과 가중치는 Git 및 설치 패키지에 포함하지 않는다.

**실제 데이터 학습·GoPro 품질 재현·Orin 엔진 빌드·FPS/지연/전력 측정은 아직 수행하지 않았다.**
성능 평가는 모델 학습/QAT 이후 별도 단계로 진행한다. 목표 해상도·FPS·전력 모드는
실측에 따라 확정하며 TensorRT engine inspector로 실제 INT8 Conv와 정밀도 변환을 확인해야 한다.
