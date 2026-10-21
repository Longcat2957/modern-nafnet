# TRT-R2R-Light — 구현 설계서

> **목표:** `modern-nafnet`에 독립적인 다중 열화 복원 모델 `trt-r2r-light`를 추가하고, Jetson AGX Orin의 TensorRT 명시적 INT8(Q/DQ) 추론까지 검증한다.
>
> **상태:** 구현 명세 / 설계 제안. 신경망 학습·엔진 빌드·실기기 지연시간을 검증한 완료 보고서가 아니다.
>
> **작성 기준일:** 2026-10-21
>
> **기반 저장소:** <https://github.com/Longcat2957/modern-nafnet> (`main`, 검토 시점 `e143393ea656036956eedd8f0ee32f5683a84407`)
>
> **예정 브랜치:** `trt-r2r-light` (본 문서는 브랜치 생성 또는 코드 변경을 수행하지 않음)
>
> **아이디어 출처:** <https://github.com/cscxwang/R2R> (`main`, 검토 시점 `bf387d56095aaf4edc0b685f8ea58cce5c64c2fc`), *Retrieve-to-Restore*, CVPR 2026
>
> **대상 하드웨어:** NVIDIA Jetson AGX Orin, 우선 GPU 추론, Batch 1, RGB NCHW, PyTorch → ONNX → TensorRT

---

## 1. 핵심 설계 결정

**한 문장:** R2R의 **열화별 지식 검색(Degradation Bank)** 개념과 기존 저장소의 **검증된 ModelOpt PTQ/QAT·ONNX 인프라**를 결합하되, 픽셀 복원 Backbone은 INT8-friendly Residual CNN으로 새로 작성한다.

| 구분 | 선택 | 이유 |
|---|---|---|
| 패키지 | 기존 `modern_nafnet` 유지 | 현재 테스트·CLI·uv·체크포인트/배포 인프라 재사용 |
| 별도 모델 식별자 | `trt-r2r-light` | 기존 NAFNet 가중치/모델 계약과 명확히 구분 |
| 작업 방식 | 단일 공유 Encoder–Decoder + 다중 라벨 열화 Routing | 입력별 전체 복원 네트워크를 동적으로 전환하지 않음 |
| 기본 Backbone | Conv3×3/1×1 + ReLU + Residual Add | 복잡한 Norm/Gate/Attention 연산 최소화 |
| 해상도 처리 | `PixelUnshuffle(2)` → 3단 Downsample → `PixelShuffle(2)` | 고해상도 MAC 감소. Shuffle 비용은 실측 필요 |
| Bank 표현 | 작업별 8개 고정 크기 Key/Value 벡터 | R2R의 대형 공간형 메모리보다 단순한 정적 그래프 |
| Bank 검색 | 작은 FP 부동소수점 GEMM + Softmax + Sigmoid | ONNX 친화적, TopK/Argmax/Gather/동적 마스크 제거 |
| Prior 적용 | Bottleneck에 1회 Add | 고해상도 반복 변조/정밀도 변환 회피 |
| INT8 정책 | W8A8 Conv 우선, 명시적 Q/DQ | 지원 여부와 실제 엔진 INT8 kernel 사용을 별도 검증 |
| 초기 High Precision | 나머지 FP32, 이후 부분 FP16 실험 | 기존 Export의 `.float()` 동작과 강타이핑 문제를 단계적으로 해결 |
| 배포 | x86에서 Train/Calibration/QAT/Export → Orin에서 Engine Build | 기존 저장소의 실행 경계 유지 |
| 초기 범위 | Derain, Deblur, Clean | 핵심 작업의 간섭/품질/INT8 성능을 먼저 입증 |
| 확장 범위 | Denoise, Dehaze, Low-light 및 복합 열화 | All-in-One 완성. 별도 학습 데이터와 평가 필수 |

**중요:** 이것은 R2R을 복제한 모델이 아니라 **R2R-inspired 독립 설계**다. 원본 `R2R/README.md`에는 5D(denoise/derain/dehaze/deblur/low-light)와 별도로 CDD11의 compositional 실험이 있으며, `model_compos.py`는 dehaze/derain/desnow/low-light의 4종 열화를 사용한다. 두 실험을 동일 구성으로 간주하지 않는다.

### 1.1 확정하지 않은 사항

- 목표 FPS, 허용 품질 저하(PSNR/SSIM), 해상도별 전력 상한은 실기기 측정 전까지 미정.
- `PixelUnshuffle`, Add, Upsample, Bank FP16 경로의 엔진 내부 실행 형식은 **추정 금지**. 엔진 Inspector로 확인.
- R2R 논문의 성능 결과를 본 모델 또는 Orin 실측 결과로 인용하지 않는다.
- 기존 `weights/nafnet-gopro-width32.pt`는 새 아키텍처의 `strict=True` 가중치로 **직접 로드 불가**. 필요하면 Deblur용 Teacher로 별도 사용한다.

---

## 2. 현재 저장소 분석과 재사용 경계

실제 확인한 주요 파일:

| 기존 파일 | 현재 역할 | 변경 방침 |
|---|---|---|
| `src/modern_nafnet/models/nafnet.py` | NAFNet/NAFNetLocal, 원본 checkpoint strict-load | **기존 NAFNet 의미·가중치 호환성 보존** |
| `src/modern_nafnet/config.py` + `configs/default.json` | 키 검증이 있는 단일 JSON 설정 | 새 아키텍처/작업 설정을 추가하되 기존 `model` 설정은 그대로 유지 |
| `src/modern_nafnet/cli.py` | `train`, `qat`, `export`, `infer`, `eval` Typer 진입점 | CLI 이름 보존, 새 작업은 `--config`로 선택 |
| `src/modern_nafnet/training/loop.py` | 단일 학습/QAT 루프·DDP·체크포인트·epoch 재개 | 모델 factory, 다중 작업 Dataset, auxiliary loss 분기 추가 |
| `src/modern_nafnet/training/quantize.py` | ModelOpt W8A8·calibration·state 복원 | **모델별 INT8 정책**과 제외 레이어 검증 추가 |
| `src/modern_nafnet/deployment/export.py` | FP ONNX(dynamo), Q/DQ ONNX(legacy), manifest | 모델 factory 및 내보내기 정밀도 정책 분기 |
| `src/modern_nafnet/data/sources.py` | `blur/sharp` 디렉터리의 엄격한 상대경로 매칭 | 기존 함수 유지, 범용 `input/target` 매칭 추가 |
| `src/modern_nafnet/data/paired.py` | Deblur Dataset·RNG·RGB `[0,1]` | 그대로 유지, 새 다중 작업 Dataset 추가 |
| `src/modern_nafnet/training/losses.py` | 원본 NAFNet PSNRLoss | 기존 클래스 유지, 새 복원/분류 loss 함수 추가 |
| `src/modern_nafnet/evaluation/metrics.py` | GoPro 참조 PSNR/SSIM | 공통 지표 재사용, 작업별 집계 별도 구현 |
| `tests/` | Export/Quantize/DDP/모델/복원 호환성 검사 | 기존 테스트 통과를 회귀 통과조건으로 유지 |

현재 `README.md`의 환경은 Python 3.12, `uv`, PyTorch 2.14.1+cu132(x86 Linux), ModelOpt 0.47.0이며, *실제 사용 중인 프로젝트 lockfile*을 우선한다. 호환성 검증 없이 버전을 최신으로 일괄 교체하지 않는다. Jetson에는 `uv.lock`의 x86 CUDA Torch wheel을 설치하지 않는다.

### 2.1 설정/체크포인트 호환성에 관한 결정

현행 NAFNet checkpoint 검증은 `checkpoint["config"]["model"]`과 런타임의 `config["model"]`을 비교한다. 따라서 **현재 `model` 객체에 `arch` 등의 키를 직접 주입하면 기존 checkpoint의 strict metadata 검증이 깨질 수 있다.**

새 설계에서는 최상위 `architecture`와 `trt_r2r_light`를 추가한다.

- 기본값 `architecture="nafnet"`: 기존 `config["model"]` 그대로 `NAFNetLocal` 생성.
- `architecture="trt-r2r-light"`: `config["trt_r2r_light"]`만 새 모델 생성자에 전달.
- 신규 체크포인트: `architecture`, `task_order`, `model_spec`, `bank_fingerprint`, `config`, `state_dict`, QAT의 `modelopt_state` 저장.
- 이전 NAFNet 가중치에서 `architecture` 필드가 없으면 **레거시 NAFNet**으로 판단. 새 아키텍처는 이 자동 판단을 사용하지 않음.
- 모델 factory 분기에 registry/framework 도입 불필요. 명시적 `if/elif`가 적합.

---

## 3. TRT-R2R-Light v0.1 아키텍처

### 3.1 영상 처리 데이터 흐름

```text
RGB [B,3,H,W]   (H/W는 16의 배수로 외부 패딩)
 ├───────────────────────────────────────────────────┐ image residual
 ▼                                                   │
PixelUnshuffle(2) -> [B,12,H/2,W/2]                   │
 ▼                                                   │
Stem 3×3: 12→32                                      │
 ▼                                                   │
E0: 32ch,  LiteResidual×1 ---------------- skip0      │
 ▼                                                   │
Down0 3×3,s2: 32→64                                 │
 ▼                                                   │
E1: 64ch,  LiteResidual×1 ---------------- skip1      │
 ▼                                                   │
Down1 3×3,s2: 64→128                                │
 ▼                                                   │
E2: 128ch, LiteResidual×2 ---------------- skip2      │
 ▼                                                   │
Down2 3×3,s2: 128→192                               │
 ▼                                                   │
Bottleneck: 192ch, LiteResidual×2                     │
 ├─────────► GAP→FC(192→64)→ Query [B,64]            │
 │                          │                         │
 │                  Compact Bank Retrieval            │
 │                          │                         │
 │                    Prior [B,192]                   │
 ▼                          │                         │
                   FP Add at H/16,W/16                │
                           │                         │
              Up2 (1×1 Conv + Nearest×2)             │
                 + skip2 -> D2 Res×1                 │
                           │                         │
              Up1 (1×1 Conv + Nearest×2)             │
                 + skip1 -> D1 Res×1                 │
                           │                         │
              Up0 (1×1 Conv + Nearest×2)             │
                 + skip0 -> D0 Res×1                 │
                           │                         │
                Tail 3×3: 32→12                      │
                           │                         │
                PixelShuffle(2) -> RGB                 │
                           │                         │
                     + Input RGB ◄────────────────────┘
                           │
                    Restored RGB
```

모든 Conv가 매 추론에 실행되는 **dense shared-backbone** 구조다. Degradation Bank는 계산 경로를 끄거나 Expert 네트워크를 선택하지 않는다. **MoE식 sparse conditional execution이 아니라, 검색된 열화 Prior로 동일한 복원 Backbone을 조건화(conditioning)**한다.

### 3.2 입력/출력 계약

| 항목 | 계약 |
|---|---|
| 입력 | `float32` 또는 타깃 명세에 따라 `float16`, `[B,3,H,W]`, RGB, linear하지 않은 일반 이미지 코드값 `[0,1]` |
| 추론 배치 | 초기 B=1 고정 |
| 패딩 | 모델 외부에서 오른쪽/아래쪽을 **16의 배수**로 pad. pad mode 명시, 학습·평가·배포 일치 |
| 1080p 예시 | `1920×1080` → `1920×1088`, 복원 후 원래 `1920×1080`으로 crop |
| 기본 출력 | **RGB 이미지만** `[B,3,H,W]` 반환 (`forward` 계약) |
| 학습 보조 출력 | `forward_with_aux`에서 `(restored, task_logits, 진단용 attention)` 반환 |
| 색상 관리 | BGR↔RGB 변환 명시, ImageNet mean/std 사용하지 않음 |
| 범위 처리 | 학습 손실 이전 강제 clip 금지; 저장 시 기존 `write_rgb`의 clamp/quantize 정책 사용 |
| 복합 열화 | 입력 하나에 여러 task label이 동시에 1이 될 수 있음 |

기존 참조 프로토타입은 `forward`가 `(restored, confidence)`를 반환하지만, **이대로 기존 `training/loop.py`, `inference.py`, `evaluation/quality.py`에 연결하면 구조 충돌이 발생한다.** 따라서 production 새 모델은 `forward(x) -> restored`로 고정하고, 별도 `forward_with_aux`를 제공한다.

### 3.3 레이어 정의

```python
class LiteResidual(nn.Module):
    # x + Conv1x1(ReLU(Conv3x3(x)))
    # Normalization/attention/deformable op 없음
    ...


class CompactDegradationBank(nn.Module):
    # Key:   [D*K, 64]
    # Value: [D*K, 192]
    # task_gate: Linear(64, D)
    ...


class TrtR2RLight(nn.Module):
    def forward(self, image: Tensor) -> Tensor:
        """복원된 RGB Tensor 하나만 반환."""
        ...

    def forward_with_aux(self, image: Tensor):
        """(restored, task_logits, optional_attention) 반환; 학습/진단 전용."""
        ...
```

`forward()`와 `forward_with_aux()`는 내부 `_forward_impl`을 공유해 동일한 복원 계산을 수행하도록 한다. ONNX Export는 반드시 단일 RGB 출력을 호출한다.

### 3.4 공간/채널 명세

| Stage | 채널 | 공간 크기 | 블록 |
|---|---:|---|---|
| 입력 | 3 | H×W | RGB |
| Shuffle/Stem | 12→32 | H/2×W/2 | PixelUnshuffle+Conv3 |
| E0 | 32 | H/2×W/2 | Res×1 |
| E1 | 64 | H/4×W/4 | Down+Res×1 |
| E2 | 128 | H/8×W/8 | Down+Res×2 |
| Bottleneck | 192 | H/16×W/16 | Down+Res×2 |
| Query/Bank | 64 key / 192 value | vector | GAP/FC/MatMul/Softmax |
| D2 | 128 | H/8×W/8 | Conv1, Nearest, Skip, Res×1 |
| D1 | 64 | H/4×W/4 | Conv1, Nearest, Skip, Res×1 |
| D0 | 32 | H/2×W/2 | Conv1, Nearest, Skip, Res×1 |
| Tail | 12→3 | H/2 → H | Conv3, PixelShuffle, Image Add |

채널 32/64/128/192는 Tensor Core 친화적인 **후보**일 뿐, 전 레이어 INT8 최적 tactic이나 성능 향상을 보장하지 않는다. 일반 Conv를 우선 채택하고 Depthwise Conv는 지연시간 측정 후 별도 실험한다.

위 구성의 **이론 Conv MACs 약 54.41G**는 Batch 1, 패딩된 1088×1920 해상도의 3×3/1×1/Down/Upsample Projection Conv 합계다. Query/Bank, Shuffle, Add, 메모리 이동은 제외되어 있다. 동일 프로토타입의 전체 파라미터 수는 약 **1.71M**이다. 두 숫자 모두 **Orin 실측 성능이 아니다.**

### 3.5 PixelUnshuffle 비교 실험

이 구조가 1080p에서 유리할지는 ONNX로 Export한 이후 TensorRT tactic에 달려 있다. 최소한 다음 두 후보를 동일 데이터/학습 예산으로 비교한다.

- **A:** 입력 원해상도 Conv Stem + 일반적인 stride downsampling.
- **B:** `PixelUnshuffle(2)` + 1/2 해상도 Stem (본 설계).

특히 `PixelUnshuffle`이 ONNX의 `SpaceToDepth` 또는 Shuffle/Reshape/Transpose 연산열로 출력될 수 있으므로, **타깃 TensorRT Parser가 받아들이는지와 Shuffle 비용**을 검증한다. 변환 실패 시 전용 플러그인부터 만들지 말고 Stride Conv 대체 구조를 먼저 검토한다.

---

## 4. Compact Degradation Bank

### 4.1 Bank 크기와 태스크

최종 5-task 기본 순서 (체크포인트에 기록하고 임의 변경 금지):

```text
0 denoise
1 derain
2 dehaze
3 deblur
4 lowlight
```

**Clean은 별도의 Bank task가 아니다.** 모든 task label이 0인 데이터이며, 목적은 불필요한 복원 방지다.

| 텐서 | 기본 크기 (D=5, K=8) | 정밀도 초기 방침 |
|---|---|---|
| Query `q` | `[B,64]` | FP32 → 이후 FP16 비교 |
| Key `K` | `[40,64]` | FP32 → 이후 FP16 비교 |
| Value `V` | `[40,192]` | FP32 → 이후 FP16 비교 |
| Attention `a` | `[B,5,8]` | FP32/FP16 |
| Task logits `t` | `[B,5]` | FP32/FP16 |
| Prior `p` | `[B,192]` | FP32/FP16 |

Key/Value 합계 원소 수는 `(40×64)+(40×192)=10,240`개, 순수 FP16 값 저장량은 **20,480 bytes(20KiB)**다. 이는 실제 엔진 메모리·추가 workspace·중간 텐서 크기를 포함하지 않는다.

초기 2-task 실험에서 `D=2`로 시작할 수 있다. 다만 `D=2` checkpoint를 `D=5`로 **바로 strict-load할 수 없으므로**, 전체 모델을 처음부터 D=5로 학습하거나 명시적인 Bank 확장/가중치 이식 스크립트를 구현한다. 학습되지 않은 task prototype이 켜진 상태로 배포되지 않도록 한다.

### 4.2 고정 그래프 Retrieval 수식

Bottleneck 특징 `F∈R[B,192,H/16,W/16]`로 Query를 만든다.

```text
q = FC64(GlobalAveragePool(F))                      # [B,64]
logits = (q @ K.T) * (1/sqrt(64))                   # [B,40]
a = Softmax(logits.reshape(B,D,K), dim=slots)       # [B,D,K]
t = TaskGate(q)                                     # [B,D] raw logits
g = Sigmoid(t)                                      # [B,D]
w = reshape(a * g.unsqueeze(-1), [B,D*K])           # [B,40]
p = w @ V                                           # [B,192]
F' = F + alpha * p[:,:,None,None]                   # [B,192,H/16,W/16]
```

- 분류 손실에는 `t`를 사용해 `BCEWithLogitsLoss(t, multi_hot_label)` 계산. Sigmoid 결과에 BCEWithLogitsLoss를 다시 적용하면 안 된다.
- 초기 버전은 **TopK/Argmax/Gather/Einsum/가변 길이 Memory 갱신을 추론 그래프에서 사용하지 않는다.**
- FP16 Bank 경로는 추후 실험 목표이지, PyTorch에서 `.half()`를 넣는 것만으로 TensorRT에서 자동 달성됐다고 기록하지 않는다.
- `q`와 `K`의 매번 L2 정규화는 초기에 생략하고 scaled dot product로 시작한다. 정규화/온도 학습은 Retrieval 성능 부족 시 추가 실험한다.
- `alpha`는 초기 0.1의 작은 상수 또는 학습 가능한 제한된 스칼라로 시작하고, Bank 없는 기준 모델 대비 출력 안정성을 확인한다.

### 4.3 복합 열화의 의미

`derain+deblur` 이미지의 라벨은 `[0,1,0,1,0]` (5-task 순서일 때)이며 두 Gate가 동시에 높은 값을 가질 수 있다. 이는 실제로 **두 작업을 각각 수행하는 독립 Expert를 조건부 실행한다**는 뜻이 아니라, **Bank Prior를 가중 합성해 하나의 Decoder에 전달한다**는 뜻이다.

각 Task의 Gate가 낮더라도 Backbone이 Clean 이미지를 바꿀 수 있으므로 Clean 데이터에 대한 Identity Loss/평가를 필수로 둔다.

### 4.4 Bank 생성의 두 경로

**V0 (필수):** Trainable Prototype Bank

- Key/Value를 작은 난수로 초기화하고 전체 Backbone과 함께 학습.
- 작업별 BCE 보조손실과 Retrieval entropy/diversity 관측을 도입.
- PTQ 전에는 Key/Value를 freeze하고 체크포인트 및 fingerprint에 확정값 기록.
- 단일 모델의 End-to-End 검증과 Export 위험 제거에 적합.

**V1 (R2R 아이디어 충실화, 후속):** GT-guided Bank Teacher + Distillation

1. 학습 전용 `MemoryTeacher`가 `clean`과 `degraded-clean`으로부터 열화 표현을 인코딩한다.
2. 열화 유형별로 Key/Value 후보를 수집한다. **Training split의 GT만 사용**하며 Validation/Test GT는 Bank 생성에 사용하지 않는다.
3. 후보를 태스크별 K=8 Prototype으로 압축한다(예: torch 기반 클러스터링/EMA/학습 가능한 Prototype).
4. Query Head와 Bank를 대조학습·증류로 학습시킨다.
5. 최종 Key/Value와 복원 모델만 checkpoint에 저장. **MemoryTeacher를 ONNX 그래프에 포함하지 않는다.**
6. Bank를 고정한 후 PTQ/QAT 진행.

이 V1은 원본 R2R의 핵심 아이디어를 활용하기 위한 별도 연구 경로다. V0가 구현되기 전에 V1의 Teacher/3D Conv/복잡한 Memory 관리부터 만들지 않는다.

### 4.5 Bank의 배포 불변성

- 추론 시 Bank 크기와 task 순서는 고정; forward 안에서 리스트 append/pop, online update 금지.
- 모델의 `state_dict`에 저장하고 ONNX Export 시 initializer로 포함. 학습 중 Parameter로 유지하더라도 QAT 이전 `requires_grad=False`로 고정 가능.
- Export 이후 Bank 값이 바뀌면 모델/ONNX/엔진을 **다시 생성**한다. 런타임 Bank hot-swap은 v0 범위 제외.
- Zero-sized bank, 비정상적 NaN/Inf, 동일 task 중복, slots 수 불일치: 조용히 fallback하지 말고 실패.

---

## 5. 학습 데이터 및 레이블 계약

### 5.1 디렉터리 구조: 데이터 Manifest 생성 금지 정책 유지

`modern-nafnet`은 데이터 쌍을 디렉터리에서 직접 찾는다. 새 복원 데이터도 같은 철학을 유지하되, 기존 `blur/sharp`에 묶이지 않고 `input/target`을 사용한다.

```text
data/restoration/
  train/
    derain/
      input/<same relative filename>.png
      target/<same relative filename>.png
    deblur/
      input/<same relative filename>.png
      target/<same relative filename>.png
    denoise/input/... , target/...
    dehaze/input/...  , target/...
    lowlight/input/..., target/...
    clean/target/...                  # input=target (명시적 Clean branch)
    derain+deblur/input/... , target/...  # 합성/실측 복합 열화
  val/
    derain/input/... , target/...
    deblur/input/... , target/...
    clean/target/...
  test/
    ...
  calibration/
    derain/input/...
    deblur/input/...
    denoise/input/...
    dehaze/input/...
    lowlight/input/...
    derain+deblur/input/...
    clean/input/...
```

규칙:

1. `input`과 `target`은 **상대 경로와 확장자까지 일치**해야 함. 순서 기반 자동 pairing 금지.
2. 폴더명은 정의된 단일 task 또는 `+`로 연결된 multi-task 조합만 허용(예: `derain+deblur`).
3. `clean`은 항상 multi-hot **전부 0**. 대상 이미지를 그대로 입력으로 복제하며 학습에서 복원 identity를 감독.
4. 오염된 샘플, 누락 GT, 중복 상대경로, 해상도 불일치는 예외 처리. 자동 resizing·silent skip 금지.
5. 복합 열화 샘플은 **해당 복합 열화에 대한 깨끗한 GT**가 있는 경우만 paired로 취급.
6. 촬영 시퀀스/장소/날짜 기준 분리. 인접 프레임이 train/val/test에 나뉘지 않도록 함.
7. 기존 GoPro `data/deblur/{train,val,...}/{blur,sharp}` 및 legacy checkpoint 동작은 그대로 유지.

`src/modern_nafnet/data/sources.py`의 기존 `load_samples()`는 변경 없이 호환 동작시킨다. 범용 `input/target` 스캔은 **새 함수 또는 공통 내부 헬퍼**로 구현하고, 반환 키를 모호하게 `blur/sharp`로 재사용하지 않는다.

### 5.2 Batch와 샘플링

- 새 Dataset 반환: `{"input": Tensor, "target": Tensor, "task_label": FloatTensor[D], "name": str, "task": str}`.
- 기존 `PairedImageDataset`의 RGB/contiguous/`[0,1]`/AlbumentationsX augment 및 worker 초기화 원칙을 재사용.
- 학습 batch는 task별 균형 샘플링 또는 명시적 가중치로 구성. 전체 태스크 데이터량에 비례한 무조건 샘플링은 큰 task의 학습 편향을 유발할 수 있음.
- DDP sampler에서 각 rank의 task 분포를 추적. 기존 epoch 경계 resume 정책, RNG/데이터 fingerprint를 유지.
- 복합 열화 합성은 Clean GT에서 별도 생성하고 **열화 적용 순서·강도**를 기록. 실제 paired 입력에 임의 열화를 추가하여 레이블의 의미를 바꾸는 동작은 기본 금지.
- Crop은 256×256부터 시작하되 Deblur의 공간적 수용영역 부족이 보이면 384/512와 다중 스케일 학습을 실험.

### 5.3 PTQ Calibration

- 모든 배포 대상 task·심도·Clean·대표 복합 열화를 포함.
- `calibration/`은 학습 augmentation 미사용, Batch 1, Export Shape와 일치한 이미지로 사전 준비.
- 현행 `training/quantize.py::calibration_inputs()`는 **blur-only**를 가정하므로 새 MultiTask 경로로 분리해야 함.
- 1차 실험: 512장 수준의 균형 대표 샘플을 후보로 정하고, 128/512/1024장 결과를 비교. 특정 수량이 정확도 보장값은 아님.
- 입력 dtype·shape·range·task 수·파일 목록 fingerprint와 activation `amax`를 manifest에 기록.

---

## 6. Loss와 학습 단계

### 6.1 권장 Loss

`y_hat`은 restored, `y`는 target, `t`는 raw logits, `m`은 D차원 multi-hot label.

```text
L_restore = L_charbonnier(y_hat,y)
          + λ_edge * L_edge(y_hat,y)
          + λ_ms   * L_multiscale(y_hat,y)

L_task = BCEWithLogitsLoss(t,m)
L_clean = 1[m.sum()==0] * L_identity(y_hat,input)
L_total = L_restore + λ_task*L_task + λ_clean*L_clean
```

- 시작점 예시: `λ_edge=0.05`, `λ_ms=0.1`, `λ_task=0.1`, `λ_clean=0.5`. **튜닝 시작값이지 최적값이 아님**.
- PSNRLoss는 기존 Deblur NAFNet 경로에 그대로 유지.
- 학습용 Edge/Multiscale Loss는 ONNX 그래프에 포함되지 않음.
- 복합 열화에서는 Clean/Degradation 분류 정확도와 픽셀 복원 품질을 모두 기록.
- bank key/value의 collapse, 한 작업 게이트 독점, Clean false positive 비율 등을 진단 지표로 관측.

### 6.2 개발/학습 Phase

| Phase | 범위 | 완료 기준 |
|---|---|---|
| P0 | Bank **없는** 공유 Residual U-Net, Derain/Deblur/Clean | 입출력·학습·ONNX FP Export·기준 화질/속도 |
| P1 | **V0 Compact Bank** 추가, 2-task 라우팅 | Bank on/off 비교; 혼동·품질·추론 overhead 기록 |
| P2 | 5-task All-in-One 학습 | 작업별·평균 지표, Clean 보존, severe-case 분석 |
| P3 | 복합 열화 학습 | `derain+deblur` 등 unseen/seen 조합 별도 평가 |
| P4 | 선택적 GT-guided Teacher 및 증류 | V0 대비 실제 이득 확인 후 유지 여부 결정 |
| P5 | INT8 PTQ→QAT→ONNX→Orin TRT | 실제 INT8 Conv tactic·기능·정량 비교 통과 |

P5의 초기 통합 Smoke Test는 P0/P1과 병행해 조기 수행한다. 학습이 끝날 때까지 ONNX/INT8 호환성을 방치하지 않는다.

### 6.3 기존 NAFNet Teacher 활용

- 원본 NAFNet GoPro FP32 checkpoint는 **Deblur Teacher**로 선택적으로 활용한다.
- 새 네트워크에 weight key를 억지로 재매핑하지 않는다.
- 정답 `target` 외에 Teacher 출력에 대한 지식증류 loss를 추가하는 것은 옵션. Teacher가 Derain 등 다른 열화의 정답을 제공한다고 가정하지 않는다.
- Distillation 사용 여부에 따른 일반화 성능은 독립 Ablation으로 측정.

---

## 7. TensorRT INT8 설계 — 가장 중요한 위험 구간

### 7.1 목표가 아닌 것

- PyTorch가 fake-quantize 한다고 해서 TensorRT에서 INT8 커널이 선택됐다고 주장하지 않음.
- ONNX에 `QuantizeLinear`/`DequantizeLinear`가 있다고 해서 네트워크 전체가 INT8로 실행된다고 판단하지 않음.
- `trtexec` 엔진 빌드 성공만으로 성능 개선을 인정하지 않음.
- 모든 연산을 무리하게 INT8로 강제하지 않음. Retrieval/소규모 Gate는 고정밀도가 오히려 유리할 수 있음.

### 7.2 단계적 정밀도 정책

**Precision Stage A (필수; 가장 먼저 구현):**

```text
input / Stem                 FP32 (비양자화)
Backbone Conv               INT8 Q/DQ W8A8
Skip/Add                    그래프 분석 후 선택적 Q/DQ
Query/Bank/TaskGate         FP32 (비양자화)
Prior Injection             FP32 (비양자화)
Decoder Conv                INT8 Q/DQ W8A8
Tail/Image Add              FP32 (비양자화)
```

이는 기존 저장소의 FP32 ONNX/QAT 경로와 호환성을 우선하는 출발점이다. Q/DQ 경계에서 FP 전환이 생기므로 실제 Latency는 프로파일링해야 한다.

**Precision Stage B (후속; 선택적):** 비양자화 구간의 데이터 타입을 명시적 FP16으로 설정한 **Typed Mixed Precision ONNX**로 변환. 핵심 조건:

- Export에서 기존 `export_model()`의 무조건 `model.to(device).float()`를 모델별 정밀도 정책으로 대체.
- FP16으로 정의한 구간의 ONNX Weight/Input/Cast, Q/DQ Scale dtype을 확인.
- `--stronglyTyped` 적용 시 `--fp16`/`--int8` 빌더 플래그를 혼용하지 않음. **온전히 ONNX 그래프의 타입과 Q/DQ로 표현**.
- 수동 half-cast가 ModelOpt symbolic Q/DQ와 충돌할 경우 Stage A를 안전한 배포 기준선으로 유지. 지원을 추정하거나 post-export cast만 임의 삽입하지 않음.
- FP16을 위한 실험용 exporter/graph transform을 별도 테스트해 ONNX Runtime와 TensorRT 수치 동등성을 확인.

### 7.3 Q/DQ 정책 표

| 대상 | PTQ/QAT | 기본 방식 | 주의 |
|---|---|---|---|
| 일반 `Conv2d` 가중치 | Quantize | signed INT8, per-output-channel(axis=0), zero-point=0 | 실제 `amax` 확인 |
| 일반 `Conv2d` 입력 | Quantize | signed INT8, per-tensor, zero-point=0 | Calibration 대표성 |
| Stem Conv | 초기 Exclude | FP32→FP16 후보 | 미세 입력 특징 손실 |
| Tail Conv | 초기 Exclude | FP32→FP16 후보 | 작은 residual 보정값 보존 |
| Query Linear | Exclude | FP32/FP16 | 소규모 연산 |
| Bank Key/Value | Exclude | FP32/FP16 initializer | 버퍼 고정, 순서 일치 |
| Softmax/Sigmoid | Exclude | FP32/FP16 | INT8 강제 불필요 |
| Residual Add | 별도 테스트 | Conv와 Add 사이 Q/DQ 위치 조정 | skip 입력 정밀도와 fusion 의존 |
| Prior Injection Add | 별도 고정밀도 경계 | Bottleneck에서 1회 | Broadcast 및 requantization 비용 |
| Nearest Upsample/Shuffle | 비가중치 연산 | 실제 엔진 format 측정 | Shuffle/Reformat overhead |

현행 `int8_config()`는 `.sca.*` attention Conv만 예외 처리하고, `check_int8_convs()`는 대부분의 Conv가 W8A8이라고 가정한다. **이 로직을 통째로 새 모델에 재사용하면 Stem/Tail 제외 정책과 충돌**한다.

수정 지침:

1. `int8_config(architecture, precision_policy)`로 정책을 분리. 기존 NAFNet 정책은 default로 보존.
2. 명명된 새 모델 Conv의 `input_quantizer`/`weight_quantizer`만 활성화하고, Stem/Tail/Bank/Query quantizer 제외.
3. `check_int8_convs(model, policy)`로 **INT8 기대 모듈**과 **FP 기대 모듈**을 구분해 검증. 모델 종류별 hardcode된 `.sca.` 검사에만 의존하지 않음.
4. `amax` 유한성/shape/axis, `is_enabled`, `num_bits`, skipped layer 수를 JSON 리포트에 저장.
5. Residual Add의 경우 **두 경로의 Q/DQ 위치**를 실험. FP Skip이 Conv+Add 융합을 방해할 수 있으며, 무조건 같은 scale로 강제하면 복원 화질이 훼손될 수 있다.
6. Fusion의 효과는 `layers.json`의 실제 layer precision/format/tactic으로 판단.

### 7.4 ONNX의 구현 제한

- Quantized ONNX는 `QuantizeLinear`/`DequantizeLinear` 형식을 사용. `QLinearConv`, `ConvInteger` 경로로 전환하지 않는다.
- 기존 `deployment/export.py`는 FP의 `dynamo=True`, ModelOpt Q/DQ의 `dynamo=False`, 기본 Opset **18**이다. **현재 재현되는 조합부터 유지**하고, 타깃 TensorRT에 맞춰만 변경.
- `TopK`, 런타임 bank update, 텐서에 대한 Python List append, 동적 task-level 분기는 최종 Export 그래프에서 제외.
- Bank 수식은 `MatMul`, `Reshape`, `Softmax`, `Sigmoid`, `Mul`, `ReduceMean`, `Add` 중심으로 제한. `einsum`으로부터 예상 못한 ONNX 노드가 생성되지 않게 한다.
- Model의 외부 패딩 계약을 manifest에 기록한다. 그래프 자체에 dynamic `F.pad`를 넣지 않는 것이 초기 검증에는 단순함.
- TensorRT로 변환 실패한 layer가 나오면 **PyTorch → ONNX 노드 → TensorRT Parser/지원 타입 → 해결** 순서로 문제를 분리해 기록한다.

### 7.5 Export/Manifest 추가 필드

기존 `.onnx.json`에 최소한 다음 필드를 확장한다.

```json
{
  "architecture": "trt-r2r-light",
  "model_revision": "v0.1",
  "task_order": ["denoise", "derain", "dehaze", "deblur", "lowlight"],
  "bank": {"slots_per_task": 8, "key_dim": 64, "value_dim": 192, "frozen": true},
  "precision_policy": "w8a8-conv-fp32-branch",
  "input": {"shape": [1, 3, 1088, 1920], "color": "RGB", "range": [0, 1]},
  "preprocess": {"pad_multiple": 16, "mode": "constant", "pad_value": 0},
  "postprocess": {"crop_to_original_size": true},
  "quantization": {"format": "ONNX-QDQ", "weight_axis": 0, "activation_axis": null},
  "tensorrt_verified": false
}
```

위 JSON은 **추가 필드의 스키마 예시**다. 실제 JSON에는 기존 exporter의 SHA-256, Opset, 환경, 원본 체크포인트 fingerprint도 그대로 보존한다. `pad_mode`는 학습·export·Orin 전처리에서 일치시킨다. 각 task 순서가 다른 모델의 Bank를 실수로 로드하지 않도록 `task_order`를 체크포인트에서 엄격하게 검증한다.

---

## 8. 구체적인 파일별 구현 계획

```text
modern-nafnet/
├── configs/
│   ├── default.json                         # 수정: 새 top-level 설정
│   └── trt-r2r-light.json                   # 신규: 예제 설정
├── docs/
│   └── TRT_R2R_LIGHT_IMPLEMENTATION.md      # 본 문서 위치 권장
├── src/modern_nafnet/
│   ├── config.py                            # 수정: 모델별 설정 검증
│   ├── cli.py                               # 수정: 설명/명령별 모델 분기(가능한 최소)
│   ├── models/
│   │   ├── nafnet.py                        # 원형 보존
│   │   ├── factory.py                       # 신규: 명시적 생성 함수
│   │   └── trt_r2r_light.py                 # 신규: Residual/CNN/Query/Bank
│   ├── data/
│   │   ├── sources.py                       # 수정: 범용 input/target 인덱스
│   │   └── multitask.py                     # 신규: Label/Dataset/Sampler 관련
│   ├── training/
│   │   ├── loop.py                          # 수정: model factory, multitask loss, DDP
│   │   ├── losses.py                        # 수정: 새 복원·분류 loss
│   │   ├── quantize.py                      # 수정: 모델별 정책·mixed 검증
│   │   └── bank_teacher.py                  # 후속 V1에서만 추가
│   ├── evaluation/
│   │   └── quality.py                       # 수정: per-task/clean/mixed 집계
│   ├── deployment/
│   │   └── export.py                        # 수정: 새 모델·manifest·정밀도 정책
│   └── inference.py                         # 수정: factory 연동
└── tests/
    ├── test_trt_r2r_light.py                # 신규
    ├── test_multitask.py                    # 신규
    ├── test_r2r_quantize.py                 # 신규
    ├── test_r2r_export.py                   # 신규
    └── test_r2r_compatibility.py            # 신규: NAFNet 회귀
```

### 8.1 모델 Factory 계약

```python
# src/modern_nafnet/models/factory.py


def build_model_from_config(config: dict):
    arch = config.get("architecture", "nafnet")
    if arch == "nafnet":
        return NAFNetLocal(**config["model"])
    if arch == "trt-r2r-light":
        return TrtR2RLight(**config["trt_r2r_light"])
    raise ValueError(f"Unsupported architecture: {arch}")
```

실제 구현 시 import는 선택 branch에서 지연하여 CLI help/기본 package import가 Torch를 요구하지 않도록 한다. `training/loop.py`, `inference.py`, `evaluation/quality.py`, `deployment/export.py`의 **직접 `NAFNetLocal(...)` 호출을 먼저 조사하고 공통 factory로 교체**한다.

### 8.2 Config 설계: 모델별 객체 분리

```json
{
  "architecture": "trt-r2r-light",
  "trt_r2r_light": {
    "tasks": ["denoise", "derain", "dehaze", "deblur", "lowlight"],
    "slots_per_task": 8,
    "query_dim": 64,
    "channels": [32, 64, 128, 192],
    "encoder_blocks": [1, 1, 2, 2],
    "decoder_blocks": [1, 1, 1],
    "bank_enabled": true,
    "prior_scale_init": 0.1
  },
  "data": {"path": "data/restoration", "crop_size": 256},
  "training": {
    "checkpoint": null,
    "output_dir": "artifacts/trt-r2r-light/train",
    "steps": 100000,
    "batch_size": 4,
    "lr": 0.0001
  },
  "quantization": {"calibration_batches": 512},
  "export": {
    "output": "artifacts/trt-r2r-light/model-int8.onnx",
    "shape": [1, 3, 256, 256],
    "opset": 18,
    "quantized": true
  }
}
```

- 위 설정은 **학습 예제**이며, 100,000 steps/Batch 4가 특정 품질을 보장한다는 의미는 아니다.
- 기본 `model` 객체는 바꾸지 않고 `architecture`에 따라 무시/사용 대상을 구분.
- 기존 `config.py`는 리스트를 **정수 리스트로만 검증**하므로 `tasks: list[str]` 예외와 task 중복 검사, enum 검사, 채널 길이/32배수 정책, encoder/decoder 블록 길이를 추가해야 한다.
- 추가 설정이 `DEFAULT_CONFIG`에 존재하지 않으면 현재 `_merge()`에서 오류가 나므로 `default.json`과 파이썬 `DEFAULT_CONFIG` 둘 다 동기화한다.
- `training.checkpoint = null`로 **새 아키텍처만 from-scratch 학습 허용**. 기존 NAFNet은 원래의 `checkpoint` 요구 정책을 유지한다.
- `quantization.calibration_batches`는 최대 수량이며, 클래스 균형 샘플 선정을 별도 로직으로 구현해야 한다.
- 1080p 엔진은 검증용 256² ONNX와 **별도 Export**하고 정확히 그 크기로 Calibration한다.

### 8.3 CLI

아키텍처명을 실행 명령으로 사용하지 말고 config로 선택한다. 기존 명령을 그대로 재사용한다.

```bash
# 브랜치 준비 (사용자가 실행)
git fetch origin
git switch main
git pull --ff-only
git switch -c trt-r2r-light

# 현행 테스트가 통과하는 상태에서 시작
uv sync --locked --all-extras
uv run --locked --all-extras pytest -q
uv run --locked ruff check .

# 다음 명령은 본 설계를 구현한 이후 실행 가능해야 하는 목표 계약
uv run --locked --extra train modern-nafnet train --config configs/trt-r2r-light.json
uv run --locked --extra qat modern-nafnet qat --config configs/trt-r2r-light.json \
  --checkpoint artifacts/trt-r2r-light/train/last.pt --steps 0
uv run --locked --extra qat modern-nafnet qat --config configs/trt-r2r-light.json \
  --checkpoint artifacts/trt-r2r-light/train/ptq.pt
uv run --locked --extra export --extra qat modern-nafnet export \
  --config configs/trt-r2r-light.json \
  --checkpoint artifacts/trt-r2r-light/train/last.pt \
  --quantized --height 256 --width 256 \
  --output artifacts/trt-r2r-light/model-int8.onnx
```

**주의:** `qat --steps 0`의 산출 경로와 일반 QAT `last.pt`는 실제 학습 실행의 `output_dir`/artifact 정책에 따라 확정해야 한다. 위는 브랜치 구현 후 만족시킬 CLI 계약 예시다. 실행 순서상 PTQ/QAT checkpoint 덮어쓰기 방지와 artifact 기록을 필수로 한다. 기존 `--resume`은 epoch 경계 재개이며 world size 제약을 유지한다.

---

## 9. Orin 배포와 엔진 분석

### 9.1 우선 대상 버전과 환경 확인

AGX Orin의 TensorRT 버전은 JetPack/L4T에 따라 달라진다. 예를 들어 JetPack 6.2.1 (L4T 36.4.4)은 TensorRT 10.3.0을 포함한다. **엔진을 만들 Orin에서 확인한 버전**을 기준으로 Export Opset·빌드 인자를 결정한다.

```bash
cat /etc/nv_tegra_release
/usr/src/tensorrt/bin/trtexec --version || trtexec --version
which trtexec || true
nvidia-smi || true         # Jetson 환경에 따라 없을 수 있음; tegrastats 사용 가능
```

현재 특정 보드의 SDK 설치·`trtexec` 경로는 이 문서 작성 중 직접 확인되지 않았다. `trtexec`가 없다면 시스템 TensorRT/JetPack 설치를 별도로 복구해야 하며, `pip`의 다른 TensorRT wheel로 대체해 성공이라고 판단하지 않는다.

### 9.2 정밀도별 엔진 빌드

강타이핑 Q/DQ ONNX용 기본 명령:

```bash
trtexec \
  --onnx=trt-r2r-light-int8.onnx \
  --saveEngine=trt-r2r-light-int8.plan \
  --stronglyTyped \
  --profilingVerbosity=detailed \
  --exportLayerInfo=trt-r2r-light-layers.json
```

엔진 검사:

```bash
trtexec \
  --loadEngine=trt-r2r-light-int8.plan \
  --profilingVerbosity=detailed \
  --dumpLayerInfo --dumpProfile --separateProfileRun \
  --exportLayerInfo=trt-r2r-light-layers.json \
  --noDataTransfers --useCudaGraph
```

타깃 `trtexec --help`에 실제 옵션이 있는지 먼저 확인한다. 순수 GPU 계산시간과 host→device, 전처리·후처리·이미지 전송을 포함한 end-to-end 시간을 **분리 측정**한다.

### 9.3 필수 Engine 검증 리포트

1. 각 Conv의 입력/출력 dtype, Tensor Format, tactic, layer fusion.
2. 예상 밖의 FP32/FP16 Conv 목록과 누락된 INT8 Conv 목록.
3. `Reformat`, `Cast`, `Copy`, `Shuffle`, `Resize`, `ElementWise`별 지연시간과 비중.
4. Query/Bank FP 경로의 GEMM, Softmax 및 Bottleneck Q/DQ 비용.
5. 256², 720p, 패딩된 1088×1920에서 Batch1 지연시간/메모리.
6. 고정 nvpmodel/클럭, 온도와 thermal throttling, 실제 전력 조건 기록.
7. FP32 PyTorch, FP32/FP16 TensorRT, INT8 PTQ TensorRT, INT8 QAT TensorRT 정확도 비교.

DLA와 2:4 sparsity는 첫 번째 릴리스 범위에서 제외한다. GPU INT8 엔진의 정확도와 성능이 재현된 뒤에만 별도로 실험한다.

---

## 10. 품질 검증과 Ablation Matrix

### 10.1 작업별 평가

- Derain: Rain100L 등 적절한 paired 벤치마크 및 자체 실제 촬영 데이터.
- Deblur: GoPro test를 포함하되 학습/평가 시퀀스 누수 금지.
- Denoise/Dehaze/Low-light: 해당 작업에 적합한 데이터와 별도의 품질·색상 왜곡 분석.
- Clean: 원본 영상의 불필요한 변형 방지(PSNR/SSIM 및 픽셀 변화).
- Composite: 최소 Derain+Deblur, 이후 Rain+Noise, Blur+Lowlight; **훈련에서 본 조합과 보지 못한 조합 구분**.

### 10.2 모델 Ablation

| 실험 | 변경 | 가설/검증 |
|---|---|---|
| A0 | NAFNet 기존 FP | Deblur 상한/Teacher 참조. 새 모델의 직접 가중치 기준선은 아님 |
| A1 | 동일 CNN, Bank 없음 | 공유 Backbone 단독 성능 |
| A2 | Prototype Bank, 단일 Bottleneck Add | Retrieval이 작업 간 간섭을 줄이는가 |
| A3 | Bank + Multi-label BCE | 자동 열화 인식 및 합성 열화 처리 개선 여부 |
| A4 | GT-guided Teacher-distilled Bank | 단순 Trainable Bank 대비 이점 |
| A5 | 추가 Injection/작은 spatial prior | 지역별 열화 처리 성능 대비 비용 |
| A6 | PixelUnshuffle 제거/대체 | Shuffle 비용 및 영상 품질 비교 |

### 10.3 배포 Ablation

| 실험 | 목적 |
|---|---|
| T0 FP32 PyTorch | Reference 출력 |
| T1 FP ONNX + ONNX Runtime | Export 동등성 |
| T2 FP32 TensorRT | 최적화 기준선 |
| T3 FP16 TensorRT (지원 방식 별도 검증) | 혼합정밀도 대비 속도 |
| T4 W8A8 PTQ + 고정밀 Bank | PTQ 화질/속도 |
| T5 W8A8 QAT + 고정밀 Bank | 정확도 회복 |
| T6 Typed FP16 Bank/Stem/Tail | Cast 경계·속도·화질 개선 여부 |
| T7 Q/DQ Residual Add 변형 | 융합 이점과 품질 영향 |

### 10.4 통과 기준 정의

**기능 게이트 — 반드시 통과:**

- `forward(x)` → 입력과 같은 `[B,3,H,W]`의 유한한 RGB 출력. Clean/Single/Composite에서 예외 없음.
- Bank 정적 shape, task 순서/라벨 고정, checkpoint strict load 및 재현되는 Export.
- 기존 `tests/` 회귀 100% 통과; 기존 NAFNet checkpoint와 `infer/train/qat/export` 동작 보존.
- Quantized ONNX checker 통과, Q/DQ 존재, 정의된 입력/출력/Bank shape 검사.
- Orin TensorRT 엔진 build 성공, 실제 기대 Conv의 INT8 tactic 확인.
- 같은 입력으로 PyTorch/ONNX/TRT 출력 비교, 지표 계산 및 차이 원인 기록.

**성능 게이트 — 초기 목표(협의 가능한 예비값):**

- FP 기준 대비 INT8 QAT의 PSNR 감소는 **우선 0.2 dB 이하**를 목표로 하되, task별/극단 사례별 품질을 우선 판단.
- Batch1 동일 해상도·전력조건에서 W8A8 엔진이 고정밀도 기준 엔진보다 **실측상 유의미하게 빠른지** 확인. 빠르지 않으면 Q/DQ·Shuffle·Reformat 비용을 분석해 구조를 재검토.
- 목표 FPS/전력은 제품 요구를 받은 뒤 별도 숫자로 고정. 현재 임의의 FPS를 성공 조건으로 적지 않음.

---

## 11. 테스트 설계와 CI

### Unit / CPU

- `LiteResidual` forward/backward 및 극단 입력(0, 1, 비정상 값) 처리.
- `[1,3,256,256]`, `[1,3,128,256]`에서 End-to-End shape. 16의 배수가 아닌 입력은 모델 경계에서 명시적으로 reject하고, 전처리의 pad/crop은 별도 테스트.
- D=2 및 D=5 Bank shape 검증, 슬롯별 Softmax 합=1, multi-label Sigmoid 독립성, NaN/Inf 방어.
- `forward()` 출력 타입이 **Tensor 하나인지** 확인. `forward_with_aux` task logit shape, Clean 0-label.
- Bank Freeze 후 모든 파라미터 `requires_grad` 정책, strict checkpoint 복원, bank/task-order fingerprint 체크.
- Task 폴더 상대경로 pairing, 누락/중복/다른 크기/알 수 없는 task/복합 라벨/빈 Clean 처리 오류 테스트.
- 모델/설정의 NAFNet legacy checkpoint/호환성 회귀 테스트.

### CUDA + ModelOpt

- 새 모델 PTQ Calibration, 최소 한 번 QAT backward, 활성화 `amax` 고정 확인.
- **양자화 대상으로 지정한 Conv**와 **제외한 Conv**의 정확한 `TensorQuantizer` 설정 검사.
- Calibration 데이터를 한 task에만 몰았을 때의 실패/경고 정책 검증.
- `modelopt_state` 포함 checkpoint 저장/복원/엄격한 model signature 확인.
- DDP rank 간 초기 Bank/Calibration 일관성, epoch 경계 resume 및 RNG 상태.

### ONNX

- FP32 그래프 shape inference, checker, ONNX Runtime와 PyTorch 수치 비교.
- Q/DQ 그래프에서 QuantizeLinear/DequantizeLinear 존재, 예상 Conv 주변에 Q/DQ 있는지 검사.
- `topk/argmax/gather/einsum` 미사용 정책 확인(ONNX node allowlist 검사, 필요한 예외는 기록).
- Batch1·256²와 고정 1088×1920 각각 Export 시 정확한 padding/metadata 검사.
- 불일치하는 `task_order`의 체크포인트/Bank 사용 시 명시적인 실패.

### Jetson 수동/자동 배포 검사

- `trtexec --stronglyTyped` Build, Inspector JSON 산출.
- 실제 INT8 Conv/Tactic, 예상하지 않은 Reformat 목록 작성.
- PSNR/SSIM 작업별 평가 및 end-to-end latency/전력 재현.

---

## 12. 단계별 구현 순서와 커밋 단위

| 단계 | 변경 범위 | 완료 조건 |
|---|---|---|
| **M0** | `trt-r2r-light` 브랜치 생성, 본 명세 추가, 기존 테스트 실행 | base SHA 및 기존 테스트 결과 기록 |
| **M1** | 모델 factory + `trt_r2r_light.py` + 새 모델 config | FP forward/backward, shape test, NAFNet 회귀 |
| **M2** | Task Dataset/폴더 매칭/라벨/샘플링 | Single/Clean/Composite 데이터 유닛 테스트 |
| **M3** | MultiTask train/eval/loss/DDP | Derain/Deblur/Clean 소규모 오버핏·평가 |
| **M4** | V0 Bank train/freeze + Bank-on/off 비교 | Task BCE와 Bank Ablation 리포트 |
| **M5** | ModelOpt W8A8 PTQ/QAT 정책·검증기 | conv별 활성화/비활성화, QAT backward/checkpoint |
| **M6** | FP/QDQ ONNX export + Task/Bank manifest | ORT 일치·ONNX Checker·타입 점검 |
| **M7** | Orin TensorRT 엔진·Inspector·벤치마크 | 실제 INT8 tactic·정량 화질/지연시간 |
| **M8** | 5-task/복합 열화 + 선택적 Teacher Bank | 데이터별 독립 리포트, 회귀 검증 |

개발상 최우선 원칙은 **M1에서 이미 FP ONNX smoke test**, **M3에서 조기 PTQ smoke test**, **M6에서 실제 Orin 빌드 smoke test**를 병렬로 검증하는 것이다. 학습을 오래 진행한 후에만 Parser 호환성을 확인하는 개발 순서는 피한다.

권장 초기 커밋 메시지:

```text
docs: specify TRT-R2R-Light architecture and INT8 deployment contract
feat: add TRT-R2R-Light shared restoration backbone
feat: add multitask paired dataset and degradation labels
feat: add compact degradation bank and auxiliary task training
feat: add model-specific W8A8 quantization policy
feat: export TRT-R2R-Light QDQ ONNX with deployment manifest
test: verify TensorRT layer precisions and restoration parity
```

---

## 13. 예상되는 실패 지점 및 대응

| 실패/위험 | 조기 발견 방법 | 대응 |
|---|---|---|
| 기존 NAFNet checkpoint 검사 실패 | legacy regression test | 최상위 `architecture` 분리, 원본 `model` 객체 보존 |
| 모델 출력 tuple 때문에 학습/평가 실패 | forward/type test | `forward`는 RGB Tensor, 보조 출력은 별도 API |
| PixelUnshuffle Parser/Shuffle 비용 | 256² ONNX+TRT smoke & Inspector | stride Conv 입력 variant 비교 |
| INT8 Conv가 실제 FP로 fallback | Inspector / `layers.json` | Q/DQ·Residual Add 정밀도 경계 검토 |
| Mix Precision `.float()` 때문에 FP16 상실 | ONNX 초기화 텐서 dtype 확인 | 모델별 typed export path, Stage A 우선 |
| Softmax/Bank 추론 병목 | layer-level profile | 슬롯 수 축소(8→4), Gate 동작 단순화, FP32↔FP16 실험 |
| Bank 쏠림/Task 혼동 | per-task confusion, slot usage | task-balanced sampling, BCE weight, diversity/distance 규제 |
| Clean에서 색 변화 | Clean PSNR 및 출력 차이 | Clean 샘플/identity loss, residual scale 조정 |
| Rain+Blur 동시 열화 실패 | composite hold-out | 실제 복합 데이터 추가, multi-label loss |
| PTQ 정확도 급락 | FP32/FP16/PTQ 비교 | Calibration 균형, QAT, Stem/Tail 및 민감 Add 정밀도 보호 |
| 낮은 Tensor Core 활용 | Conv channels/tactics/format | C32 정렬 조정, Block 폭/깊이 trade-off 측정 |
| 고정 Bank가 잘못 export | ONNX initializer fingerprint | checkpoint→export strict meta 검증 |
| 외부 코드 라이선스 불명확 | THIRD_PARTY_NOTICES 점검 | R2R 원본 코드 직접 복사 금지, 설계 아이디어 독립 구현 |

---

## 14. 라이선스, 출처 및 배포 책임

- `modern-nafnet` 프로젝트 자체는 **AGPL-3.0-only**. 기존 `LICENSE`, `THIRD_PARTY_NOTICES.md`, NAFNet 관련 고지 유지.
- 검토 시점의 `cscxwang/R2R` 저장소 최상위 목록에서는 독립 `LICENSE` 파일을 확인하지 못했다. **공개 저장소라는 사실만으로 소스 코드를 복제·재라이선스할 권한이 생기는 것은 아니다.** R2R 코드 직접 복제, 체크포인트 포함 또는 원본 구현 재배포가 필요하다면 저작권자 허락/라이선스를 별도 확인.
- 본 설계는 **Retrieval-based task-conditioning 아이디어를 참고한 독립 설계**이며, R2R의 클래스/함수 구현을 생산 코드에 복사하지 않는 것을 기본으로 한다.
- 외부 데이터셋/체크포인트의 라이선스·연구용 조건·상용 활용 제한은 각각 검토하고 기록.
- JetPack, TensorRT, ModelOpt 및 CUDA SDK의 실제 버전과 배포 조건은 target runtime manifest에 분리 기록.

---

## 15. 참조 문헌 및 확인한 구현

1. 기반 저장소 및 코드: <https://github.com/Longcat2957/modern-nafnet> (`main`, `e143393ea656036956eedd8f0ee32f5683a84407`). 확인한 핵심 파일: `README.md`, `AGENTS.md`, `configs/default.json`, `config.py`, `cli.py`, `models/nafnet.py`, `training/quantize.py`, `training/loop.py`, `deployment/export.py`, `data/sources.py` 및 테스트.
2. R2R: <https://github.com/cscxwang/R2R> (`main`, `bf387d56095aaf4edc0b685f8ea58cce5c64c2fc`). 확인 파일: `README.md`, `net/model_5D.py`, `net/feature_bank_5D.py`, `net/model_compos.py`, `net/feature_bank_compos.py`.
3. CVPR 2026 논문: <https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Retrieve-to-Restore_Efficient_All-in-One_Image_Restoration_with_a_Retrieval-Based_Degradation_Bank_CVPR_2026_paper.html>.
4. TensorRT Explicit Quantization / Q/DQ: <https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html>.
5. TensorRT Engine Inspector & Debug: <https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-tools.html>.
6. NVIDIA TensorRT 진단 명령: <https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/troubleshooting.html>.
7. NVIDIA Model Optimizer: <https://github.com/NVIDIA/Model-Optimizer>.
8. JetPack 6.2.1 / L4T 36.4.4의 TensorRT 버전: <https://docs.nvidia.com/jetson/jetpack/release-notes/index.html>.

---

## 최종 구현 판정

**이 설계서의 `trt-r2r-light`는 ‘한 모델로 여러 이미지를 복원’하는 기능이 우선이고, 그 다음 제약이 ‘TensorRT W8A8을 실제 커널까지 유지할 것’이다.** 따라서 원본 R2R의 공간형 Bank/NAFBlock/가변 Mask/TopK를 그대로 가져오지 않는다. 먼저 기존 `modern-nafnet`의 안정된 환경 안에서 **Residual CNN + 고정 벡터형 Bank + 다중 라벨 Gate + Bottleneck Add**를 구현한다. 성능 개선 여부가 검증된 요소만 이후 버전에 추가한다.

최종 성공은 ‘ONNX 파일 생성’이 아니라 다음의 동시 충족으로 정의한다.

> **(a)** Derain/Deblur/기타 대상 열화 및 복합 열화에서 복원 성능이 확인됨. **(b)** PyTorch–ONNX–TensorRT의 입력/출력 및 수치 경계가 재현됨. **(c)** AGX Orin의 엔진에서 기대한 Convolution이 실제 INT8 커널로 실행됨. **(d)** 품질 저하·latency·전력의 균형이 요구사항을 만족함. **(e)** 기존 NAFNet 코드·체크포인트·CLI·테스트를 깨지 않음.
