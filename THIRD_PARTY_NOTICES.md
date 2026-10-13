# Third-party notices

프로젝트 자체 코드는 AGPL-3.0-only로 공개한다. 아래 원저작물의 고지와
라이선스는 유지하며, 원문은 `licenses/NAFNet-BasicSR.txt`에 포함한다.

## NAFNet / BasicSR reference

- Repository: https://github.com/megvii-research/NAFNet
- Pinned commit: `2b4af71ebe098a92a75910c233a3965a3e93ede4`
- Copyright (c) 2022 megvii-model. All Rights Reserved.
- Copyright 2018-2020 BasicSR Authors.
- NAFNet: MIT; bundled BasicSR: Apache License 2.0.

| Upstream reference | Reimplementation | Changes |
|---|---|---|
| `basicsr/models/archs/NAFNet_arch.py` | `models/nafnet.py`, `models/layers.py` | Same architecture, module names and strict-load shapes; independent package without BasicSR imports |
| `basicsr/models/archs/arch_util.py` | `models/layers.py` | Channel-wise LayerNorm with standard autograd; FP32 statistics for half inputs; no custom backward |
| `basicsr/models/archs/local_arch.py` | `models/tlc.py`, `models/nafnet.py` | Fixed kernels equivalent to the train-size dummy forward; non-inplace cumsum, FP32 half-input accumulation; exact path only (`fast_imp=False`) |
| `basicsr/models/losses/losses.py` | `training/losses.py` | Imagewise logarithmic PSNRLoss, original constants; device-safe optional Y conversion and FP32 loss arithmetic |
| `basicsr/metrics/psnr_ssim.py` | `evaluation/metrics.py` | RGB/border-zero imagewise PSNR and channel-inclusive Gaussian SSIM3d; explicit device and input validation |

The upstream SSIM source also attributes
https://github.com/mayorx/matlab_ssim_pytorch_implementation/blob/main/calc_ssim.py.
This attribution is retained in `evaluation/metrics.py`.

Production files were reimplemented from the pinned references. The development
oracle snapshot is stored separately under ignored `artifacts/reference/`; it is
never imported by production code. Tests compare outputs and gradients against it.
The SSIM/PSNR range heuristic (`pred.max() <= 1`) is preserved for exact reference
compatibility, including the very dark uint8 image edge case.

## Official checkpoint

- Asset: NAFNet GoPro width32, linked by the pinned upstream README.
- Source: https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view
- SHA-256: `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`
- Local validation asset: `artifacts/checkpoints/NAFNet-GoPro-width32.pth` (ignored).

The checkpoint is an external asset; the project's own AGPL metadata does not
relicense it. The upstream repository contains the MIT/Apache notices above;
no separate checkpoint license notice was identified in its README. The weights
are downloaded for local validation and are not bundled in this source package.

## Dependencies

AlbumentationsX is selected under its AGPL-3.0-only license and imported using
`albumentations`. Other installed dependencies retain their respective licenses;
the resolved versions and origins are recorded in `uv.lock`.
