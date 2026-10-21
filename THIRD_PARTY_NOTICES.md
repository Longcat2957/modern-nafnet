# Third-party notices

프로젝트 자체 코드는 AGPL-3.0-only로 공개한다. 아래 원저작물의 고지와
라이선스는 유지하며, 원문은 `licenses/NAFNet-BasicSR.txt`에 포함한다.

## TRT-R2R-Light design attribution

`models/trt_r2r_light.py` is an independent implementation of the local
`TRT_R2R_LIGHT_IMPLEMENTATION.md` design, released as project code under
AGPL-3.0-only. Its task-conditioned prototype retrieval concept is inspired by
[Retrieve-to-Restore / R2R](https://github.com/cscxwang/R2R). The design document
cites upstream revision `bf387d56095aaf4edc0b685f8ea58cce5c64c2fc` as its idea
reference. No R2R source files, pretrained weights, teacher, or memory-bank
artifacts are copied or distributed by this implementation. R2R's reported
restoration results do not describe this new architecture.

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

Production files were reimplemented from the pinned references. The unmodified
development oracle snapshot is bundled under `tests/reference/NAFNet/<commit>/`;
it is never imported by production code. Tests compare outputs and gradients
against it. The original license is included alongside the reference files.
The SSIM/PSNR range heuristic (`pred.max() <= 1`) is preserved for exact reference
compatibility, including the very dark uint8 image edge case.

## Official checkpoint

- Asset: NAFNet GoPro width32, linked by the pinned upstream README.
- Source: https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view
- SHA-256: `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`
- Bundled conversion: `weights/nafnet-gopro-width32.pt` (68,683,541 bytes).
- Converted SHA-256: `f6abb1b16853670e8a03bc0842e4d948f0276a1110999c5886fa87f73cb760cb`.
- Conversion: official `params` repackaged as `state_dict`, with `config.model`,
  initialized `tlc_kernels` and original provenance. All 664 tensors retain their
  original FP32 dtype and values. No retraining, precision reduction or
  quantization was performed.

The checkpoint is an external asset; the project's own AGPL metadata does not
relicense it. The [pinned upstream LICENSE](https://github.com/megvii-research/NAFNet/blob/2b4af71ebe098a92a75910c233a3965a3e93ede4/LICENSE)
contains the MIT/Apache notices above. No separate checkpoint license or
restriction was identified in the [pinned upstream README](https://github.com/megvii-research/NAFNet/blob/2b4af71ebe098a92a75910c233a3965a3e93ede4/readme.md)
or [pretrained-model README](https://github.com/megvii-research/NAFNet/blob/2b4af71ebe098a92a75910c233a3965a3e93ede4/experiments/pretrained_models/README.md);
this is not an explicit statement of a separate
weights license. The converted weights are bundled in the repository, outside
the Python module, so a repository clone needs no upstream checkpoint download.

## Dependencies

AlbumentationsX is selected under its AGPL-3.0-only license and imported using
`albumentations`. Other installed dependencies retain their respective licenses;
the resolved versions and origins are recorded in `uv.lock`.
