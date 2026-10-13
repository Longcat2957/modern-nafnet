# Pretrained weights

`nafnet-gopro-width32.pt` is the official NAFNet GoPro width32 FP32 checkpoint,
repackaged for Modern NAFNet. Clone this repository to use it directly; no
upstream download, Git LFS client or separate asset fetch is required.

The checkpoint contains `state_dict`, `config.model`, `tlc_kernels` and
`provenance`. All 664 original tensors retain their FP32 dtype and values;
conversion does not train, quantize or otherwise modify the weights. Training,
evaluation, inference and export already support this checkpoint format.

- Original SHA-256: `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`
- Converted SHA-256: `f6abb1b16853670e8a03bc0842e4d948f0276a1110999c5886fa87f73cb760cb`
- Size: 68,683,541 bytes (65.50 MiB)
- Source and license notices: [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#official-checkpoint)
  and [original LICENSE](../licenses/NAFNet-BasicSR.txt).

Keep this initial checkpoint fixed; generated fine-tuned/PTQ/QAT checkpoints
belong in ignored `artifacts/`. This binary is tracked in regular Git and adds
about 65.5 MiB to the repository checkout.
