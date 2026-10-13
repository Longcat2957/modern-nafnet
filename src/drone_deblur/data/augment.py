"""AlbumentationsX supplies the shared random decisions for each image pair."""

import albumentations as A


def build_train_transform(crop_size: int, seed: int) -> A.Compose:
    if not isinstance(crop_size, int) or isinstance(crop_size, bool) or crop_size < 1:
        raise ValueError("crop_size must be a positive integer")
    return A.Compose(
        [A.RandomCrop(height=crop_size, width=crop_size), A.D4(p=1)],
        additional_targets={"sharp": "image"},
        seed=seed,
        strict=True,
    )
