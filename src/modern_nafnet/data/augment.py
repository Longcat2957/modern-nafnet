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


def build_blur_transform(blur_range: list[int], seed: int) -> A.Compose:
    """Apply degradation to the cropped input only, never to the sharp target."""
    if (
        not isinstance(blur_range, (list, tuple))
        or len(blur_range) != 2
        or any(type(n) is not int or n < 3 or n % 2 == 0 for n in blur_range)
        or blur_range[0] > blur_range[1]
    ):
        raise ValueError("blur_range must be two ordered odd integers >= 3")
    return A.Compose(
        [A.MotionBlur(blur_range=tuple(blur_range), allow_shifted=False, p=1)],
        seed=seed,
        strict=True,
    )
