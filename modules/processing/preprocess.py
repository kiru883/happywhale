import albumentations as A

from albumentations.pytorch import ToTensorV2


def exp_a_preprocessing(msize=256):
    normalize = lambda x: x / 255.0

    transform_train = A.Compose([
        # vadim aug
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.1),
        A.ShiftScaleRotate(
            shift_limit=(-0.0625, 0.0625),
            scale_limit=(-0.1, 0.1),
            rotate_limit=(-20, 20),
            p=0.3,
        ),
        A.Blur(p=0.15),

        A.LongestMaxSize(max_size=msize),
        A.PadIfNeeded(min_height=msize, min_width=msize, border_mode=0),    # 0 - constant replace(0)

        ToTensorV2(p=1.0),
        A.Lambda(image=normalize, always_apply=True)

    ])
    transform_val = A.Compose([
        A.LongestMaxSize(max_size=msize),
        A.PadIfNeeded(min_height=msize, min_width=msize, border_mode=0),

        ToTensorV2(p=1.0),
        A.Lambda(image=normalize, always_apply=True)
    ])

    return transform_train, transform_val


def exp_a1_preprocessing(msize=224):
    transform_train = A.Compose([
            A.Resize(msize, msize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)

    transform_val = A.Compose([
            A.Resize(msize, msize),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)

    return transform_train, transform_val


def exp_a_basic_preprocessing(msize=224):
    transform_train = A.Compose([
            A.Resize(msize, msize),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)

    transform_val = A.Compose([
            A.Resize(msize, msize),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)

    return transform_train, transform_val


