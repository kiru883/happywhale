import albumentations as A

from albumentations.pytorch import ToTensorV2


def exp_a_preprocessing(msize=256):
    transform_train = A.Compose([


        A.LongestMaxSize(max_size=msize),
        A.PadIfNeeded(min_height=msize, min_width=msize, border_mode=0),    # 0 - constant replace(0)
        #A.Normalize()

    ])
    transform_val = A.Compose([
        A.LongestMaxSize(max_size=msize),
        A.PadIfNeeded(min_height=msize, min_width=msize, border_mode=0),
        #ToTensor()
        #A.Normalize()
    ])

    return transform_train, transform_val

