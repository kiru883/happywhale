import torch
import pandas as pd
import joblib
import os

from modules.utils.seed_everything import SEED_EVERYTHING


if __name__ == "__main__":
    SEED = 42
    SEED_EVERYTHING(SEED)

    DATA_PATH = '/media/kirill/Windows 10/kaggle/'
    test_seg_image_path = DATA_PATH + 'segmented/seg_img_test/'
    test_orig_image_path = DATA_PATH + 'happy-whale-and-dolphin/test_images/'

    un_seg = set(os.listdir(test_seg_image_path))
    un_orig = set(os.listdir(test_orig_image_path))

    print(f"Lenght seg: {len(un_seg)}")
    print(f"Lenght orig: {len(un_orig)}")
