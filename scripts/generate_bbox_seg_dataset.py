import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import shutil

from tqdm import tqdm
from modules.utils.seed_everything import SEED_EVERYTHING


def extract_bbox(image, gray_thr=1):
    gray_image = np.sum(255 - image, axis=2) / 3
    gray_image = np.where(gray_image > gray_thr, 255, 0)
    gray_image = gray_image.astype(np.uint8)

    ys = np.where(np.any(gray_image, axis=1))[0]
    xs = np.where(np.any(gray_image, axis=0))[0]

    # empty image...
    if len(ys) <= 1 or len(xs) <= 1:
        return image

    y_min, y_max = ys[0], ys[-1]
    x_min, x_max = xs[0], xs[-1]

    #print(xs)
    #plt.imshow(image)
    #plt.show()

    image = image[y_min:y_max, x_min:x_max]
    return image


if __name__ == '__main__':
    #SEED = 42
    #SEED_EVERYTHING(SEED)

    DATA_PATH = '/media/kirill/Windows 10/kaggle/'
    seg_orig_train_images_path = DATA_PATH + 'segmented/seg_img/'
    seg_orig_test_images_path = DATA_PATH + 'segmented/seg_img_test/'
    seg_orig_csv_path = DATA_PATH + 'segmented/seg_train.csv'

    seg_bbox_dataset_path = DATA_PATH + 'seg_bbox_dataset/'
    seg_bbox_train_images_savepath = seg_bbox_dataset_path + 'train_images/'
    seg_bbox_test_images_savepath = seg_bbox_dataset_path + 'test_images/'

    os.mkdir(seg_bbox_dataset_path)
    os.mkdir(seg_bbox_train_images_savepath)
    os.mkdir(seg_bbox_test_images_savepath)
    shutil.copy(seg_orig_csv_path, seg_bbox_dataset_path + 'seg_train.csv')

    # preproc. train images
    train_images = os.listdir(seg_orig_train_images_path)
    for timage in tqdm(train_images):
        img = cv2.imread(seg_orig_train_images_path + timage)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = extract_bbox(img)

        #timage = timage.replace('.png', '.jpg')
        try:
            cv2.imwrite(seg_bbox_train_images_savepath + timage, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except:
            print("wtf: ", img)
            plt.imshow(img)
            plt.show()

    # test images
    test_images = os.listdir(seg_orig_test_images_path)
    for timage in tqdm(test_images):
        img = cv2.imread(seg_orig_test_images_path + timage)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = extract_bbox(img)

        #timage = timage.replace('.png', '.jpg')
        cv2.imwrite(seg_bbox_test_images_savepath + timage, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
