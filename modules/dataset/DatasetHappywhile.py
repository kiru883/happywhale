import torch
import pandas as pd
import joblib
import sklearn
import numpy as np
import cv2
import matplotlib.pyplot as plt


class DatasetHappywhile(torch.utils.data.Dataset):
    def __init__(self, image_path, ohe_path, train_csv_path=None, specific_indx=None, preprocessing=None):#!!!!!
        self.image_path = image_path

        self.ohe = joblib.load(ohe_path)
        self.train_csv = pd.read_csv(train_csv_path)

        if specific_indx is not None:
            self.train_csv = self.train_csv.iloc[specific_indx]

        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, item):
        item = self.train_csv.iloc[item]
        iid = item['individual_id']
        image_name = item['image']

        label = self.ohe.transform([[iid]])
        label = label.flatten()

        input = cv2.imread(self.image_path + image_name) #!!!
        input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
        input = self.preprocessing(image=input)['image']
        input = input.type(torch.float32)

        return {'input': input, 'label': label}


class DatasetHappywhileInference(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets=None, train=False, preprocessing=None):#!!!!!
        self.image_paths = image_paths
        self.targets = targets
        self.train = train
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img_path = self.image_paths[item]
        img_tensor = cv2.imread(img_path)
        img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_RGB2BGR)
        img_tensor = self.preprocessing(image=img_tensor)['image']
        img_tensor = img_tensor.type(torch.float32)

        target = None
        if self.train:
            return {'input': img_tensor, 'label': self.targets[item]}
        else:
            return {'input': img_tensor}





