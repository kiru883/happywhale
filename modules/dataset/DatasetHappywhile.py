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
        input = torch.tensor(input, dtype=torch.float32)
        input /= 255.0 #torch.max(input)
        input = torch.moveaxis(input, 2, 0)
        #print(input.dtype)
        #print(input.shape)
        #print(input)

        return {'input': input, 'label': label}






