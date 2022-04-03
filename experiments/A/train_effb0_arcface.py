import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import pytorch_lightning as ptl

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.models.ArcFaceHead import ArcMarginProduct
from modules.models.Effnet import EffNetBx
from modules.utils.seed_everything import SEED_EVERYTHING
from modules.dataset.DatasetHappywhile import DatasetHappywhile
from modules.ptl.PtlWrapper import PtlWrapper
from modules.processing.preprocess import exp_a_preprocessing


class EffB0_Arc(nn.Module):
    def __init__(self, n_classes=15587, s=30.0, m=0.50, easy_margin=False, eff_name='efficientnet-b0'):
        super(EffB0_Arc, self).__init__()

        self.effnet = EffNetBx(model_name=eff_name)
        self.head = ArcMarginProduct(in_features=512, out_features=n_classes, s=s, m=m, easy_margin=easy_margin)
        self.softmax = nn.Softmax()

    def forward(self, input, label):
        embedding = self.effnet(input)
        cos_dist = self.head(embedding, label)
        #output = self.softmax(cos_dist)
        return embedding, cos_dist#output

    def forward_eval(self, input):
        embedding = self.effnet(input)
        return embedding


if __name__ == "__main__":
    SEED = 42
    PROJ_PATH = '/home/kirill/projects/personal_projects/happywhale/'
    ohe_path = PROJ_PATH + 'data/process/ohe.joblib'
    skf_path = PROJ_PATH + 'data/process/skf5_id_fold_mapping.joblib'
    train_csv_path = PROJ_PATH + 'data/raw/happy-whale-and-dolphin/train.csv'
    train_image_path = PROJ_PATH + 'data/raw/happy-whale-and-dolphin/train_images/'
    model_save_path = PROJ_PATH + 'data/model/'
    FOLD_NUM = 0
    BATCH_SIZE = 16 # 32
    EPOCHS = 30
    GPUS = 1
    LR = 1e-3
    SIZE = 224

    SEED_EVERYTHING(SEED)

    # model
    model = PtlWrapper(model=EffB0_Arc(),
                       lr=LR)
    model.train()

    # open folds
    skf_folds = joblib.load(skf_path)
    train_idx, val_idx = skf_folds[FOLD_NUM]['train_idx'], skf_folds[FOLD_NUM]['test_idx']

    # datasets, preprocessor
    train_prep, val_prep = exp_a_preprocessing(msize=SIZE)
    train_dataset = DatasetHappywhile(image_path=train_image_path,
                                      train_csv_path=train_csv_path,
                                      specific_indx=train_idx,
                                      ohe_path=ohe_path,
                                      preprocessing=train_prep)
    train_dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataset = DatasetHappywhile(image_path=train_image_path,
                                    train_csv_path=train_csv_path,
                                    specific_indx=val_idx,
                                    ohe_path=ohe_path,
                                    preprocessing=val_prep)
    val_dataset = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # checkpoint callbacks, schedulers
    #step_lr_sch = torch.
    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path,
                                          monitor='val_map5',
                                          save_top_k=3,
                                          filename='{epoch}-{train_cross_entropy_loss:.2f}-{train_map5:.2f}-{val_map5:.2f}')

    # train
    trainer = ptl.Trainer(max_epochs=EPOCHS,
                          gpus=GPUS,
                          callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataset, val_dataset)











