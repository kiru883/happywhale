import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import pytorch_lightning as ptl

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.utils.seed_everything import SEED_EVERYTHING
from modules.dataset.DatasetHappywhile import DatasetHappywhile
from modules.ptl.PtlWrapper import PtlWrapper
from modules.processing.preprocess import exp_a_preprocessing, exp_a_basic_preprocessing
from modules.models.experemental.EffBo_Arc_exp_A import EffB0_Arc


if __name__ == "__main__":
    SEED = 42
    PROJ_PATH = '/home/kkirill/happywhale/'
    ohe_path = PROJ_PATH + 'data/process/ohe.joblib'
    skf_path = PROJ_PATH + 'data/process/skf5_id_fold_mapping.joblib'
    model_save_path = PROJ_PATH + 'data/model/A.3'
    #last_best_model_path = model_save_path + 'epoch=1-train_cross_entropy_loss=21.66-train_map5=0.00-val_map5=0.00.ckpt'

    DATA_PATH = '/home/vadbeg/Data_SSD/Kaggle/happywhale/'
    train_csv_path = DATA_PATH + 'train.csv'
    train_image_path = DATA_PATH + 'train_images/'
    # train_csv_path = PROJ_PATH + 'data/raw/happy-whale-and-dolphin/train.csv'
    # train_image_path = PROJ_PATH + 'data/raw/happy-whale-and-dolphin/train_images/'

    FOLD_NUM = 0
    BATCH_SIZE = 32
    EPOCHS = 30
    GPUS = 1
    LR = 1e-3
    SIZE = 224

    SEED_EVERYTHING(SEED)

    # model, scheduler
    # step_lr_sch_settings = {
    #     'scheduler': torch.optim.lr_scheduler.StepLR,
    #     'step_size': 10,
    #     'gamma': 0.3
    # }
    # load last best model
    # state_dict = torch.load(last_best_model_path)['state_dict']
    # state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}
    model = EffB0_Arc_
    # model.load_state_dict(state_dict)

    model = PtlWrapper(model=model,
                       lr=LR
                       )
    model.train()

    # open folds
    skf_folds = joblib.load(skf_path)
    train_idx, val_idx = skf_folds[FOLD_NUM]['train_idx'], skf_folds[FOLD_NUM]['test_idx']

    # datasets, preprocessor
    train_prep, val_prep = exp_a_basic_preprocessing(msize=SIZE)
    train_dataset = DatasetHappywhile(image_path=train_image_path,
                                      train_csv_path=train_csv_path,
                                      specific_indx=train_idx,
                                      ohe_path=ohe_path,
                                      preprocessing=train_prep)
    train_dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_dataset = DatasetHappywhile(image_path=train_image_path,
                                    train_csv_path=train_csv_path,
                                    specific_indx=val_idx,
                                    ohe_path=ohe_path,
                                    preprocessing=val_prep)
    val_dataset = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # checkpoint callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=model_save_path,
                                          monitor='val_top5',
                                          save_top_k=3,
                                          filename='{epoch}-{train_cross_entropy_loss:.2f}-{train_map5:.2f}-{val_map5:.2f}',
                                          mode='max')

    # train
    trainer = ptl.Trainer(max_epochs=EPOCHS,
                          gpus=GPUS,
                          callbacks=[checkpoint_callback],

                          auto_lr_find=True)
    trainer.fit(model, train_dataset, val_dataset)











