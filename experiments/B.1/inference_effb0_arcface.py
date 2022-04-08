import torch
import pandas as pd
import joblib
import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.utils.seed_everything import SEED_EVERYTHING
from modules.models.InferenceModel import InferenceModelKNN_Normalizer
from modules.processing.preprocess import exp_a_basic_preprocessing
from modules.models.experemental.EffBo_Arc_exp_A import EffB0_Arc
from modules.dataset.DatasetHappywhile import DatasetHappywhileInference


def map_per_image(label, predictions):
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


if __name__ == "__main__":
    SEED = 42
    FOLD_NUM = 0
    MODEL_NAME = 'tf_efficientnet_b0'
    MSIZE = 224
    BATCH_SIZE = 64
    THR = 0.45
    SEED_EVERYTHING(SEED)

    PROJ_PATH = '/home/kirill/projects/happywhale/'
    ohe_path = PROJ_PATH + 'data/process/ohe.joblib'
    skf_path = PROJ_PATH + 'data/process/skf5_id_fold_mapping.joblib'
    model_save_path = PROJ_PATH + 'data/model/B.1/epoch=21-train_cross_entropy_loss=0.59-train_map5=0.95-val_map5=0.15.ckpt'

    DATA_PATH = '/media/kirill/Windows 10/kaggle/'
    train_csv_path = DATA_PATH + 'segmented/seg_train.csv'
    train_image_path = DATA_PATH + 'segmented/seg_img/'
    test_image_path = DATA_PATH + 'segmented/seg_img_test/'
    submit_save_path = PROJ_PATH + 'data/process/exp/B.1/submit.csv'

    # train and val knn data
    skf_folds = joblib.load(skf_path)
    train_idx, val_idx = skf_folds[FOLD_NUM]['train_idx'], skf_folds[FOLD_NUM]['test_idx']

    # TEST
    #train_idx, val_idx = train_idx[:100], val_idx[:100]

    train_df = pd.read_csv(train_csv_path)
    train_df, val_df = train_df.iloc[train_idx], train_df.iloc[val_idx]
    # get img paths and targets
    train_imgs_paths, val_imgs_paths = [train_image_path + i for i in train_df['image']], [train_image_path + i for i in val_df['image']]
    test_imgs_paths = [test_image_path + i for i in os.listdir(test_image_path)]
    train_targets, val_targets = train_df['individual_id'].tolist(), val_df['individual_id'].tolist()

    # get datasets
    _, val_prep = exp_a_basic_preprocessing(msize=MSIZE)
    train_dataset = DatasetHappywhileInference(
        image_paths=train_imgs_paths,
        targets=train_targets,
        train=True,
        preprocessing=val_prep
    )
    train_dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    val_dataset = DatasetHappywhileInference(
        image_paths=val_imgs_paths,
        targets=val_targets,
        train=False,
        preprocessing=val_prep
    )
    val_dataset = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    test_dataset = DatasetHappywhileInference(
        image_paths=test_imgs_paths,
        train=False,
        preprocessing=val_prep
    )
    test_dataset = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

    # train
    model = EffB0_Arc(eff_name=MODEL_NAME)
    inf_model = InferenceModelKNN_Normalizer(model=model,
                                  weights_path=model_save_path,
                                  ohe_path=ohe_path,
                                  img_preprocessor=val_prep,
                                  device=torch.device('cuda:0'))
    # get embeds
    train_embeds = inf_model.get_embeddings(train_dataset)
    val_embeds = inf_model.get_embeddings(val_dataset)

    inf_model.train(train_embeds, train_targets)

    test_embeds = inf_model.get_embeddings(test_dataset)
    preds = inf_model.predict(test_embeds, thr=THR)
    imgs = os.listdir(test_image_path)
    sub_preds = [' '.join(i) for i in preds]
    sub_df = pd.DataFrame({
        'image': imgs,
        'predictions': sub_preds
    })
    sub_df.to_csv(submit_save_path, index=False)


    # compute MaP5 for different quantiles:
    val_targets_filtered = []
    for v in val_targets:
        if v not in train_targets:
            v = 'new_individual'

        val_targets_filtered.append(v)

    thrs = [0.1, 0.2, 0.3, 0.4, 0.55, 0.6, 0.63, 0.65, 0.67, 0.69, 0.7, 0.8]

    # MIN MODE
    print("\n\n\nMode: min")
    for thr in thrs:
        #preds = inf_model.predict(val_embeds, print_quantile=True, thr=thr)
        preds = inf_model.predict(val_embeds, thr=thr, print_quantile=True, mode='min')

        map5 = []
        for val_target, pred in zip(val_targets_filtered, preds):
            map5.append(map_per_image(val_target, pred))

        print(f'Mean MaP5 for thr {thr}: ', sum(map5) / len(map5))











