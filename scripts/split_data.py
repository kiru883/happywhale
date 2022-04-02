import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from modules.utils.seed_everything import SEED_EVERYTHING


if __name__ == '__main__':
    N_SPLITS = 5
    SHUFFLE = True
    SEED = 42

    SEED_EVERYTHING(SEED)

    PROJ_PATH = '/home/kirill/projects/personal_projects/happywhale/'
    skf_save_path = PROJ_PATH + 'data/process/skf5_id_fold_mapping.joblib'
    train_csv_path = PROJ_PATH + 'data/raw/happy-whale-and-dolphin/train.csv'

    train_data = pd.read_csv(train_csv_path)
    X, y = list(range(len(train_data))), train_data['individual_id'].to_numpy()
    X = np.array(X)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=SEED)
    mapping = dict()
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        mapping[i] = {'train_idx': train_idx, 'test_idx': test_idx}

    print(mapping)
    joblib.dump(mapping, skf_save_path)