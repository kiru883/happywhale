import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder
from modules.utils.seed_everything import SEED_EVERYTHING


if __name__ == '__main__':
    SEED_EVERYTHING(42)

    PROJ_PATH = '/home/kirill/projects/personal_projects/happywhale/'
    ohe_save_path = PROJ_PATH + 'data/process/ohe.joblib'
    train_csv_path = PROJ_PATH + 'data/raw/happy-whale-and-dolphin/train.csv'

    train_data = pd.read_csv(train_csv_path)
    individual_id = train_data['individual_id'].unique()
    individual_id = individual_id.reshape(-1, 1)

    ohe = OneHotEncoder(sparse=False)
    ohe.fit(individual_id)
    joblib.dump(ohe, ohe_save_path)