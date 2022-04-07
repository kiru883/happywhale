import pandas as pd
import joblib


if __name__ == "__main__":
    FOLD_NUM = 0

    PROJ_PATH = '/home/kkirill/happywhale/'
    DATA_PATH = '/home/vadbeg/Data_SSD/Kaggle/happywhale/'
    skf_path = PROJ_PATH + 'data/process/skf5_id_fold_mapping.joblib'
    train_csv_path = DATA_PATH + 'train.csv'

    skf_folds = joblib.load(skf_path)
    train_idx, val_idx = skf_folds[FOLD_NUM]['train_idx'], skf_folds[FOLD_NUM]['test_idx']
    train_df = pd.read_csv(train_csv_path)

    train_df, val_df = train_df.iloc[train_idx], train_df.iloc[val_idx]

    unique_id_train = train_df['individual_id'].unique()
    unique_id_val = val_df['individual_id'].unique()

    print("Unique lenght of train_df fold 0: ", len(unique_id_train))
    print("Unique lenght of val_df fold 0: ", len(unique_id_val))
    print("Intersection lenght of both datasets: ", len(set(unique_id_train).intersection(set(unique_id_val))))
    print("Number of val ids who are not in train df: ", len(set(unique_id_val).difference(set(unique_id_train))))