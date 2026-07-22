# data preparation logic

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]



def remove_outliers(df):
    '''remove outliers from datasets'''
    df = df.copy()

    Q1 = df.select_dtypes(include=[np.number]).quantile(0.25)
    Q3 = df.select_dtypes(include=[np.number]).quantile(0.75)


    IQR = Q3 - Q1

    upr_bnd = Q3 + IQR * 1.5
    lwr_bnd = Q1 - IQR * 1.5

    outliers = ((df.select_dtypes(include=[np.number]) > upr_bnd) | (df.select_dtypes(include=[np.number]) < lwr_bnd))
    df.drop(index=(outliers[outliers.any(axis=1)].index), inplace=True)

    return df



def prepare(data_path: str, save_dir: str):
    '''Data preparation func'''

    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    save_dir = Path(save_dir)
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Data preparation starting...")
    df = pd.read_csv(data_path)

    # drop null values 
    df.dropna(inplace=True)
    print("Null values [removed]")

    # remove outliers
    df = remove_outliers(df)
    print("outliers [removed]")

    # training pipeline
    num_cols = df.select_dtypes(include=[np.number]).columns.to_list()
    # cat_cols = df.select_dtypes(include=[np.object_]).columns.to_list().remove("species")


    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())
    ])


    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
    ])


    le = LabelEncoder()

    y = le.fit_transform(df['species'])
    X = preprocessor.fit_transform(df.drop(['species'], axis=1))

    print("Scaled and onehot encoded [Done]")


    prepared_data_path = save_dir / "prepared_data.npy"
    prepared_target_path = save_dir / "prepared_target.npy"

    np.save(prepared_data_path, X)
    np.save(prepared_target_path, y)

    print("data prepared successfully and saved at following locations:")
    print(prepared_data_path)
    print(prepared_target_path)



def main():
    '''Main execution func'''

    parser = argparse.ArgumentParser(description="Data preparation program. Run this file in root dir.")
    parser.add_argument("-d", "--datapath", type=str, default="data/raw/iris.csv", help="Source data path")
    parser.add_argument("-s", "--savedir", type=str, default="data/processed/", help="Source data path")

    args = parser.parse_args()

    prepare(args.datapath, args.savedir)

    print("execution completed!")


if __name__ == "__main__":
    main()
