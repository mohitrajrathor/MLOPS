# data preparation logic

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse



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


    np.save(save_dir + "/prepared_data.npy", X)
    np.save(save_dir + "/prepared_target.npy", y)

    print("data prepared successfully and saved at following locations:")
    print(save_dir + "/prepared_data.npy")
    print(save_dir + "/prepared_target.npy")



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