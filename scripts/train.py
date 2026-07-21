# train.py 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle
import json
import argparse


def train(model_name: str = "model_v1"):
    '''training logistic regression model'''

    try: 
        X = np.load("data/processed/prepared_data.npy")
        y = np.load("data/processed/prepared_target.npy")
    except Exception as e:
        print(e)
        print("Error while loading datasets aborting....")
        exit()

    # data splitting
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    print("data splitting...   [done]")

    log_reg = LogisticRegression()
    log_reg.fit(train_X, train_y)

    accuracy = accuracy_score(test_y, log_reg.predict(test_X))


    # save model
    os.makedirs("./artifacts", exist_ok=True)
    with open(f"./artifacts/{model_name}.pkl", "wb") as model_file:
        pickle.dump(log_reg, model_file)

    print(f"model successfully saved at artifacts/{model_name}.pkl")

    # save accuracy
    eval_rp = {
        "accuracy" : accuracy, 
        "model_name": model_name
    }

    with open(f"./artifacts/{model_name}_report.json", "w") as report:
        json.dump(eval_rp, report)

    print(f"Model stats successfully saved at /artifacts/{model_name}_report.json")



def main():
    '''main execution func'''
    parser = argparse.ArgumentParser(description="Model training program")

    parser.add_argument("-n", "--name", default="model_v1", help="model name to diffrentiate with already stored model otherwise it will rewirte the model file.")

    args = parser.parse_args()

    try:
        print("Training started...")
        train(args.name)    
        print("Execution Completed !")
    except Exception as e:
        print(e)
        print("Error occured\nAborting....")
        exit()
    

if __name__ == "__main__":
    main()