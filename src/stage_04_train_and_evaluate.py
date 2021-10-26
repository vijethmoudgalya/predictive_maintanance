# load the train and test
# train algo
# save the metrices, params
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.scaling import scale
from sklearn.ensemble import RandomForestRegressor
from stage_01_get_data import read_params
import argparse
import joblib
import json


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["RandomForestRegressor"]["params"]["n_estimators"]
    max_features = config["estimators"]["RandomForestRegressor"]["params"]["max_features"]
    min_samples_split = config["estimators"]["RandomForestRegressor"]["params"]["min_samples_split"]
    bootstrap = config["estimators"]["RandomForestRegressor"]["params"]["bootstrap"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    x_train,x_test = scale(train_x,test_x)

    rfr = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_features=max_features,
        min_samples_split = min_samples_split,
        bootstrap=bootstrap,
        random_state=random_state)
    rfr.fit(x_train, train_y)

    predicted_qualities = rfr.predict(x_test)
    
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Random Forest Regressor model (n_estimators=%f, max_features=%s,min_samples_split=%f):" % (n_estimators, max_features,min_samples_split))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

#####################################################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "min_samples_split": min_samples_split
        }
        json.dump(params, f, indent=4)
#####################################################


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(rfr, model_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)