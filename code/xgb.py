import os
import time
import pandas as pd
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from utils import *


SEED = 1000
HF_TOKEN = "xhf_XURkoNhwOIPtEdHfNeRpVkjEwKSkhtigFi"


def main():
    setup_reproducibility(SEED)
    path = hf_ds_download(HF_TOKEN, repo_id="ArbaazBeg/kaggle-spectogram")
    files = sorted(os.listdir(path))

    csv_path = os.path.join(path, files[11])
    df = pd.read_csv(csv_path)

    input_cols = df.columns[1:2049]
    target_cols = df.columns[2050:]

    targets  = df[target_cols].dropna().to_numpy()

    df = df[input_cols]
    df['Unnamed: 1'] = df['Unnamed: 1'].str.replace("[\[\]]", "", regex=True).astype('int64')
    df['Unnamed: 2048'] = df['Unnamed: 2048'].str.replace("[\[\]]", "", regex=True).astype('int64')

    inputs = df.to_numpy().reshape(-1, 2, 2048)
    inputs = inputs.mean(axis=1)
    inputs = get_advanced_spectra_features(inputs)

    inputs = inputs.reshape(-1, 3 * 2048).astype(np.float32)
    targets = targets.astype(np.float32)

    train_inputs, eval_inputs, train_targets, eval_targets = split(inputs, targets, SEED)

    _, _, mean, std = get_stats(train_inputs, p=False, r=True)
    train_inputs = zscore(train_inputs, mean, std)
    eval_inputs = zscore(eval_inputs, mean, std)

    def objective(trial):
        #n_estimators = trial.suggest_int("n_estimators", 200, 1000)
        #max_depth = trial.suggest_int("max_depth", 3, 20)
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.8, log=True)
        
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=learning_rate,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            #reg_lambda=1.0,
            eval_metric="rmse",  
            #early_stopping_rounds=5,        
            tree_method="hist", 
            device="cuda",
        )

        model.fit(
            train_inputs, train_targets,
            eval_set=[(eval_inputs, eval_targets)],
            verbose=False,
        )

        preds = model.predict(eval_inputs)
        return r2_score(eval_targets, preds)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name="xgb",
        direction="maximize",
        sampler=sampler,
        storage="sqlite:///experiements.db",
        load_if_exists=True
    )
    
    start = time.perf_counter()
    study.optimize(objective, n_trials=16, n_jobs=1)
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f}s")


main()