import os
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pandas as pd
from utils import * 


SEED = 1000
HF_TOKEN = "xhf_XURkoNhwOIPtEdHfNeRpVkjEwKSkhtigFi"

def inference(path, files, model, study):
    csv_path = os.path.join(path, files[1])
    test_df = pd.read_csv(csv_path)
    
    row1 = test_df.columns[1:].to_numpy().copy()
    row1[-1] = "5611"
    row1 = row1.astype(np.float64)

    cols = test_df.columns[1:]
    test_df = test_df[cols]
    test_df[" 5611]"] = test_df[' 5611]'].str.replace('[\[\]]', '', regex=True).astype('int64')
    test = test_df.to_numpy()

    test = np.insert(test, 0, row1, axis=0)
    test = test.reshape(-1, 2, 2048).mean(axis=1)

    print(get_stats(test))
    test = get_advanced_spectra_features(test)
    test = test.reshape(-1, 3 * 2048)
    test = zscore(test, mean, std)
    print(test.shape, test.dtype, get_stats(test))
    
    preds = model.predict(test)
    print("preds shape", preds.shape)
    
    column_names = ['Glucose', 'Sodium Acetate', 'Magnesium Sulfate']
    preds_df = pd.DataFrame(preds, columns=column_names)
    preds_df.insert(0, 'ID', [i+1 for i in range(len(preds_df))])
    print(preds_df)
    
    name = f"/xgboost.optuna.{study.best_value:.4f}.csv"
    preds_df.to_csv(name, index=False)
    f = pd.read_csv(f"{name}")
    print(f)


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
    
    study = optuna.load_study(
        study_name="xgb",
        storage="sqlite:///experiements.db"
    )
    
    model = XGBRegressor(
        n_estimators=study.best_params["n_estimators"],
        learning_rate=study.best_params["learning_rate"],
        max_depth=study.best_params["max_depth"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        #reg_lambda=1.0,
        eval_metric="rmse",  
        early_stopping_rounds=study.best_params["early_stopping_rounds"],        
        tree_method="hist", 
        device="cuda",
    )

    model.fit(
        train_inputs, train_targets,
        eval_set=[(eval_inputs, eval_targets)],
        verbose=False,
    )

    preds = model.predict(eval_inputs)
    print("Eval score", r2_score(eval_targets, preds))
    inference(path, files, model, study)
    
main()
