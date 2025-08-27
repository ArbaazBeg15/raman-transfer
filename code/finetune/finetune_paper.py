import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from model import *
from utils import *
from train_utils import *
from dataset import get_ds


SEED = 1000
print(SEED)
config = {
    'initial_cnn_channels': 32,
    'cnn_channel_factor': 1.279574024454846,
    'num_cnn_layers': 8,
    'kernel_size': 3,
    'stride': 2,
    'activation_function': 'ELU',
    'fc_dropout': 0.10361700399831791,
    'lr': 0.001,
    'gamma': 0.9649606352621118,
    'baseline_factor_bound': 0.748262317340447,
    'baseline_period_lower_bound': 0.9703081695287203,
    'baseline_period_span': 19.79744237606427,
    'original_datapoint_weight': 0.4335003268130408,
    'augment_slope_std': 0.08171025264382692,
    'batch_size': 32,
    'fc_dims': 226,
    'rolling_bound': 2,
    'num_blocks': 2,
}



def load_transfer_data(path, files):
    csv_path = os.path.join(path, files[8])
    df = pd.read_csv(csv_path)

    input_cols = df.columns[1:2049]
    target_cols = df.columns[2050:]

    targets  = df[target_cols].dropna().to_numpy()

    df = df[input_cols]
    df['Unnamed: 1'] = df['Unnamed: 1'].str.replace("[\[\]]", "", regex=True).astype('int64')
    df['Unnamed: 2048'] = df['Unnamed: 2048'].str.replace("[\[\]]", "", regex=True).astype('int64')

    inputs = df.to_numpy().reshape(-1, 2, 2048)
    inputs = inputs.mean(axis=1)
    
    return inputs, targets


def preprocess_transfer_data(path, files):
    inputs, targets = load_transfer_data(path, files)
    
    spectra_selection = np.logical_and(
        300 <= np.array([float(one) for one in range(2048)]),
        np.array([float(one) for one in range(2048)]) <= 1942,
    )
    
    inputs = inputs[:, spectra_selection]
    
    wns = np.array([
        float(one) for one in range(2048)
    ])[spectra_selection]
    wavenumbers = np.arange(300, 1943)
    
    interpolated_data = np.array(
        [np.interp(wavenumbers, xp=wns, fp=i) for i in inputs]
    )
    
    normed_spectra = interpolated_data / np.max(interpolated_data)
    return normed_spectra, targets



def main():
    setup_reproducibility(SEED)

    hf_token = "xhf_XURkoNhwOIPtEdHfNeRpVkjEwKSkhtigFi"
    path = hf_ds_download(hf_token, "ArbaazBeg/kaggle-spectogram")
    print(path)
    files = os.listdir(path)
 
    inputs, targets = preprocess_transfer_data(path, files)

    epochs = 100
    weight_decay = 1e-3
    lr = 1e-4
    
    drop_out = 0.5
    drop_path_rate = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resume = False
    
    config["dtype"] = torch.float32
    config["spectra_size"] = 1643
    config["spectra_channels"] = 1
    config["fc_dims"] = [
        config["fc_dims"],
        int(config["fc_dims"] / 2),
        3,
    ]

    inputs_mean_std = []
    targets_mean_std = []
    scores = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = kfold.split(inputs)
    
    for fold, (train_idx, eval_idx) in enumerate(splits):
        model_name = f"resnet.finetune.fold.{fold}.{SEED}"
        checkpoint_name = f"paper.finetune.fold.{fold}.{SEED}.pt"
        
        train_inputs = inputs[train_idx]
        train_targets = targets[train_idx]
        eval_inputs = inputs[eval_idx]
        eval_targets = targets[eval_idx]
    
        train_ds = get_ds(train_inputs, train_targets, config)
        
        inputs_mean_std.append((fold, train_ds.s_mean, train_ds.s_std))
        targets_mean_std.append((fold, train_ds.concentration_means, train_ds.concentration_stds))
        continue
        eval_ds = get_ds(eval_inputs, eval_targets, config, (train_ds.s_mean, train_ds.s_std), (train_ds.concentration_means, train_ds.concentration_stds))
        
        batch_size = 32
        train_dl, eval_dl = return_dls(SEED, train_ds, eval_ds, batch_size, len(eval_ds))
        
        #model = ResNet(input_channels=1, dropout=DROPOUT).to(device)
        model = ReZeroNet(**config).to(device)
        if fold == 0: print(get_model_size(model))
        
        ckpt = get_ckpt("/paper.pretrain.avg.pt")
        model.load_state_dict(ckpt)#["state_dict"])
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=True)
        scaler = torch.amp.GradScaler(device)
        scheduler = get_scheduler(optimizer, train_dl, epochs)
        #loss_fn = MSEIgnoreNans()
        
        if True:
            neptune = setup_neptune(
                seed=SEED,
                model_name=model_name, 
                lr=lr, 
                weight_decay=weight_decay, 
                epochs=epochs,
                batch_size=batch_size,
                dropout=None,
                drop_path_rate=None,
                resume=False
            )
        else:
            neptune = None
                
        score = train(
            model, 
            optimizer, 
            device,
            scaler,
            scheduler,
            train_dl, 
            eval_dl,
            loss_fn,
            epochs,
            checkpoint_name,
            neptune_run=neptune,
        )
        
        scores.append(score)
        
    [print(inputs_mean_std[i]) for i in range(5)]   
    #[print(targets_mean_std[i]) for i in range(5)]   
    for i in range(5):
        print(i)
        print()
        #for j in range(3):
        print("mean", inputs_mean_std[i][1].item())
        print("std", inputs_mean_std[i][2].item())

main()