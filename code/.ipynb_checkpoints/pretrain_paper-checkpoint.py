from utils import *
from dataset import get_ds


SEED = 1000
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

def main():
    setup_reproducibility(SEED)

    hf_token = "xhf_XURkoNhwOIPtEdHfNeRpVkjEwKSkhtigFi"
    path = hf_ds_download(hf_token, "ArbaazBeg/kaggle-spectogram")
    print(path)
    files = os.listdir(path)

    ds_names = [
        "anton_532.csv",
        "anton_785.csv",
        "kaiser.csv",
        "mettler_toledo.csv",
        "metrohm.csv",
        "tornado.csv",
        "tec5.csv",
        "timegate.csv"
    ]
    inputs, targets = load_datasets(path, ds_names)

    import warnings#; warnings.filterwarnings("ignore")


    EPOCHS = 100
    WD = 1e-3
    LR = 1e-4
    
    DROPOUT = 0.5
    DROP_PATH_RATE = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESUME = False
    NEPTUNE = None
    config["dtype"] = torch.float32
    config["spectra_size"] = 1643
    config["spectra_channels"] = 1
    config["fc_dims"] = [
        config["fc_dims"],
        int(config["fc_dims"] / 2),
        3,
    ]
    
    #mse_loss_function = MSEIgnoreNans()


    from sklearn.model_selection import KFold


    inputs_mean_std = []
    targets_mean_std = []
    scores = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = kfold.split(inputs)
    
    for fold, (train_idx, eval_idx) in enumerate(splits):
        MODEL_NAME = f"resnet.paper.pretrain.fold.{fold}"
        checkpoint_name = f"paper.pretrain.fold.{fold}.pt"
        
        train_inputs = inputs[train_idx]
        train_targets = targets[train_idx]
        eval_inputs = inputs[eval_idx]
        eval_targets = targets[eval_idx]
    
        train_ds = get_ds(train_inputs, train_targets, config)
        
        inputs_mean_std.append((fold, train_ds.s_mean, train_ds.s_std))
        targets_mean_std.append((fold, train_ds.concentration_means, train_ds.concentration_stds))
        
        eval_ds = get_ds(eval_inputs, eval_targets, config, (train_ds.s_mean, train_ds.s_std), (train_ds.concentration_means, train_ds.concentration_stds))
        
        BATCH_SIZE = 32
        train_dl, eval_dl = return_dls(SEED, train_ds, eval_ds, BATCH_SIZE, len(eval_ds))
        
        #model = ResNet(input_channels=1, dropout=DROPOUT).to(device)
        model = ReZeroNet(**config).to(device)
        if fold == 0: print(get_model_size(model))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=True)
        scaler = torch.amp.GradScaler(device)
        scheduler = get_scheduler(optimizer, train_dl, EPOCHS)
        
        score = train(
                model, 
                optimizer, 
                device,
                scaler,
                scheduler,
                train_dl, 
                eval_dl,
                EPOCHS,
                checkpoint_name,
                neptune_run=NEPTUNE,
            )
        
        scores.append(score)


main()