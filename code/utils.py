import random
import torch
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from huggingface_hub import login, snapshot_download
from tqdm.auto import tqdm


def setup_reproducibility(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_float32_matmul_precision("high")


def get_stats(tensor, p=True, r=False):
    mean, std = tensor.mean(), tensor.std()
    min, max =  tensor.min(), tensor.max()
    
    if p: print(f"Min: {min}, Max: {max}, Mean: {mean}, Std: {std}")
    if r: return min, max, mean, std
    
    
def zscore(tensor, mean=None, std=None):
    if mean is None: mean = tensor.mean()
    if std is None: std = tensor.std()
    return (tensor - mean) / (std + 1e-6)


def get_model_size(model):
    print(sum(p.numel() for p in model.parameters()) / 1e6)
    

def get_index(iterable):
    return random.randint(0, len(iterable) - 1)


def split(inputs, targets, seed):
    return train_test_split(
        inputs,
        targets, 
        test_size=0.2,
        shuffle=True, 
        random_state=seed
    ) 


def show_waves(waves, dpi=100):
    """
    waves: numpy array of shape (3, N)
    Creates three separate figures that stretch wide.
    """

    N = waves.shape[1]
    t = np.arange(N)

    # Wide aspect ratio; height modest so each window fills width
    for i in range(waves.shape[0]):
        fig = plt.figure(figsize=(14, 4), dpi=dpi)  # wide figure
        ax = fig.add_subplot(111)
        ax.plot(t, waves[i], linewidth=1)
        ax.set_title(f"Wave {i+1}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        fig.tight_layout()  # reduce margins to use width
    plt.show()
    
    
def hf_ds_download(hf_token, repo_id):
    login(hf_token[1:])
    return snapshot_download(repo_id, repo_type="dataset")


def get_advanced_spectra_features(X):
    """Create multi-channel features from spectra: raw, 1st derivative, 2nd derivative."""
    X_processed = np.zeros_like(X)
    # Baseline correction and SNV
    for i in tqdm(range(X.shape[0])):
        poly = np.polyfit(np.arange(X.shape[1]), X[i], 3)
        baseline = np.polyval(poly, np.arange(X.shape[1]))
        corrected_spec = X[i] - baseline
        X_processed[i] = (corrected_spec - corrected_spec.mean()) / (corrected_spec.std() + 1e-8)

    # Calculate derivatives
    deriv1 = signal.savgol_filter(X_processed, window_length=11, polyorder=3, deriv=1, axis=1)
    deriv2 = signal.savgol_filter(X_processed, window_length=11, polyorder=3, deriv=2, axis=1)

    # Stack as channels
    return np.stack([X_processed, deriv1, deriv2], axis=1)