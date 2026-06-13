import os
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm
from joblib import Parallel, delayed  
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
BAND_DECOMP_DIR = os.path.join(DATASET_PATH, "band_decomposition")
OUTPUT_BASE = os.path.join(DATASET_PATH, "connectivity", "PLV")
FREQUENCY_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
for band in FREQUENCY_BANDS:
    os.makedirs(os.path.join(OUTPUT_BASE, band), exist_ok=True)
def plv(signal1, signal2):
    """
    Compute Phase Locking Value between two signals.
    signal1, signal2 : 1D numpy arrays (time series)
    Returns a float between 0 and 1.
    """
    phase1 = np.angle(hilbert(signal1))
    phase2 = np.angle(hilbert(signal2))
    phase_diff = phase1 - phase2
    # PLV = |mean(e^{j * phase_diff})|
    return np.abs(np.mean(np.exp(1j * phase_diff)))
def compute_plv_matrix(epoch_data):
    """
    Compute full PLV matrix for one epoch.
    epoch_data : numpy array (n_channels, n_times)
    Returns : (n_channels, n_channels) PLV matrix (symmetric, diagonal = 1)
    """
    n_channels, n_times = epoch_data.shape
    plv_mat = np.eye(n_channels, dtype=np.float32)  # diagonal = 1 (self-synchrony)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            val = plv(epoch_data[i], epoch_data[j])
            plv_mat[i, j] = val
            plv_mat[j, i] = val
    return plv_mat
def process_band(band_name):
    """
    Process all epoch files for a single frequency band.
    """
    band_input_dir = os.path.join(BAND_DECOMP_DIR, band_name)
    band_output_dir = os.path.join(OUTPUT_BASE, band_name)
    epoch_files = [f for f in os.listdir(band_input_dir) if f.endswith('.npy')]
    if not epoch_files:
        print(f"   No files found for band {band_name}")
        return
    label_dir = os.path.join(DATASET_PATH, "processed_data_epochs")
    for epoch_file in tqdm(epoch_files, desc=f"PLV {band_name}"):
        band_data = np.load(os.path.join(band_input_dir, epoch_file))
        n_epochs, n_channels, n_times = band_data.shape
        base_name = epoch_file.replace(f"_{band_name}.npy", "")
        label_file = os.path.join(label_dir, f"{base_name}_labels.npy")
        if os.path.exists(label_file):
            labels = np.load(label_file)
        else:
            print(f"   Warning: no label file for {base_name}, skipping")
            continue
        for ep_idx in range(n_epochs):
            epoch_signal = band_data[ep_idx]  
            plv_mat = compute_plv_matrix(epoch_signal)
            out_name = f"{base_name}_epoch{ep_idx:03d}_plv.npy"
            out_path = os.path.join(band_output_dir, out_name)
            np.save(out_path, plv_mat)
        label_out_path = os.path.join(band_output_dir, f"{base_name}_labels.npy")
        np.save(label_out_path, labels)
        print(f"   Saved {n_epochs} PLV matrices for {base_name} ({band_name})")
if __name__ == "__main__":
    print("Starting PLV computation...\n")
    for band in FREQUENCY_BANDS:
        process_band(band)
    print(f"\n All PLV matrices saved in:\n {OUTPUT_BASE}")
