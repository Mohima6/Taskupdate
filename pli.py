import os
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm

DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
BAND_DECOMP_DIR = os.path.join(DATASET_PATH, "band_decomposition")
OUTPUT_BASE = os.path.join(DATASET_PATH, "connectivity", "PLI")

FREQUENCY_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

for band in FREQUENCY_BANDS:
    os.makedirs(os.path.join(OUTPUT_BASE, band), exist_ok=True)


def compute_pli(signal1, signal2):
    """
    Phase Lag Index (PLI)
    """

    phase1 = np.angle(hilbert(signal1))
    phase2 = np.angle(hilbert(signal2))

    phase_diff = phase1 - phase2

    imag_cross = np.sin(phase_diff)

    pli = np.abs(np.mean(np.sign(imag_cross)))

    return pli


def compute_connectivity_matrix(epoch_data):

    n_channels, _ = epoch_data.shape

    mat = np.eye(n_channels, dtype=np.float32)

    for i in range(n_channels):
        for j in range(i + 1, n_channels):

            val = compute_pli(epoch_data[i], epoch_data[j])

            mat[i, j] = val
            mat[j, i] = val

    return mat


print("Starting PLI computation...")

for band in FREQUENCY_BANDS:

    band_input_dir = os.path.join(BAND_DECOMP_DIR, band)
    band_output_dir = os.path.join(OUTPUT_BASE, band)

    epoch_files = [
        f for f in os.listdir(band_input_dir)
        if f.endswith(".npy")
    ]

    label_dir = os.path.join(DATASET_PATH, "processed_data_epochs")

    for epoch_file in tqdm(epoch_files, desc=f"PLI {band}"):

        band_data = np.load(
            os.path.join(band_input_dir, epoch_file)
        )

        n_epochs = band_data.shape[0]

        base_name = epoch_file.replace(f"_{band}.npy", "")

        label_file = os.path.join(
            label_dir,
            f"{base_name}_labels.npy"
        )

        if not os.path.exists(label_file):
            continue

        labels = np.load(label_file)

        for ep_idx in range(n_epochs):

            conn_mat = compute_connectivity_matrix(
                band_data[ep_idx]
            )

            np.save(
                os.path.join(
                    band_output_dir,
                    f"{base_name}_epoch{ep_idx:03d}_pli.npy"
                ),
                conn_mat
            )

        np.save(
            os.path.join(
                band_output_dir,
                f"{base_name}_labels.npy"
            ),
            labels
        )

print("PLI computation completed.")
