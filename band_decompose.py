import os
import numpy as np
import mne
from scipy.signal import hilbert
from tqdm import tqdm

# ========== CONFIGURATION ==========
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
CLEAN_EPOCHS_DIR = os.path.join(DATASET_PATH, "processed_data_epochs")
OUTPUT_BASE = os.path.join(DATASET_PATH, "band_decomposition")

# Frequency bands (name: (low, high))
FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 22),
    "gamma": (22, 30)
}

# Create output folders for each band
for band_name in FREQUENCY_BANDS.keys():
    os.makedirs(os.path.join(OUTPUT_BASE, band_name), exist_ok=True)

# ========== FIND ALL CLEAN EPOCH FILES ==========
epoch_files = []
for root, dirs, files in os.walk(CLEAN_EPOCHS_DIR):
    for file in files:
        if file.endswith("_epochs.npy"):
            epoch_files.append(os.path.join(root, file))

print(f"Found {len(epoch_files)} clean epoch files\n")


# ========== HELPER FUNCTION ==========
def bandpass_filter_epochs(epochs_data, sfreq, l_freq, h_freq):
    """
    Apply zero‑phase bandpass filter to epochs data.
    epochs_data : numpy array (n_epochs, n_channels, n_times)
    sfreq : sampling frequency (Hz)
    l_freq, h_freq : band edges (Hz)
    Returns filtered data (same shape).
    """
    # Create a temporary MNE RawArray to use its robust filtering
    n_epochs, n_channels, n_times = epochs_data.shape
    # We'll filter each epoch individually to avoid edge artefacts
    filtered = np.zeros_like(epochs_data, dtype=np.float32)

    for ep_idx in range(n_epochs):
        # Create a single-epoch RawArray
        info = mne.create_info(ch_names=[f"ch_{i}" for i in range(n_channels)], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(epochs_data[ep_idx], info, verbose=False)
        raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
        filtered[ep_idx] = raw.get_data()

    return filtered


def compute_envelope(data):
    """Compute analytic amplitude (envelope) using Hilbert transform."""
    return np.abs(hilbert(data, axis=-1))


# ========== MAIN PROCESSING ==========
# We need the sampling frequency – it's the same for all files, but we must read one to get it.
# Since the clean epochs are already downsampled? We'll infer sfreq from the original EDF later.
# For safety, ask user or estimate from the data shape? Actually, the epoch length and number of time points give sfreq.
# Let's assume 128 Hz (common in Mumtaz dataset). If not, we can calculate from the original .edf.
# But easier: we can read one raw EDF file to get sfreq. We'll do that.

# Find any original .edf file to get sfreq (or you can hardcode if you know)
edf_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(".edf"):
            edf_files.append(os.path.join(root, file))
            break
    if edf_files:
        break

if not edf_files:
    raise RuntimeError("No .edf file found to determine sampling frequency. Please set SFREQ manually.")

raw_demo = mne.io.read_raw_edf(edf_files[0], preload=False, verbose=False)
SFREQ = raw_demo.info['sfreq']
print(f"Sampling frequency: {SFREQ} Hz\n")

# Process each clean epoch file
for epoch_file in tqdm(epoch_files, desc="Band decomposition"):
    # Load clean epochs
    epochs_data = np.load(epoch_file).astype(np.float64)  # keep float64 for filtering precision
    print(f"\nProcessing {os.path.basename(epoch_file)}: shape {epochs_data.shape}")

    base_name = os.path.splitext(os.path.basename(epoch_file))[0].replace("_epochs", "")

    # For each frequency band
    for band_name, (l_freq, h_freq) in FREQUENCY_BANDS.items():
        # Bandpass filter
        band_data = bandpass_filter_epochs(epochs_data, SFREQ, l_freq, h_freq)

        # Optional: compute envelope (uncomment if you want the amplitude envelope instead)
        # band_data = compute_envelope(band_data)

        # Convert to float16 to save space
        band_data = band_data.astype(np.float16)

        # Save
        out_file = os.path.join(OUTPUT_BASE, band_name, f"{base_name}_{band_name}.npy")
        np.save(out_file, band_data)

    print(f"   Saved 5 band files for {base_name}")

print(f"\n All band decomposition completed. Results saved in:\n {OUTPUT_BASE}")