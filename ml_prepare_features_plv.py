import os
import numpy as np
from itertools import combinations
from collections import Counter
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
CONN_METRIC = "PLV"   
CONN_BASE = os.path.join(DATASET_PATH, "connectivity", CONN_METRIC)
FREQUENCY_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
OUTPUT_FEATURES = os.path.join(DATASET_PATH, "machine_learning", "features")
os.makedirs(OUTPUT_FEATURES, exist_ok=True)
def load_and_average_subject(band_dir, subject_base):
    """Average all epoch matrices for one subject."""
    epoch_files = [f for f in os.listdir(band_dir)
                   if f.startswith(subject_base) and f.endswith(f"_{CONN_METRIC.lower()}.npy")]
    if not epoch_files:
        return None
    matrices = []
    for ef in epoch_files:
        mat = np.load(os.path.join(band_dir, ef))
        matrices.append(mat)
    return np.mean(matrices, axis=0)
for band in FREQUENCY_BANDS:
    print(f"\nProcessing {band}...")
    band_dir = os.path.join(CONN_BASE, band)
    if not os.path.exists(band_dir):
        print(f"  Folder {band_dir} not found, skip")
        continue
    label_files = [f for f in os.listdir(band_dir) if f.endswith("_labels.npy")]
    if not label_files:
        print(f"  No label files, skip")
        continue
    subject_list = []
    labels_list = []
    for lf in label_files:
        base = lf.replace("_labels.npy", "")
        labels = np.load(os.path.join(band_dir, lf))
        if len(labels) > 0:
            subject_list.append(base)
            labels_list.append(labels[0])
    subject_matrices = []
    valid_labels = []
    for base, lab in zip(subject_list, labels_list):
        avg_mat = load_and_average_subject(band_dir, base)
        if avg_mat is not None:
            subject_matrices.append(avg_mat)
            valid_labels.append(lab)
    if len(subject_matrices) == 0:
        print(f"  No valid subjects, skip")
        continue
    shapes = [m.shape[0] for m in subject_matrices]
    if len(set(shapes)) > 1:
        common_n = Counter(shapes).most_common(1)[0][0]
        filtered = [(m, l) for m, l in zip(subject_matrices, valid_labels) if m.shape[0] == common_n]
        subject_matrices, valid_labels = zip(*filtered)
        subject_matrices = list(subject_matrices)
        valid_labels = list(valid_labels)
        print(f"  Inconsistent channels: kept {common_n} ch, subjects {len(subject_matrices)}")
    n_channels = subject_matrices[0].shape[0]
    # Flatten upper triangle (excluding diagonal)
    pairs = list(combinations(range(n_channels), 2))
    n_features = len(pairs)
    X = np.zeros((len(subject_matrices), n_features))
    for i, mat in enumerate(subject_matrices):
        for f, (p, q) in enumerate(pairs):
            X[i, f] = mat[p, q]
    y = np.array(valid_labels)
    np.save(os.path.join(OUTPUT_FEATURES, f"{CONN_METRIC}_{band}_X.npy"), X)
    np.save(os.path.join(OUTPUT_FEATURES, f"{CONN_METRIC}_{band}_y.npy"), y)
    print(f"  Saved {X.shape[0]} subjects, {n_features} features")
