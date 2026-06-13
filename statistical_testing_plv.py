import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
CONN_METRIC = "PLV"  
CONN_DIR = os.path.join(DATASET_PATH, "connectivity", CONN_METRIC)
STATS_OUTPUT = os.path.join(DATASET_PATH, "statistics", CONN_METRIC)
FREQUENCY_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
ALPHA = 0.05
for band in FREQUENCY_BANDS:
    band_out = os.path.join(STATS_OUTPUT, band)
    os.makedirs(band_out, exist_ok=True)
    os.makedirs(os.path.join(band_out, "figures"), exist_ok=True)
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std
def rank_biserial(x, y):
    u, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
    n1, n2 = len(x), len(y)
    return 1 - (2 * u) / (n1 * n2)
def load_and_average_subject(band_dir, subject_base):
    """Load all epoch matrices for a subject and average across epochs."""
    epoch_files = [f for f in os.listdir(band_dir)
                   if f.startswith(subject_base) and f.endswith(f"_{CONN_METRIC.lower()}.npy")]
    if not epoch_files:
        return None
    matrices = []
    for ef in epoch_files:
        mat = np.load(os.path.join(band_dir, ef))
        matrices.append(mat)
    return np.mean(matrices, axis=0)
print(f"Starting statistical testing for {CONN_METRIC}...\n")
for band in FREQUENCY_BANDS:
    print(f"\n=== Processing band: {band} ===")
    band_dir = os.path.join(CONN_DIR, band)
    if not os.path.exists(band_dir):
        print(f"  Band folder {band_dir} not found, skipping")
        continue
    label_files = [f for f in os.listdir(band_dir) if f.endswith("_labels.npy")]
    if not label_files:
        print(f"  No label files found in {band_dir}, skipping")
        continue
    subject_base_names = []
    labels_list = []
    for lf in label_files:
        base = lf.replace("_labels.npy", "")
        labels = np.load(os.path.join(band_dir, lf))
        if len(labels) > 0:
            subject_base_names.append(base)
            labels_list.append(labels[0])  
    subject_matrices = []
    valid_labels = []
    valid_bases = []
    for base, lab in zip(subject_base_names, labels_list):
        avg_mat = load_and_average_subject(band_dir, base)
        if avg_mat is not None:
            subject_matrices.append(avg_mat)
            valid_labels.append(lab)
            valid_bases.append(base)
    if len(subject_matrices) == 0:
        print(f"  No valid subjects for {band}")
        continue
    shapes = [mat.shape[0] for mat in subject_matrices]
    unique_shapes = set(shapes)
    if len(unique_shapes) > 1:
        shape_counts = Counter(shapes)
        common_n = shape_counts.most_common(1)[0][0]
        print(f"  Inconsistent channel counts: {dict(shape_counts)}")
        print(f"  Keeping only subjects with {common_n} channels")
        filtered = [(mat, lab, base) for mat, lab, base in zip(subject_matrices, valid_labels, valid_bases)
                    if mat.shape[0] == common_n]
        if not filtered:
            print(f"  No subjects with consistent channel count, skipping {band}")
            continue
        subject_matrices, valid_labels, valid_bases = zip(*filtered)
        subject_matrices = list(subject_matrices)
        valid_labels = list(valid_labels)
        valid_bases = list(valid_bases)
        n_channels = common_n
    else:
        n_channels = shapes[0]
    n_subj = len(subject_matrices)
    healthy_count = valid_labels.count(0)
    mdd_count = valid_labels.count(1)
    print(f"  Subjects after filtering: {n_subj} (Healthy: {healthy_count}, MDD: {mdd_count})")

    if healthy_count < 3 or mdd_count < 3:
        print(f"  Not enough subjects in one group (min 3 required), skipping {band}")
        continue

    healthy_indices = [i for i, lab in enumerate(valid_labels) if lab == 0]
    mdd_indices = [i for i, lab in enumerate(valid_labels) if lab == 1]

    pairs = list(combinations(range(n_channels), 2))
    results = []

    print(f"  Computing statistics for {len(pairs)} channel pairs...")
    for (i, j) in tqdm(pairs, desc="  Channel pairs"):
        values_healthy = [subject_matrices[idx][i, j] for idx in healthy_indices]
        values_mdd = [subject_matrices[idx][i, j] for idx in mdd_indices]


        p_norm_h = stats.shapiro(values_healthy)[1] if len(values_healthy) >= 3 else 0.5
        p_norm_m = stats.shapiro(values_mdd)[1] if len(values_mdd) >= 3 else 0.5
        both_normal = (p_norm_h > 0.05 and p_norm_m > 0.05) and len(values_healthy) >= 3 and len(values_mdd) >= 3

        if both_normal:
            _, p_val = stats.ttest_ind(values_healthy, values_mdd, equal_var=False)
            effect = cohen_d(values_healthy, values_mdd)
            test_used = "t-test"
        else:
            _, p_val = stats.mannwhitneyu(values_healthy, values_mdd, alternative='two-sided')
            effect = rank_biserial(values_healthy, values_mdd)
            test_used = "Mann-Whitney"

        results.append({
            "channel_i": i,
            "channel_j": j,
            "p_raw": p_val,
            "test_used": test_used,
            "effect_size": effect,
            "normality_p_healthy": p_norm_h,
            "normality_p_mdd": p_norm_m
        })

    
    df = pd.DataFrame(results)
    _, p_fdr = fdrcorrection(df["p_raw"].values, alpha=ALPHA)
    df["p_fdr"] = p_fdr
    df["significant_fdr"] = df["p_fdr"] < ALPHA

    
    csv_path = os.path.join(STATS_OUTPUT, band, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    sig_df = df[df["significant_fdr"] == True]
    sig_path = os.path.join(STATS_OUTPUT, band, "significant_connections.csv")
    sig_df.to_csv(sig_path, index=False)
    print(f"  Significant connections: {len(sig_df)} / {len(df)}")
    effect_mat = np.zeros((n_channels, n_channels))
    for _, row in df.iterrows():
        i, j = int(row["channel_i"]), int(row["channel_j"])
        effect_mat[i, j] = row["effect_size"]
        effect_mat[j, i] = row["effect_size"]
    np.save(os.path.join(STATS_OUTPUT, band, "effect_sizes.npy"), effect_mat)

    
    fig_dir = os.path.join(STATS_OUTPUT, band, "figures")
    if len(sig_df) > 0:
        top_sig = sig_df.nsmallest(10, "p_fdr")
        n_plot = min(10, len(top_sig))
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        for idx, (_, row) in enumerate(top_sig.head(n_plot).iterrows()):
            i, j = int(row["channel_i"]), int(row["channel_j"])
            vals_h = [subject_matrices[idx][i, j] for idx in healthy_indices]
            vals_m = [subject_matrices[idx][i, j] for idx in mdd_indices]
            ax = axes[idx]
            bp = ax.boxplot([vals_h, vals_m], labels=["Healthy", "MDD"], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('salmon')
            ax.set_title(f"Ch{i}-Ch{j}\np={row['p_fdr']:.4f}, d={row['effect_size']:.3f}")
            ax.set_ylabel(f"{CONN_METRIC} value")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "boxplots_top10.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved boxplots_top10.png")

    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(effect_mat, dtype=bool), k=1)
    sns.heatmap(effect_mat, mask=mask, cmap='RdBu_r', center=0,
                square=True, cbar_kws={"label": "Effect size"})
    plt.title(f"{CONN_METRIC} - {band} band: Effect sizes\n(Red=higher in MDD, Blue=higher in Healthy)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "effect_size_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved effect_size_heatmap.png")

    
    plt.figure(figsize=(6, 4))
    counts = [len(sig_df), len(df) - len(sig_df)]
    labels = ['Significant', 'Non-significant']
    colors = ['firebrick', 'lightgray']
    plt.bar(labels, counts, color=colors)
    plt.ylabel('Number of connections')
    plt.title(f"{CONN_METRIC} - {band} band\nSignificant connections (FDR < {ALPHA})")
    for i, v in enumerate(counts):
        plt.text(i, v + 2, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "significant_counts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved significant_counts.png")

print(f"\nAll statistical testing for {CONN_METRIC} completed.")
print(f"Results saved in: {STATS_OUTPUT}")
