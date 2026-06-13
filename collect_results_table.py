import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas import ExcelWriter

# ========== CONFIGURATION ==========
DATASET_PATH = r"C:\Users\mohimaCHAKRABORTY\Taskupdate"
RESULTS_BASE = os.path.join(DATASET_PATH, "machine_learning", "results")
OUTPUT_DIR = os.path.join(DATASET_PATH, "machine_learning", "final_tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = ["PLV", "PLI", "wPLI"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
CLASSIFIERS = ["LogisticRegression", "DecisionTree", "RandomForest", "SVM", "XGBoost"]

# Map classifier names to short labels for display
CLASSIFIER_LABELS = {
    "LogisticRegression": "LR",
    "DecisionTree": "DT",
    "RandomForest": "RF",
    "SVM": "SVM",
    "XGBoost": "XGB"
}

def extract_mean(value_str):
    """Extract the mean from '0.8727 ± 0.0782'."""
    try:
        return float(value_str.split(" ± ")[0])
    except:
        return np.nan

# ========== BUILD THE THREE DATAFRAMES ==========
# We will use MultiIndex columns: (Metric, Classifier)
acc_df = pd.DataFrame(index=BANDS, columns=pd.MultiIndex.from_product([METRICS, CLASSIFIERS]))
sens_df = pd.DataFrame(index=BANDS, columns=pd.MultiIndex.from_product([METRICS, CLASSIFIERS]))
spec_df = pd.DataFrame(index=BANDS, columns=pd.MultiIndex.from_product([METRICS, CLASSIFIERS]))

missing_files = []
for metric in METRICS:
    for band in BANDS:
        result_file = os.path.join(RESULTS_BASE, metric, band, "model_comparison.csv")
        if not os.path.exists(result_file):
            missing_files.append(result_file)
            continue
        df = pd.read_csv(result_file, index_col=0)
        for clf in CLASSIFIERS:
            if clf in df.index:
                acc_df.loc[band, (metric, clf)] = extract_mean(df.loc[clf, "accuracy"])
                sens_df.loc[band, (metric, clf)] = extract_mean(df.loc[clf, "sensitivity"])
                spec_df.loc[band, (metric, clf)] = extract_mean(df.loc[clf, "specificity"])
            else:
                # Classifier missing (e.g., XGBoost not installed)
                acc_df.loc[band, (metric, clf)] = np.nan
                sens_df.loc[band, (metric, clf)] = np.nan
                spec_df.loc[band, (metric, clf)] = np.nan

if missing_files:
    print("Warning: The following result files were missing. Tables will have NaN for those entries.")
    for f in missing_files:
        print(f"  {f}")

# ========== SAVE AS EXCEL ==========
excel_path = os.path.join(OUTPUT_DIR, "summary_tables.xlsx")
with ExcelWriter(excel_path, engine='openpyxl') as writer:
    acc_df.to_excel(writer, sheet_name="Accuracy")
    sens_df.to_excel(writer, sheet_name="Sensitivity")
    spec_df.to_excel(writer, sheet_name="Specificity")
print(f"Excel file saved: {excel_path}")

# ========== SAVE AS PDF (using matplotlib) ==========
def dataframe_to_pdf_table(df, title, pdf_pages):
    """Render a DataFrame as a table and add to PDF."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    # Flatten MultiIndex columns for display
    if isinstance(df.columns, pd.MultiIndex):
        col_flat = [f"{m}_{CLASSIFIER_LABELS.get(c,c)}" for m, c in df.columns]
    else:
        col_flat = df.columns
    table = ax.table(cellText=df.round(4).values,
                     rowLabels=df.index,
                     colLabels=col_flat,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title(title, fontsize=14, weight='bold', y=1.05)
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close()

pdf_path = os.path.join(OUTPUT_DIR, "summary_tables.pdf")
with PdfPages(pdf_path) as pdf:
    dataframe_to_pdf_table(acc_df, "Accuracy (mean over folds)", pdf)
    dataframe_to_pdf_table(sens_df, "Sensitivity / Recall", pdf)
    dataframe_to_pdf_table(spec_df, "Specificity", pdf)
print(f"PDF file saved: {pdf_path}")

# ========== ALSO SAVE AS CSV FILES ==========
for name, df in [("Accuracy", acc_df), ("Sensitivity", sens_df), ("Specificity", spec_df)]:
    # Flatten columns to single level for CSV
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = [f"{m}_{CLASSIFIER_LABELS.get(c,c)}" for m, c in df.columns]
        df_flat = df.copy()
        df_flat.columns = flat_cols
    else:
        df_flat = df
    csv_path = os.path.join(OUTPUT_DIR, f"{name}_table.csv")
    df_flat.to_csv(csv_path)
    print(f"CSV saved: {csv_path}")

print("\nAll tables generated successfully.")
print(f"Files saved in: {OUTPUT_DIR}")
print("  - summary_tables.xlsx (Excel with 3 sheets)")
print("  - summary_tables.pdf (3 pages, one per metric)")
print("  - Accuracy_table.csv, Sensitivity_table.csv, Specificity_table.csv")