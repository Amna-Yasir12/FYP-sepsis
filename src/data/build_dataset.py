import os
import pandas as pd
import numpy as np

RAW_DIR_A = "data/raw/physionet_training_setA"
RAW_DIR_B = "data/raw/physionet_training_setB"
PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

LABEL_COL = "SepsisLabel"
TIME_COL = "ICULOS"
ID_COL = "patient_id"
CAT_FEATURES = ["Gender", "Unit1", "Unit2"]

# Columns dropped for excessive missingness, following the reference paper's approach
HIGH_MISSING_COLS = [
    "TroponinI", "EtCO2", "Bilirubin_direct", "Bilirubin_total", "Fibrinogen",
    "Alkalinephos", "Calcium", "Phosphate", "SaO2", "PaCO2", "HCO3", "Magnesium", "Chloride",
]

WINDOW_LENGTH = 10      # hours per window, matches the paper
LABEL_HORIZON = 6       # standard PhysioNet horizon (we deliberately use 6, not the paper's non-standard 12)

def load_patient_file(filepath, patient_id):
    df = pd.read_csv(filepath, sep="|")
    df = df.drop(columns=[c for c in HIGH_MISSING_COLS if c in df.columns])
    df["ShockIndex"] = df["HR"] / (df["SBP"] + 1e-6)
    df = df.ffill()
    df[ID_COL] = patient_id
    return df

def load_all_patients(raw_dir, hospital_label):
    all_dfs = []
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".psv"))
    for fname in files:
        patient_id = fname.replace("p", "").replace(".psv", "")
        filepath = os.path.join(raw_dir, fname)
        df = load_patient_file(filepath, patient_id)
        df["hospital"] = hospital_label
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

def main():
    print("Loading Set A...")
    df_a = load_all_patients(RAW_DIR_A, "A")
    print(f"Set A: {len(df_a)} rows, {df_a[ID_COL].nunique()} patients")

    print("Loading Set B...")
    df_b = load_all_patients(RAW_DIR_B, "B")
    print(f"Set B: {len(df_b)} rows, {df_b[ID_COL].nunique()} patients")

    df_combined = pd.concat([df_a, df_b], ignore_index=True)
    print(f"Combined: {len(df_combined)} rows, {df_combined[ID_COL].nunique()} patients")

    out_path = os.path.join(PROC_DIR, "combined_raw.csv")
    df_combined.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
