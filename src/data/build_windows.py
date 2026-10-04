import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

PROC_DIR = "data/processed"
COMBINED_PATH = os.path.join(PROC_DIR, "combined_raw.csv")

SEED = 42
np.random.seed(SEED)

LABEL_COL = "SepsisLabel"
TIME_COL = "ICULOS"
ID_COL = "patient_id"
CAT_FEATURES = ["Gender", "Unit1", "Unit2"]

WINDOW_LENGTH = 10
LABEL_HORIZON = 6

def load_and_split():
    df = pd.read_csv(COMBINED_PATH)

    patient_sepsis = df.groupby(ID_COL)[LABEL_COL].max()
    patient_ids = patient_sepsis.index.values
    patient_labels = patient_sepsis.values

    train_ids, temp_ids, train_y, temp_y = train_test_split(
        patient_ids, patient_labels, test_size=0.3, random_state=SEED, stratify=patient_labels
    )
    valid_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=SEED, stratify=temp_y
    )

    df_train = df[df[ID_COL].isin(train_ids)].copy()
    df_valid = df[df[ID_COL].isin(valid_ids)].copy()
    df_test = df[df[ID_COL].isin(test_ids)].copy()

    print(f"Train: {df_train[ID_COL].nunique()} patients ({train_y.mean()*100:.1f}% sepsis)")
    print(f"Valid: {df_valid[ID_COL].nunique()} patients")
    print(f"Test: {df_test[ID_COL].nunique()} patients")

    return df_train, df_valid, df_test

def scale_and_impute(df_train, df_valid, df_test):
    exclude_cols = [ID_COL, LABEL_COL, TIME_COL, "hospital"] + CAT_FEATURES
    num_cols = [c for c in df_train.columns if c not in exclude_cols]

    for col in CAT_FEATURES:
        if col in df_train.columns:
            mode_val = df_train[col].mode(dropna=True)[0]
            for d in (df_train, df_valid, df_test):
                d[col] = d[col].fillna(mode_val)

    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_valid[num_cols] = scaler.transform(df_valid[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])

    imputer = IterativeImputer(max_iter=10, random_state=SEED, n_nearest_features=10)
    df_train[num_cols] = imputer.fit_transform(df_train[num_cols])
    df_valid[num_cols] = imputer.transform(df_valid[num_cols])
    df_test[num_cols] = imputer.transform(df_test[num_cols])

    return df_train, df_valid, df_test, num_cols

def build_windows_for_patient(df_patient, feature_cols):
    df_patient = df_patient.sort_values(TIME_COL).reset_index(drop=True)
    has_sepsis = (df_patient[LABEL_COL] == 1).any()

    if has_sepsis:
        onset_idx = df_patient.index[df_patient[LABEL_COL] == 1][0]
        onset_time = df_patient.loc[onset_idx, TIME_COL]
        usable = df_patient.iloc[:onset_idx + 1]
        stride = 1
    else:
        onset_time = -1
        usable = df_patient
        stride = WINDOW_LENGTH

    windows, labels, end_times = [], [], []
    n = len(usable)

    for end in range(WINDOW_LENGTH, n + 1, stride):
        start = end - WINDOW_LENGTH
        window = usable.iloc[start:end][feature_cols].values
        end_time = usable.iloc[end - 1][TIME_COL]
        label = 1 if (has_sepsis and onset_time - LABEL_HORIZON <= end_time <= onset_time) else 0
        windows.append(window)
        labels.append(label)
        end_times.append(end_time)

    if n < WINDOW_LENGTH and n > 0:
        pad_rows = WINDOW_LENGTH - n
        real = usable[feature_cols].values
        padded = np.vstack([np.zeros((pad_rows, real.shape[1])), real])
        end_time = usable.iloc[-1][TIME_COL]
        label = 1 if (has_sepsis and onset_time - LABEL_HORIZON <= end_time <= onset_time) else 0
        windows.append(padded)
        labels.append(label)
        end_times.append(end_time)

    return windows, labels, end_times, [onset_time] * len(windows)

def main():
    print("Loading and splitting patients...")
    df_train, df_valid, df_test = load_and_split()

    print("Scaling and imputing...")
    df_train, df_valid, df_test, feature_cols = scale_and_impute(df_train, df_valid, df_test)

    for split_name, df_split in [("train", df_train), ("valid", df_valid), ("test", df_test)]:
        print(f"Building windows for {split_name}...")
        all_windows, all_labels, all_end_times, all_onsets, all_hospitals = [], [], [], [], []

        for patient_id, df_patient in df_split.groupby(ID_COL):
            hospital = df_patient["hospital"].iloc[0]
            windows, labels, end_times, onsets = build_windows_for_patient(df_patient, feature_cols)
            all_windows.extend(windows)
            all_labels.extend(labels)
            all_end_times.extend(end_times)
            all_onsets.extend(onsets)
            all_hospitals.extend([hospital] * len(windows))

        X = np.stack(all_windows)
        y = np.array(all_labels)

        print(f"  {split_name}: {X.shape[0]} windows, {int(y.sum())} positive ({y.mean()*100:.2f}%)")

        out = {
            "X": X,
            "y": y,
            "end_times": np.array(all_end_times),
            "onset_times": np.array(all_onsets),
            "hospital": np.array(all_hospitals),
            "feature_cols": feature_cols,
        }
        with open(os.path.join(PROC_DIR, f"{split_name}_windows.pkl"), "wb") as f:
            pickle.dump(out, f)

    print("Done.")


if __name__ == "__main__":
    main()
