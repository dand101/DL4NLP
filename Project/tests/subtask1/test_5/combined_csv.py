import pandas as pd
from pathlib import Path
import numpy as np


def balanced_split(df, label_col, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    counts = df[label_col].value_counts()
    if len(counts) < 2:
        n_test = max(1, int(round(len(df) * test_size)))
        test_df = df.iloc[:n_test]
        train_df = df.iloc[n_test:]
        return train_df, test_df

    n_test = int(round(len(df) * test_size))
    n_each = n_test // 2

    df0 = df[df[label_col] == 0]
    df1 = df[df[label_col] == 1]

    n_each = min(n_each, len(df0), len(df1))
    if n_each == 0:
        n_test = max(1, int(round(len(df) * test_size)))
        test_df = df.iloc[:n_test]
        train_df = df.iloc[n_test:]
        return train_df, test_df

    test0_idx = rng.choice(df0.index.to_numpy(), size=n_each, replace=False)
    test1_idx = rng.choice(df1.index.to_numpy(), size=n_each, replace=False)
    test_idx = np.concatenate([test0_idx, test1_idx])

    test_df = df.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = df.drop(index=test_idx).reset_index(drop=True)
    return train_df, test_df


def split_per_language(input_folder, train_out, test_out, label_col="polarization", test_size=0.1, seed=42):
    input_folder = Path(input_folder)
    train_out = Path(train_out)
    test_out = Path(test_out)

    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    for csv_path in input_folder.glob("*.csv"):
        df = pd.read_csv(csv_path)

        lang = csv_path.stem
        if "language" not in df.columns:
            df["language"] = lang

        if label_col in df.columns:
            train_df, test_df = balanced_split(df, label_col=label_col, test_size=test_size, seed=seed)
        else:
            n_test = max(1, int(round(len(df) * test_size)))
            test_df = df.sample(n=n_test, random_state=seed)
            train_df = df.drop(test_df.index)

        train_df.to_csv(train_out / f"{lang}_train.csv", index=False)
        test_df.to_csv(test_out / f"{lang}_test.csv", index=False)

        if label_col in df.columns:
            vc = test_df[label_col].value_counts()
            print(f"{lang}: train={len(train_df)}, test={len(test_df)}, test_counts={dict(vc)}")
        else:
            print(f"{lang}: train={len(train_df)}, test={len(test_df)}")


def combine_and_shuffle_csvs(folder, output_path, seed=42):
    folder = Path(folder)

    wanted = {
        "khm.csv",
        "mya.csv"
    }

    csv_files = [folder / f for f in wanted if (folder / f).exists()]
    dfs = [pd.read_csv(f) for f in csv_files]
    if not dfs:
        raise ValueError("No CSV files found")
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined

combine_and_shuffle_csvs(r"D:\Facultate\DL4NLP\Project\data\dev_phase\subtask1\train", r"D:\Facultate\DL4NLP\Project\data\dev_phase\subtask1\train\khm_mya.csv")
