import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_per_language(
    input_folder,
    train_out,
    test_out,
    test_size=0.1,
    seed=42
):
    input_folder = Path(input_folder)
    train_out = Path(train_out)
    test_out = Path(test_out)

    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    for csv_path in input_folder.glob("*.csv"):
        df = pd.read_csv(csv_path)

        if "language" not in df.columns:
            lang = csv_path.stem
            df["language"] = lang
        else:
            lang = df["language"].iloc[0]

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["polarization"],
            random_state=seed
        )

        train_df.to_csv(train_out / f"{lang}_train.csv", index=False)
        test_df.to_csv(test_out / f"{lang}_test.csv", index=False)

        print(f"{lang}: train={len(train_df)}, test={len(test_df)}")

# split_per_language(
#     input_folder="../../data/dev_phase/subtask1/train",
#     train_out="../../data/dev_phase/subtask1/train_split",
#     test_out="../../data/dev_phase/subtask1/test",
#     test_size=0.1
# )


def combine_and_shuffle_csvs(folder, output_path, seed=42):
    csv_files = Path(folder).glob("*.csv")
    dfs = [pd.read_csv(f) for f in csv_files]

    if not dfs:
        raise ValueError("No CSV files found")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    combined.to_csv(output_path, index=False)

    return combined

combine_and_shuffle_csvs(
    folder="../../data/dev_phase/subtask1/train_split",
    output_path="../../data/dev_phase/subtask1/train/all_languages.csv"
)
