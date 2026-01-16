import pandas as pd
import glob
import os

def load_and_tag(folder_path, split_name):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []

    for f in files:
        filename = os.path.basename(f).split('.')[0]

        lang = filename.replace("_augmented", "").replace("_aug", "")

        if "_augumented" in filename:
                lang = filename.replace("_augumented", "")
        
        df = pd.read_csv(f)
        df['text'] = df['text'].astype(str).fillna("")
        df['lang'] = lang
        df['split'] = split_name
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)
