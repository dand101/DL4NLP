import os
from datetime import datetime

import pandas as pd

from tests.subtask1.test_1.baseline_train import train
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

if __name__ == '__main__':

    language_files = ["eng.csv"]
    # language_files = []
    all_results = []
    if len(language_files) == 0:
        for file in os.listdir('../../../data/dev_phase/subtask1/train'):
            if file.endswith('.csv'):
                language_files.append(file)

    for file in language_files:
        print(f"=== TRAINING {file} ===")
        results = train(file)
        result = {"language": file}
        result.update(results)
        all_results.append(result)

    df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"outputs/training_results_{timestamp}.txt"
    os.makedirs("outputs", exist_ok=True)

    with open(filename, "w") as f:
        f.write(df.to_string(index=False))


