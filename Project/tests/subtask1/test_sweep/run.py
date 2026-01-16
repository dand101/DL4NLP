import sys
import subprocess

import wandb
import sys
print(">>> PYTHON EXECUTABLE:", sys.executable)
print(">>> PYTHON VERSION:", sys.version)

wandb.agent(
    sweep_id="mwbiss3b",
    entity="gheorghitastefana-alexandru-ioan-cuza-university-iasi",
    project="DL4NLP-Project_tests_subtask1_test_sweep",
    count=30
)
