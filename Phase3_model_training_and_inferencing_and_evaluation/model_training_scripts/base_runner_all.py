import subprocess
import sys
import numpy as np


# Run scripts one after another


for fold in range(4):

    # Skip folds:
    # if fold in list(np.array([0])):
    #     continue

    for variant in range(5):

        # Skip variants:
        if variant in list(np.array([0])):
            continue

        # subprocess.run([sys.executable, "p4_run_experiments_all.py", "--variant", str(variant), "--fold", str(fold), "--scenario", "standard_3class"])


