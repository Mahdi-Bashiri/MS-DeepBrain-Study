import os
import json
import numpy as np
from sklearn.model_selection import KFold

# ─────────────────────────────────────────────
# Patient IDs
# ─────────────────────────────────────────────
local_patients_id = [
    '101228', '101627', '102035', '102313', '104252', '104280', '104420',
    '104447', '104453', '104474', '104518', '104520', '104670', '104797',
    '104810', '104871', '104899', '104937', '105074', '105302', '105465',
    '105549', '105597', '105755', '105911', '105917', '105978', '106063',
    '106200', '106270', '106506', '106536', '106639', '106780', '106905',
    '106976', '107130', '107233', '107455', '107508', '107539', '107630',
    '107680', '107739', '107966', '107997', '108295', '108344', '108444',
    '108726', '108807', '108975', '109141', '109267', '109395', '109654',
    '109816', '109923', '109944', '110012', '110157', '110218', '110280',
    '110327', '110497', '110540', '110543', '110784', '111008', '111140',
    '111189', '111489', '111691', '111852', '112055', '112378', '112414',
    '112657', '112659', '112730', '112765', '112776', '112997', '113046',
    '113394', '113845', '114058', '114128', '114266', '114304', '114454',
    '114525', '114585', '114770', '114836', '114903', '114990', '115588',
    '115628', '115788',
]

public_patients_id = [
    'c01p01', 'c01p02', 'c01p03', 'c01p04', 'c01p05',
    'c07p01', 'c07p02', 'c07p03', 'c07p04', 'c07p05',
    'c08p01', 'c08p02', 'c08p03', 'c08p04', 'c08p05',
]

RANDOM_SEED = 42
N_FOLDS = 4


# ─────────────────────────────────────────────────────────────────────────────
# make_folds_exact  (LOCAL)
#   Carves n_val_per_fold * n_folds patients as an exclusive val pool,
#   then rotates the val window.  Val sets are perfectly non-overlapping.
# ─────────────────────────────────────────────────────────────────────────────
def make_folds_exact(trainval, n_val_per_fold, n_folds, rng):
    arr = np.array(trainval)
    rng.shuffle(arr)

    total_val_pool = n_folds * n_val_per_fold           # 5 * 10 = 50
    assert total_val_pool <= len(arr), (
        f"Not enough trainval ({len(arr)}) for {n_folds} x {n_val_per_fold} val = {total_val_pool}"
    )
    val_pool   = arr[:total_val_pool]                   # 50 dedicated val patients
    train_base = arr[total_val_pool:]                   # 29 always-train patients

    folds = {}
    for fold_idx in range(n_folds):
        val_pts = val_pool[fold_idx * n_val_per_fold:(fold_idx + 1) * n_val_per_fold].tolist()
        other_val = np.concatenate([
            val_pool[:fold_idx * n_val_per_fold],
            val_pool[(fold_idx + 1) * n_val_per_fold:]
        ])
        train_pts = np.concatenate([other_val, train_base]).tolist()
        folds[f"fold_{fold_idx}"] = {
            "train_patients": train_pts,
            "val_patients":   val_pts,
            "n_train": len(train_pts),
            "n_val":   len(val_pts),
        }
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# make_folds_kfold  (PUBLIC)
#   With only 12 trainval patients and 5 folds, KFold is the only way to keep
#   val sets strictly non-overlapping. Val sizes will be 3,3,2,2,2.
#   (5 * 3 = 15 > 12, so exact 3 per fold is mathematically impossible without
#    overlap; KFold is the standard, correct solution.)
# ─────────────────────────────────────────────────────────────────────────────
def make_folds_kfold(trainval, n_folds, rng):
    arr = np.array(trainval)
    rng.shuffle(arr)

    kf = KFold(n_splits=n_folds, shuffle=False)        # arr already shuffled
    folds = {}
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(arr)):
        folds[f"fold_{fold_idx}"] = {
            "train_patients": arr[train_idx].tolist(),
            "val_patients":   arr[val_idx].tolist(),
            "n_train": len(train_idx),
            "n_val":   len(val_idx),
        }
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL  --  70 / 10 / 20
#   99 total  ->  test=20, val=10 per fold, train=69 per fold
# ─────────────────────────────────────────────────────────────────────────────
n_local = len(local_patients_id)                       # 99
n_local_test         = round(n_local * 0.20)           # 20
n_local_val_per_fold = round(n_local * 0.10)           # 10

rng_local = np.random.default_rng(RANDOM_SEED)
local_arr = np.array(local_patients_id)
rng_local.shuffle(local_arr)

local_test     = local_arr[:n_local_test].tolist()     # 20
local_trainval = local_arr[n_local_test:].tolist()     # 79

local_folds = make_folds_exact(
    local_trainval,
    n_val_per_fold=n_local_val_per_fold,
    n_folds=N_FOLDS,
    rng=np.random.default_rng(RANDOM_SEED + 1),
)

local_split = {
    "metadata": {
        "dataset": "Local_SAI",
        "total_patients": n_local,
        "test_patients": n_local_test,
        "trainval_patients": len(local_trainval),
        "target_split": "70/10/20 (train/val/test)",
        "exact_counts": "train=69, val=10, test=20 per fold",
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
    },
    "test_set": {"patients": local_test, "n_patients": n_local_test},
    "folds": local_folds,
}

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC  --  60 / 20 / 20
#   15 total  ->  test=3 (center-balanced), trainval=12
#   KFold(5) on 12 -> val sizes: 3,3,2,2,2  (non-overlapping, closest to 20%)
#   train sizes:                             9,9,10,10,10
# ─────────────────────────────────────────────────────────────────────────────
n_public = len(public_patients_id)                     # 15

# Center-balanced test: 1 patient per center
centers = {}
for pid in public_patients_id:
    centers.setdefault(pid[:3], []).append(pid)

public_test = []
public_trainval = []
for center, pids in sorted(centers.items()):
    arr = np.array(pids)
    np.random.default_rng(RANDOM_SEED + hash(center) % 1000).shuffle(arr)
    public_test.append(arr[0])           # 1 test per center  -> 3 total
    public_trainval += arr[1:].tolist()  # 4 trainval per center -> 12 total

public_folds = make_folds_kfold(
    public_trainval,
    n_folds=N_FOLDS,
    rng=np.random.default_rng(RANDOM_SEED + 2),
)

public_split = {
    "metadata": {
        "dataset": "Public_MSSEG",
        "total_patients": n_public,
        "test_patients": len(public_test),
        "trainval_patients": len(public_trainval),
        "target_split": "60/20/20 (train/val/test)",
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
        "center_balanced_test": True,
    },
    "test_set": {"patients": public_test, "n_patients": len(public_test)},
    "folds": public_folds,
}

# ─────────────────────────────────────────────────────────────────────────────
# CONCATENATED
# ─────────────────────────────────────────────────────────────────────────────
concat_test = local_test + public_test
concat_folds = {}
for fold_key in local_folds:
    lf = local_folds[fold_key]
    pf = public_folds[fold_key]
    concat_folds[fold_key] = {
        "train_patients": lf["train_patients"] + pf["train_patients"],
        "val_patients":   lf["val_patients"]   + pf["val_patients"],
        "n_train": lf["n_train"] + pf["n_train"],
        "n_val":   lf["n_val"]   + pf["n_val"],
    }

concat_split = {
    "metadata": {
        "datasets": ["Local_SAI", "Public_MSSEG"],
        "total_patients": n_local + n_public,
        "test_patients": len(concat_test),
        "trainval_patients": len(local_trainval) + len(public_trainval),
        "local_split": "70/10/20",
        "public_split": "60/20/20",
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
    },
    "test_set": {"patients": concat_test, "n_patients": len(concat_test)},
    "folds": concat_folds,
}

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
output_dir = os.path.dirname(os.path.abspath(__file__))

for name, data in [
    ("local_fold_assignments.json",  local_split),
    ("public_fold_assignments.json", public_split),
    ("concat_fold_assignments.json", concat_split),
]:
    path = os.path.join(output_dir, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SANITY CHECK ===")
for label, split_data in [("LOCAL", local_split), ("PUBLIC", public_split), ("CONCAT", concat_split)]:
    test_pts = set(split_data["test_set"]["patients"])
    print(f"\n{label}  (test={len(test_pts)})")
    val_sets = []
    for fold_key, fold in split_data["folds"].items():
        train_pts = set(fold["train_patients"])
        val_pts   = set(fold["val_patients"])
        val_sets.append(val_pts)
        tv_overlap   = len(train_pts & val_pts)
        tst_overlap  = len((train_pts | val_pts) & test_pts)
        print(f"  {fold_key}: train={len(train_pts):3d}, val={len(val_pts):2d} | "
              f"train/val overlap={tv_overlap} | (train+val)/test overlap={tst_overlap}")
    bad = [f"f{i}&f{j}" for i in range(len(val_sets)) for j in range(i+1, len(val_sets)) if val_sets[i] & val_sets[j]]
    print(f"  Val sets unique across folds: {'FAIL: ' + str(bad) if bad else 'OK'}")