"""
P1 Article - Prediction Script for New Data (No Ground Truth)

Predicts specialized Gray Matter segmentation masks for new HC/MS cohort patients.

Outputs per patient:
  - {patient_id}_gm_mask.nii.gz   → binary gm mask (class 1)

Developer:
Mahdi Bashiri Bawil
"""

import tensorflow as tf
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import argparse

print("TensorFlow Version:", tf.__version__)


###################### GPU Configuration ######################

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✅ GPU memory growth enabled  ({len(physical_devices)} GPU(s) found)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️  No GPU detected – inference will run on CPU")


###################### Configuration ######################

class PredictConfig:
    """
    All settings for the new-data prediction pipeline.
    Edit the values in __init__ or pass overrides via the CLI at the bottom.
    """

    def __init__(
        self,
        # ── Model settings ──────────────────────────────────────────────────
        variant: int = 1,
        preprocessing: str = "standard",
        class_scenario: str = "binary",
        architecture_name: str = "unet",
        model_name: str = "best_dice_model.h5",
        fold_id: int = 0,

        # ── Slice range (1-based, inclusive) ────────────────────────────────
        # Only slices within [slice_start, slice_end] are fed to the model.
        # All other slices receive empty (zero) masks.
        slice_start: int = 1,
        slice_end: int = 20,

        # ── Data root ───────────────────────────────────────────────────────
        data_root: str = "/mnt/d/TEMP_P4",

        # ── Post-processing ─────────────────────────────────────────────────
        apply_postprocess: bool = False,
        min_object_size: int = 5,
        closing_kernel_size: int = 2,
    ):
        # Experiment
        self.variant = variant
        self.fold_id = fold_id
        self.preprocessing = preprocessing
        self.class_scenario = class_scenario
        self.architecture_name = architecture_name
        self.model_name = model_name

        # Classes
        self.num_classes = 2
        self.class_names = ["Background", "Specialized_GM"]

        # Image dimensions (must match training)
        self.img_width = 256
        self.img_height = 256

        # Slice range (1-based, inclusive)
        self.slice_start = slice_start
        self.slice_end = slice_end

        # Post-processing
        self.apply_postprocess = apply_postprocess
        print(f'\n \t apply_postprocess: {apply_postprocess} \n')
        self.min_object_size = min_object_size
        self.closing_kernel_size = closing_kernel_size

        # Data root
        self.data_root = Path(data_root)

        # Cohort sub-directories (relative to data_root)
        self.cohorts = {
            "HC": self.data_root / "HC_COHORT_PREP_prepared" / "FLAIR_Preprocessed",
            "MS": self.data_root / "MS_COHORT_PREP_prepared" / "FLAIR_Preprocessed",
        }

        # Model path
        self.results_dir = Path(
            f"results_fold_{fold_id}_var_{variant}_bet_zscore_gm"   # adjust if you use a single fold
        )
        self.models_dir = self.results_dir / "models" / f"{preprocessing}_{class_scenario}"

        # ── Print summary ────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print("PREDICTION CONFIGURATION (New Data)")
        print(f"{'='*70}")
        print(f"  Variant          : {self.variant}")
        print(f"  Fold             : {self.fold_id}")
        print(f"  Preprocessing    : {self.preprocessing}")
        print(f"  Class scenario   : {self.class_scenario} ({self.num_classes} classes)")
        print(f"  Architecture     : {self.architecture_name}")
        print(f"  Model file       : {self.model_name}")
        print(f"  Slice range      : {self.slice_start} – {self.slice_end}  (1-based)")
        print(f"  Post-processing  : {self.apply_postprocess}")
        print(f"  Data root        : {self.data_root}")
        print(f"{'='*70}\n")


###################### Utility Helpers ######################

def load_nifti(path: Path):
    """Load a NIfTI file and return (numpy_array, nib_image)."""
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img


def save_binary_nifti(mask: np.ndarray, save_path: Path, reference_img):
    """
    Save a binary 3-D mask as a NIfTI file.

    Args:
        mask         : (H, W, S) or (S, H, W) boolean/uint8 array
        save_path    : destination path (*.nii.gz)
        reference_img: nibabel image whose affine/header are reused
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    nifti_out = nib.Nifti1Image(
        mask.astype(np.uint8),
        reference_img.affine,
        reference_img.header,
    )
    nib.save(nifti_out, str(save_path))


def preprocess_slice(slice_2d: np.ndarray, target_h: int = 256, target_w: int = 256) -> np.ndarray:
    """
    Resize a 2-D slice to (target_h, target_w) if necessary and
    return a float32 array with shape (1, H, W, 1) ready for the model.

    The data files are assumed to be already normalised to [0, 1] and
    z-score normalised (as stated in the task description), so no
    additional intensity normalisation is applied here.
    """
    import cv2  # lightweight resize; falls back to zoom if cv2 unavailable

    h, w = slice_2d.shape
    if h != target_h or w != target_w:
        slice_2d = cv2.resize(
            slice_2d, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    # shape → (1, H, W, 1)
    return slice_2d[np.newaxis, :, :, np.newaxis].astype(np.float32)


def post_process_pred(pred_classes: np.ndarray, num_classes: int = 2,
                      min_object_size: int = 5, closing_kernel_size: int = 2) -> np.ndarray:
    """
    Morphological post-processing for a single 2-D prediction slice.
    Identical to the function used during training inference.

    Pipeline (per foreground class):
      1. Extract binary mask from the label map.
      2. binary_closing  – fill small holes / bridge tiny gaps.
      3. remove_small_objects – discard isolated noise specks.
      4. Reconstruct integer label map.
    """
    from skimage.morphology import remove_small_objects, binary_closing, disk

    kernel = disk(closing_kernel_size)

    def clean(mask):
        if not mask.any():
            return mask
        mask = binary_closing(mask, kernel)
        mask = remove_small_objects(mask, min_size=min_object_size)
        return mask

    gm_mask = (pred_classes == 1)

    gm_mask  = clean(gm_mask)

    post_pred = np.zeros_like(pred_classes)
    post_pred[gm_mask] = 1

    return post_pred


###################### Model Loading ######################

def load_model(config: PredictConfig, fold_id: int):
    """
    Build the model architecture and load weights for the given fold.

    Returns the loaded generator (keras Model).
    """
    if config.architecture_name == "unet":
        from unet_model import build_unet_3class as build_fn
    elif config.architecture_name == "attnunet":
        from attn_unet_model import build_attention_unet_3class as build_fn
    elif config.architecture_name == "dlv3unet":
        from dlv3_unet_model_GN import build_deeplabv3_unet_3class as build_fn
    elif config.architecture_name == "transunet":
        from trans_unet_model import build_trans_unet_3class as build_fn
    else:
        raise ValueError(f"Unknown architecture: {config.architecture_name}")

    model_path = (
        config.models_dir
        / f"fold_{fold_id}"
        / config.model_name
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    generator = build_fn(
        input_shape=(config.img_height, config.img_width, 1),
        num_classes=config.num_classes,
    )
    generator.load_weights(str(model_path))
    print(f"  ✅ Fold {fold_id} model loaded from: {model_path}")
    return generator


###################### Per-Patient Prediction ######################

def predict_patient(
    patient_id: str,
    flair_path: Path,
    brain_mask_path: Path,
    models: list,          # list of keras generators (one per fold)
    config: PredictConfig,
    gm_out_dir: Path,
):
    """
    Run inference for a single patient and save Specialized_GM masks.

    Steps:
      1. Load FLAIR volume and brain mask.
      2. Apply brain mask (multiply) → brain-extracted volume.
      3. For each slice in [slice_start, slice_end]:
           a. Resize to 256×256.
           b. Run through all fold models and average softmax outputs.
           c. argmax → class label.
           d. Optional post-processing.
      4. Slices outside the range → empty (zero) predictions.
      5. Save: main prediction, Specialized_GM binary mask.
    """
    # ── Load data ────────────────────────────────────────────────────────────
    flair_data, flair_img = load_nifti(flair_path)       # (H, W, S)
    brain_mask, _         = load_nifti(brain_mask_path)  # (H, W, S) binary

    # Brain extraction: zero out non-brain voxels
    brain_mask_bool = brain_mask > 0
    flair_brain = np.copy(flair_data)
    flair_brain[~brain_mask_bool] = np.min(flair_data)

    # flair_brain = flair_data * brain_mask                # (H, W, S)

    num_slices = flair_brain.shape[2]

    # Convert to 0-based slice indices for the active range
    # Input: slice_start / slice_end are 1-based (as stated in the task).
    active_start = config.slice_start - 1   # inclusive, 0-based
    active_end   = config.slice_end   - 1   # inclusive, 0-based

    # Clamp to actual volume depth
    active_start = max(0, active_start)
    active_end   = min(num_slices - 1, active_end)

    # Initialise output volumes (H, W, S) – same spatial shape as the input
    H, W = flair_brain.shape[0], flair_brain.shape[1]
    pred_volume = np.zeros((H, W, num_slices), dtype=np.uint8)  # main prediction
    gm_volume = np.zeros((H, W, num_slices), dtype=np.uint8)    # binary Specialized_GM

    # ── Inference loop ───────────────────────────────────────────────────────
    for s in range(num_slices):

        if s < active_start or s > active_end:
            # Outside desired range: leave masks empty
            continue

        slice_2d = flair_brain[:, :, s]                  # (H, W)
        model_input = preprocess_slice(                   # (1, 256, 256, 1)
            slice_2d, config.img_height, config.img_width
        )

        # Ensemble: average softmax probabilities across all fold models
        softmax_sum = np.zeros(
            (1, config.img_height, config.img_width, config.num_classes),
            dtype=np.float32,
        )
        for gen in models:
            softmax_sum += gen(model_input, training=False).numpy()

        softmax_avg = softmax_sum / len(models)           # (1, H, W, C)
        pred_slice  = np.argmax(softmax_avg, axis=-1)[0]  # (H, W)

        # Optional post-processing
        if config.apply_postprocess:
            pred_slice = post_process_pred(
                pred_slice,
                num_classes=config.num_classes,
                min_object_size=config.min_object_size,
                closing_kernel_size=config.closing_kernel_size,
            )

        # If model output is 256×256 but original slice is different size, resize back
        if pred_slice.shape != (H, W):
            import cv2
            pred_slice = cv2.resize(
                pred_slice.astype(np.float32), (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)

        pred_volume[:, :, s] = pred_slice

        # Binary masks
        # Specialized_GM = class 1 in 2-class
        gm_volume[:, :, s] = (pred_slice == 1).astype(np.uint8)

    # ── Save outputs ─────────────────────────────────────────────────────────
    gm_path = gm_out_dir / f"{patient_id}_gm_mask.nii.gz"

    save_binary_nifti(gm_volume, gm_path, flair_img)

    n_gm = int(gm_volume.sum())
    print(
        f"    Patient {patient_id}: "
        f"GM voxels = {n_gm:6d}"
    )
    print(f"      → {gm_path}")


###################### Main Prediction Pipeline ######################

def run_prediction(config: PredictConfig, fold_ids: list = None):
    """
    Full prediction pipeline for all patients in HC and MS cohorts.

    Args:
        config   : PredictConfig object.
        fold_ids : List of fold IDs to ensemble (e.g. [0, 1, 2, 3]).
                   If None, defaults to [0, 1, 2, 3].
    """
    if fold_ids is None:
        fold_ids = [0]

    # ── Load all fold models ─────────────────────────────────────────────────
    print(f"\nLoading models for folds: {fold_ids}")
    models = []
    for fold_id in fold_ids:
        gen = load_model(config, fold_id)
        models.append(gen)
    print(f"✅ {len(models)} model(s) loaded\n")

    # ── Iterate over cohorts ─────────────────────────────────────────────────
    for cohort_name, cohort_flair_dir in config.cohorts.items():
        files_dir       = cohort_flair_dir / "files"
        brain_masks_dir = cohort_flair_dir / "Brain_Masks"
        gm_out_dir    = cohort_flair_dir / "GM_Masks"

        # Create output directories
        gm_out_dir.mkdir(parents=True, exist_ok=True)

        # Discover patients from the files directory
        flair_files = sorted(files_dir.glob("*.nii.gz"))
        if not flair_files:
            print(f"⚠️  No FLAIR files found in {files_dir} – skipping {cohort_name} cohort")
            continue

        print(f"\n{'='*70}")
        print(f"COHORT: {cohort_name}  ({len(flair_files)} patients found)")
        print(f"  FLAIR dir        : {files_dir}")
        print(f"  Brain masks dir  : {brain_masks_dir}")
        print(f"  Output GM dir    : {gm_out_dir}")
        print(f"{'='*70}")

        skipped = 0
        for flair_path in tqdm(flair_files, desc=f"{cohort_name} patients"):
            # Extract 6-digit patient ID from filename
            patient_id = flair_path.stem.replace(".nii", "")  # handles double .nii.gz

            brain_mask_path = brain_masks_dir / f"{patient_id}_brain_mask.nii.gz"

            if not brain_mask_path.exists():    # or patient_id != '110214':
                print(
                    f"\n  ⚠️  Brain mask not found for patient {patient_id} "
                    f"(expected: {brain_mask_path}) – skipping"
                )
                skipped += 1
                continue

            try:
                predict_patient(
                    patient_id=patient_id,
                    flair_path=flair_path,
                    brain_mask_path=brain_mask_path,
                    models=models,
                    config=config,
                    gm_out_dir=gm_out_dir,
                )
            except Exception as exc:
                print(f"\n  ❌ Error processing patient {patient_id}: {exc}")
                skipped += 1

        done = len(flair_files) - skipped
        print(
            f"\n  ✅ {cohort_name} cohort done: {done} predicted, {skipped} skipped\n"
        )

    print("\n" + "="*70)
    print("ALL COHORTS PROCESSED")
    print("="*70)


###################### Entry Point ######################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="P4 – Predict Specialized_GM for new HC / MS cohort data"
    )
    parser.add_argument("--variant",          type=int,   default=1)
    parser.add_argument("--preprocessing",    type=str,   default="standard")
    parser.add_argument("--class_scenario",   type=str,   default="binary",
                        choices=["binary"])
    parser.add_argument("--architecture",     type=str,   default="unet",
                        choices=["unet", "attnunet", "dlv3unet", "transunet"])
    parser.add_argument("--model_name",       type=str,   default="best_dice_generator.h5")
    parser.add_argument("--folds",            type=int,   nargs="+", default=[0],
                        help="Fold IDs to ensemble (e.g. --folds 0 1 2 3)")
    parser.add_argument("--slice_start",      type=int,   default=1,
                        help="First slice to predict (1-based, inclusive)")
    parser.add_argument("--slice_end",        type=int,   default=20,
                        help="Last slice to predict (1-based, inclusive)")
    parser.add_argument("--data_root",        type=str,   default="/mnt/d/TEMP_P4")
    parser.add_argument("--no_postprocess",   action="store_false",
                        help="Disable morphological post-processing")
    parser.add_argument("--min_object_size",  type=int,   default=5)
    parser.add_argument("--closing_size",     type=int,   default=2)
    args = parser.parse_args()

    config = PredictConfig(
        variant=args.variant,
        preprocessing=args.preprocessing,
        class_scenario=args.class_scenario,
        architecture_name=args.architecture,
        model_name=args.model_name,
        slice_start=args.slice_start,
        slice_end=args.slice_end,
        data_root=args.data_root,
        apply_postprocess=not args.no_postprocess,
        min_object_size=args.min_object_size,
        closing_kernel_size=args.closing_size,
    )

    run_prediction(config, fold_ids=args.folds)
