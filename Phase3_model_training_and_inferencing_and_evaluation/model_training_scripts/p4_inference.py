"""
P4 Article - Inference Script for ventricles and WMH segmentation task

Developer:
Mahdi Bashiri Bawil
"""

import tensorflow as tf
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import nibabel as nib
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report

from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion
from scipy.ndimage import label as nd_label

from unet_model import build_unet_3class # must be updated with the actual used model for traininig

# Import data loader
from p4_data_loader import DataConfig, P2DataLoader

# Error analysis
from p4_error_analysis import run_error_analysis


print("TensorFlow Version:", tf.__version__)

###################### GPU Configuration ######################

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("✅ GPU memory growth enabled")
        print(f"   Available GPUs: {len(physical_devices)}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️  No GPU detected - inference will be slow")


###################### Inference Configuration ######################

class InferenceConfig:
    """Configuration for inference"""
    
    def __init__(self, 
                 variant: int = 5,
                 preprocessing: str = 'standard',
                 class_scenario: str = '4class',
                 fold_id: int = 0,
                 model_name: str = 'best_dice_generator.h5',
                 architecture_name: str = 'unet'
                 ):
        
        # Experiment identification
        self.variant = variant
        self.preprocessing = preprocessing
        self.class_scenario = class_scenario
        self.fold_id = fold_id
        self.model_name = model_name
        self.architecture_name = architecture_name
        
        # Number of classes
        self.num_classes = 3 if class_scenario == '3class' else 4
        
        # Class names
        if self.num_classes == 4:
            self.class_names = ['Background', 'Ventricles', 'Normal_WMH', 'Abnormal_WMH']
        elif self.num_classes == 3:
            self.class_names = ['Background', 'Ventricles', 'Abnormal_WMH']
        
        # Image dimensions
        self.batch_size = 1  # Use batch_size=1 for inference
        self.img_width = 256
        self.img_height = 256
        
        # Paths
        self.results_dir = Path(f"results_fold_{fold_id}_var_{variant}_zscore2")
        self.models_dir = self.results_dir / "models" / f"{preprocessing}_{class_scenario}"
        self.checkpoint_dir = self.models_dir / f"fold_{fold_id}"
        
        # Output directories
        self.inference_dir = self.results_dir / "inference_all_test" / f"{preprocessing}_{class_scenario}"
        # self.predictions_dir = self.inference_dir / "predictions"
        self.visualizations_dir = self.inference_dir / "visualizations"
        self.metrics_dir = self.inference_dir / "metrics"
        
        # Create directories
        # self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Model path
        self.model_path = self.checkpoint_dir / self.model_name
        
        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"\n{'='*70}")
        print(f"INFERENCE CONFIGURATION")
        print(f"{'='*70}")
        print(f"Variant: {self.variant}")
        print(f"Preprocessing: {self.preprocessing}")
        print(f"Class scenario: {self.class_scenario} ({self.num_classes} classes)")
        print(f"Fold: {self.fold_id}")
        print(f"Architecture: {self.architecture_name}")
        print(f"Model: {self.model_name}")
        print(f"Model path: {self.model_path}")
        print(f"Output directory: {self.inference_dir}")
        print(f"{'='*70}\n")


###################### Utility Functions ######################

def prepare_input(paired_input):
    """
    Extract and normalize FLAIR from paired input
    
    Args:
        paired_input: (bs, 256, 512, 1) with FLAIR + mask
        
    Returns:
        flair_normalized: FLAIR normalized to [-1, 1]
    """
    # Extract FLAIR (left half)
    flair_normalized = paired_input[:, :, :256, :]
    return flair_normalized

def compute_hd95(mask1, mask2):
    """
    Compute 95th percentile Hausdorff Distance between two binary masks
    
    Args:
        mask1: Binary mask 1
        mask2: Binary mask 2
    
    Returns:
        HD95 value in pixels
    """
    # Get boundary points
    if not np.any(mask1) or not np.any(mask2):
        return np.nan
    
    # Compute distance transforms
    dt1 = distance_transform_edt(~mask1.astype(bool))
    dt2 = distance_transform_edt(~mask2.astype(bool))
    
    # Get surface points
    surface1 = mask1.astype(bool) & (dt1 <= 1)
    surface2 = mask2.astype(bool) & (dt2 <= 1)
    
    if not np.any(surface1) or not np.any(surface2):
        return np.nan
    
    # Get coordinates of surface points
    coords1 = np.argwhere(surface1)
    coords2 = np.argwhere(surface2)
    
    # Compute distances from surface1 to surface2
    distances1 = np.min(np.sqrt(np.sum((coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]) ** 2, axis=2)), axis=1)
    # Compute distances from surface2 to surface1
    distances2 = np.min(np.sqrt(np.sum((coords2[:, np.newaxis, :] - coords1[np.newaxis, :, :]) ** 2, axis=2)), axis=1)
    
    # Combine distances
    all_distances = np.concatenate([distances1, distances2])
    
    # Return 95th percentile
    return np.percentile(all_distances, 95)

def compute_hd95_3d(mask1, mask2):
    """
    Compute 95th percentile Hausdorff Distance for 3D volume
    Uses only surface voxels for efficiency
    
    Args:
        mask1: Binary mask (N, H, W)
        mask2: Binary mask (N, H, W)
    
    Returns:
        HD95 value in pixels
    """
    if not np.any(mask1) or not np.any(mask2):
        return np.nan
    
    # Extract surface voxels only (border voxels)
    from scipy.ndimage import binary_erosion
    
    # Surface = original mask minus eroded mask
    surface1 = mask1.astype(bool) & ~binary_erosion(mask1.astype(bool))
    surface2 = mask2.astype(bool) & ~binary_erosion(mask2.astype(bool))
    
    # Get surface coordinates
    coords1 = np.argwhere(surface1)
    coords2 = np.argwhere(surface2)
    
    if len(coords1) == 0 or len(coords2) == 0:
        return np.nan
    
    # Subsample if still too large (>10k points each)
    max_points = 10000
    if len(coords1) > max_points:
        idx1 = np.random.choice(len(coords1), max_points, replace=False)
        coords1 = coords1[idx1]
    if len(coords2) > max_points:
        idx2 = np.random.choice(len(coords2), max_points, replace=False)
        coords2 = coords2[idx2]
    
    # Compute distances
    distances1 = np.min(cdist(coords1, coords2, metric='euclidean'), axis=1)
    distances2 = np.min(cdist(coords2, coords1, metric='euclidean'), axis=1)
    
    # Combine all distances
    all_distances = np.concatenate([distances1, distances2])
    
    # Return 95th percentile
    return np.percentile(all_distances, 95)


def compute_lesion_level_metrics(gt_volume, pred_volume, iou_threshold=0.1):
    """
    Compute lesion-level (instance-level) metrics by treating each connected
    component in the GT as an individual lesion.

    A GT lesion is considered DETECTED if its overlap (IoU) with any single
    predicted component exceeds `iou_threshold`.
    A predicted component is a TRUE POSITIVE if it overlaps any GT lesion
    above threshold, otherwise it is a FALSE POSITIVE lesion.

    Args:
        gt_volume   : binary 3-D numpy array (S, H, W) — ground truth for ONE class
        pred_volume : binary 3-D numpy array (S, H, W) — prediction for ONE class
        iou_threshold: minimum IoU to count a GT lesion as detected (default 0.1)

    Returns:
        dict with keys:
            n_gt_lesions      : total number of GT lesions
            n_pred_lesions    : total number of predicted lesion clusters
            tp_lesions        : GT lesions that were detected
            fn_lesions        : GT lesions that were missed
            fp_lesions        : predicted clusters with no GT overlap
            lesion_sensitivity: tp_lesions / n_gt_lesions
            lesion_precision  : tp_lesions / n_pred_lesions
            lesion_f1         : harmonic mean of lesion sensitivity and precision
    """
    gt_bin   = gt_volume.astype(bool)
    pred_bin = pred_volume.astype(bool)

    # Label connected components
    gt_labeled,   n_gt   = nd_label(gt_bin)
    pred_labeled, n_pred = nd_label(pred_bin)

    tp_lesions = 0
    detected_pred_ids = set()

    for gt_id in range(1, n_gt + 1):
        gt_mask = (gt_labeled == gt_id)
        # Find all predicted components that overlap this GT lesion
        overlapping_pred_ids = np.unique(pred_labeled[gt_mask])
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

        detected = False
        for pred_id in overlapping_pred_ids:
            pred_mask = (pred_labeled == pred_id)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union        = np.logical_or(gt_mask, pred_mask).sum()
            iou = intersection / (union + 1e-7)
            if iou >= iou_threshold:
                detected = True
                detected_pred_ids.add(pred_id)

        if detected:
            tp_lesions += 1

    fn_lesions = n_gt   - tp_lesions
    fp_lesions = n_pred - len(detected_pred_ids)

    lesion_sensitivity = tp_lesions / (n_gt   + 1e-7)
    lesion_precision   = tp_lesions / (n_pred + 1e-7) if n_pred > 0 else 0.0
    lesion_f1 = (2 * lesion_sensitivity * lesion_precision /
                 (lesion_sensitivity + lesion_precision + 1e-7))

    return {
        'n_gt_lesions'       : int(n_gt),
        'n_pred_lesions'     : int(n_pred),
        'tp_lesions'         : int(tp_lesions),
        'fn_lesions'         : int(fn_lesions),
        'fp_lesions'         : int(fp_lesions),
        'lesion_sensitivity' : float(lesion_sensitivity),
        'lesion_precision'   : float(lesion_precision),
        'lesion_f1'          : float(lesion_f1),
    }


def compute_metrics_from_predictions(y_true, y_pred, num_classes, exclude_class=None):
    """
    Compute comprehensive metrics from predictions
    
    Args:
        y_true: Ground truth class labels (N, H, W)
        y_pred: Predicted class labels (N, H, W)
        num_classes: Number of classes
        exclude_class: Class to exclude from metrics (e.g., 2 for Normal_WMH in 4-class)
    
    Returns:
        Dictionary containing metrics
    """
    # Convert to one-hot
    y_true_onehot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)
    y_pred_onehot = tf.one_hot(y_pred, depth=num_classes, dtype=tf.float32)
    
    # Flatten spatial dimensions
    y_true_flat = tf.reshape(y_true_onehot, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred_onehot, [-1, num_classes])
    
    # Convert to numpy
    y_true_np = y_true_flat.numpy()
    y_pred_np = y_pred_flat.numpy()
    
    metrics = {
        'dice': {},
        'precision': {},
        'recall': {},
        'iou': {},
        'specificity': {},
        'hd95': {},
        'TP': {}
    }
    
    classes_to_evaluate = [c for c in range(num_classes) if c != exclude_class]
    
    for class_idx in classes_to_evaluate:
        # Extract binary masks for this class
        true_class = y_true_np[:, class_idx]
        pred_class = y_pred_np[:, class_idx]
        
        # Compute confusion matrix elements
        TP = np.sum((true_class == 1) & (pred_class == 1))
        FP = np.sum((true_class == 0) & (pred_class == 1))
        FN = np.sum((true_class == 1) & (pred_class == 0))
        TN = np.sum((true_class == 0) & (pred_class == 0))
        
        # Dice Score: 2*TP / (2*TP + FP + FN)
        dice = (2 * TP) / (2 * TP + FP + FN + 1e-7)
        
        # Precision: TP / (TP + FP)
        precision = TP / (TP + FP + 1e-7)
        
        # Recall (Sensitivity): TP / (TP + FN)
        recall = TP / (TP + FN + 1e-7)
        
        # IoU (Jaccard): TP / (TP + FP + FN)
        iou = TP / (TP + FP + FN + 1e-7)
        
        # Specificity: TN / (TN + FP)
        specificity = TN / (TN + FP + 1e-7)
        
        # HD95: Hausdorff Distance 95th percentile
        # Compute on entire volume (all samples combined) for fairness
        true_class_volume = y_true_np[:, class_idx].reshape(y_true.shape[0], y_true.shape[1], y_true.shape[2])
        pred_class_volume = y_pred_np[:, class_idx].reshape(y_pred.shape[0], y_pred.shape[1], y_pred.shape[2])

        hd95_value = compute_hd95_3d(true_class_volume, pred_class_volume)

        metrics['dice'][f'class_{class_idx}'] = float(dice)
        metrics['precision'][f'class_{class_idx}'] = float(precision)
        metrics['recall'][f'class_{class_idx}'] = float(recall)
        metrics['iou'][f'class_{class_idx}'] = float(iou)
        metrics['specificity'][f'class_{class_idx}'] = float(specificity)
        metrics['hd95'][f'class_{class_idx}'] = float(hd95_value)
        metrics['TP'][f'class_{class_idx}'] = float(TP)
    
    # Compute mean metrics (excluding the excluded class)
    for metric_name in ['dice', 'precision', 'recall', 'iou', 'specificity', 'hd95', 'TP']:
        metrics[metric_name]['mean'] = np.mean([v for v in metrics[metric_name].values()])

    # --- Lesion-level metrics (connected-component analysis) ---
    metrics['lesion'] = {}
    for class_idx in classes_to_evaluate:
        if class_idx <= 1:   # skip background and ventricles
            continue
        true_vol = y_true_np[:, class_idx].reshape(y_true.shape)
        pred_vol = y_pred_np[:, class_idx].reshape(y_pred.shape)
        metrics['lesion'][f'class_{class_idx}'] = compute_lesion_level_metrics(
            true_vol, pred_vol, iou_threshold=0.1
        )
        
    return metrics


# def aggregate_patient_metrics(per_patient_metrics, num_classes):
#     """
#     Returns both a flat structure (compatible with original overall_metrics)
#     and an extended structure with std/n for richer reporting.
#     """
#     flat_metrics = {m: {} for m in ['dice', 'precision', 'recall', 'iou', 'specificity', 'hd95', 'TP']}
#     rich_metrics = {m: {} for m in ['dice', 'precision', 'recall', 'iou', 'specificity', 'hd95', 'TP']}

#     metric_names = ['dice', 'precision', 'recall', 'iou', 'specificity', 'hd95', 'TP']

#     for metric_name in metric_names:
#         for class_idx in range(num_classes):
#             if class_idx == 0: continue

#             key = f'class_{class_idx}'

#             values = [
#                 per_patient_metrics[pid][metric_name][key]
#                 for pid in per_patient_metrics
#                 if key in per_patient_metrics[pid][metric_name]
#                 and not np.isnan(per_patient_metrics[pid][metric_name][key])
#             ]

#             TP_values = [
#                 per_patient_metrics[pid]['TP'][key]
#                 for pid in per_patient_metrics
#                 if key in per_patient_metrics[pid]['TP']
#                 and not np.isnan(per_patient_metrics[pid]['TP'][key])
#             ]

#             weighted_mean_values = np.sum((np.array(values) * np.array(TP_values)) / np.sum(np.array(TP_values)))

#             mean_val = float(np.mean(values)) if values else np.nan
#             std_val  = float(np.std(values))  if values else np.nan

#             # Flat: backward compatible with all existing print/save code
#             flat_metrics[metric_name][key] = weighted_mean_values if metric_name != 'hd95' else mean_val

#             # Rich: for extended reporting
#             rich_metrics[metric_name][key] = {
#                 'mean': mean_val,
#                 'std':  std_val,
#                 'n':    len(values)
#             }

#         # Mean across classes — same for both
#         class_means = [
#             flat_metrics[metric_name][f'class_{c}']
#             for c in range(num_classes)
#             if c!=0 and not np.isnan(flat_metrics[metric_name][f'class_{c}'])
#         ]
#         mean_across_classes = float(np.mean(class_means)) if class_means else np.nan
#         flat_metrics[metric_name]['mean'] = mean_across_classes
#         rich_metrics[metric_name]['mean'] = mean_across_classes

#     return flat_metrics, rich_metrics

def aggregate_patient_metrics(per_patient_metrics, num_classes):
    """
    Returns both a flat structure (compatible with original overall_metrics)
    and an extended structure with std/n for richer reporting.
    
    Includes lesion-level metrics (connected-component analysis):
        - lesion_sensitivity : mean across patients of (tp_lesions / n_gt_lesions)
        - lesion_precision   : mean across patients of (tp_lesions / n_pred_lesions)
        - lesion_f1          : mean across patients of harmonic mean of the above
        - n_gt_lesions       : total GT lesions summed across all patients
        - n_pred_lesions     : total predicted lesion clusters summed across all patients
        - tp_lesions         : total TP lesions summed across all patients
        - fn_lesions         : total FN lesions summed across all patients
        - fp_lesions         : total FP lesions summed across all patients
    """
    # ── Voxel-level metrics (unchanged) ─────────────────────────────────────
    voxel_metric_names = ['dice', 'precision', 'recall', 'iou', 'specificity', 'hd95', 'TP']
    flat_metrics = {m: {} for m in voxel_metric_names}
    rich_metrics = {m: {} for m in voxel_metric_names}

    for metric_name in voxel_metric_names:
        for class_idx in range(num_classes):
            if class_idx == 0:
                continue

            key = f'class_{class_idx}'

            values = [
                per_patient_metrics[pid][metric_name][key]
                for pid in per_patient_metrics
                if key in per_patient_metrics[pid][metric_name]
                and not np.isnan(per_patient_metrics[pid][metric_name][key])
            ]

            TP_values = [
                per_patient_metrics[pid]['TP'][key]
                for pid in per_patient_metrics
                if key in per_patient_metrics[pid]['TP']
                and not np.isnan(per_patient_metrics[pid]['TP'][key])
            ]

            weighted_mean_values = np.sum(
                (np.array(values) * np.array(TP_values)) / np.sum(np.array(TP_values))
            )

            mean_val = float(np.mean(values)) if values else np.nan
            std_val  = float(np.std(values))  if values else np.nan

            flat_metrics[metric_name][key] = weighted_mean_values if metric_name != 'hd95' else mean_val
            rich_metrics[metric_name][key] = {
                'mean': mean_val,
                'std':  std_val,
                'n':    len(values)
            }

        # Mean across classes
        class_means = [
            flat_metrics[metric_name][f'class_{c}']
            for c in range(num_classes)
            if c != 0 and not np.isnan(flat_metrics[metric_name][f'class_{c}'])
        ]
        mean_across_classes = float(np.mean(class_means)) if class_means else np.nan
        flat_metrics[metric_name]['mean'] = mean_across_classes
        rich_metrics[metric_name]['mean'] = mean_across_classes

    # ── Lesion-level metrics (new) ───────────────────────────────────────────
    # Scalar fields: averaged across patients (mean ± std)
    lesion_scalar_keys = ['lesion_sensitivity', 'lesion_precision', 'lesion_f1']
    # Count fields: summed across patients (total pool)
    lesion_count_keys  = ['n_gt_lesions', 'n_pred_lesions', 'tp_lesions', 'fn_lesions', 'fp_lesions']

    flat_metrics['lesion'] = {}
    rich_metrics['lesion'] = {}

    for class_idx in range(num_classes):
        if class_idx <= 1:   # skip background and ventricles
            continue

        key = f'class_{class_idx}'
        flat_metrics['lesion'][key] = {}
        rich_metrics['lesion'][key] = {}

        # --- Scalar metrics: mean ± std across patients ---
        for sk in lesion_scalar_keys:
            vals = [
                per_patient_metrics[pid]['lesion'][key][sk]
                for pid in per_patient_metrics
                if 'lesion' in per_patient_metrics[pid]
                and key in per_patient_metrics[pid]['lesion']
            ]
            mean_val = float(np.mean(vals)) if vals else np.nan
            std_val  = float(np.std(vals))  if vals else np.nan
            flat_metrics['lesion'][key][sk] = mean_val
            rich_metrics['lesion'][key][sk] = {
                'mean': mean_val,
                'std':  std_val,
                'n':    len(vals)
            }

        # --- Count metrics: sum across patients ---
        for ck in lesion_count_keys:
            vals = [
                per_patient_metrics[pid]['lesion'][key][ck]
                for pid in per_patient_metrics
                if 'lesion' in per_patient_metrics[pid]
                and key in per_patient_metrics[pid]['lesion']
            ]
            flat_metrics['lesion'][key][ck] = int(np.sum(vals)) if vals else 0
            rich_metrics['lesion'][key][ck] = int(np.sum(vals)) if vals else 0

    # Mean lesion scalars across foreground classes
    for sk in lesion_scalar_keys:
        class_vals = [
            flat_metrics['lesion'][f'class_{c}'][sk]
            for c in range(num_classes)
            if c > 1 and not np.isnan(flat_metrics['lesion'][f'class_{c}'][sk])
        ]
        mean_across = float(np.mean(class_vals)) if class_vals else np.nan
        flat_metrics['lesion'][f'mean_{sk}'] = mean_across
        rich_metrics['lesion'][f'mean_{sk}'] = mean_across

    # Summed counts across foreground classes
    for ck in lesion_count_keys:
        flat_metrics['lesion'][f'total_{ck}'] = int(np.sum([
            flat_metrics['lesion'][f'class_{c}'][ck]
            for c in range(num_classes) if c > 1
        ]))
        rich_metrics['lesion'][f'total_{ck}'] = flat_metrics['lesion'][f'total_{ck}']

    return flat_metrics, rich_metrics


###################### Original Visualization Functions ######################

def visualize_prediction(flair, ground_truth, prediction, 
                        probability_map, save_path, 
                        sample_id, num_classes):
    """
    Create comprehensive visualization of prediction
    
    Args:
        flair: Input FLAIR image (H, W)
        ground_truth: Ground truth mask (H, W)
        prediction: Predicted mask (H, W)
        probability_map: Max probability map (H, W)
        save_path: Path to save figure
        sample_id: Sample identifier
        num_classes: Number of classes
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Input FLAIR
    axes[0, 0].imshow(flair, cmap='gray')
    axes[0, 0].set_title('Input FLAIR', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth
    im1 = axes[0, 1].imshow(ground_truth, cmap='jet', vmin=0, vmax=num_classes-1)
    axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Prediction
    im2 = axes[0, 2].imshow(prediction, cmap='jet', vmin=0, vmax=num_classes-1)
    axes[0, 2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Max probability
    im3 = axes[1, 0].imshow(probability_map, cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Error map
    error_map = (prediction != ground_truth).astype(float)
    im4 = axes[1, 1].imshow(error_map, cmap='Reds', vmin=0, vmax=1)
    axes[1, 1].set_title('Error Map (Red=Wrong)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Overlay: FLAIR + Prediction contours
    axes[1, 2].imshow(flair, cmap='gray')
    # Create contours for each class
    from scipy import ndimage
    for class_idx in range(1, num_classes):  # Skip background
        class_mask = (prediction == class_idx)
        contours = class_mask ^ ndimage.binary_erosion(class_mask)
        if np.any(contours):
            axes[1, 2].contour(contours, colors=[plt.cm.jet(class_idx/(num_classes-1))], linewidths=1.5)
    axes[1, 2].set_title('FLAIR + Prediction Overlay', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Sample: {sample_id}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_prediction_short(flair, ground_truth, prediction, 
                        probability_map, save_path, 
                        sample_id, num_classes):
    """
    Create comprehensive visualization of prediction
    
    Args:
        flair: Input FLAIR image (H, W)
        ground_truth: Ground truth mask (H, W)
        prediction: Predicted mask (H, W)
        probability_map: Max probability map (H, W)
        save_path: Path to save figure
        sample_id: Sample identifier
        num_classes: Number of classes
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))

    cmap = plt.cm.jet
    flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    flair_rgb = np.stack([flair_norm] * 3, axis=-1)

    for ax, mask, title in zip(axes, [ground_truth, prediction], ['Ground Truth Overlay', 'Prediction Overlay']):
        mask_rgb = cmap(mask / (num_classes - 1))[..., :3]  # (H, W, 3)
        foreground = mask > 0
        alpha = np.where(foreground, 0.6, 0.0)[..., np.newaxis]  # fade non-background
        blended = flair_rgb * (1 - alpha) + mask_rgb * alpha

        ax.imshow(blended)
        # ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_classes - 1))
    sm.set_array([])
    # fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)

    # plt.suptitle(f'Sample: {sample_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    except:
        print(f"\n Unsaved image: {save_path}")
    plt.close()


def save_prediction_as_nifti(prediction, save_path, reference_nifti=None):
    """
    Save prediction as NIfTI file
    
    Args:
        prediction: Prediction array (H, W) or (H, W, D)
        save_path: Path to save NIfTI file
        reference_nifti: Optional reference NIfTI for header info
    """
    if reference_nifti is not None:
        # Use reference header
        nifti_img = nib.Nifti1Image(prediction.astype(np.uint8), reference_nifti.affine, reference_nifti.header)
    else:
        # Create new NIfTI with identity affine
        nifti_img = nib.Nifti1Image(prediction.astype(np.uint8), np.eye(4))    

    nib.save(nifti_img, save_path)


###################### Post-processing Function ######################

def post_process_pred(pred_classes, num_classes=3, min_object_size=5, closing_kernel_size=2):
    """
    Post-process a single 2-D multi-class prediction slice.

    Input
    -----
    pred_classes      : np.ndarray of shape (H, W) — integer class labels
                        produced by tf.argmax(...).numpy()[0] inside the
                        inference loop (one slice at a time).
    num_classes       : 3  → classes are 0=BG, 1=Vent, 2=AbWMH
                        4  → classes are 0=BG, 1=Vent, 2=NormWMH, 3=AbWMH
    min_object_size   : connected components smaller than this (pixels) are
                        removed after morphological cleaning. Default 5.
    closing_kernel_size: radius of the disk used for binary_closing. Default 2.

    Output
    ------
    post_pred : np.ndarray of shape (H, W), same dtype as pred_classes,
                with cleaned and overlap-resolved integer class labels.

    Processing pipeline (per class)
    --------------------------------
    1. Extract binary mask for each foreground class from the label map.
    2. Apply binary_closing  → fill small holes / bridge tiny gaps.
    3. Apply remove_small_objects → discard isolated noise specks.
    4. Resolve overlaps by anatomical priority:
           Ventricles  >  Normal WMH  >  Abnormal WMH
       (a higher-priority class always wins contested pixels)
    5. Reconstruct the integer label map from the cleaned binary masks.
    """
    from skimage.morphology import remove_small_objects, binary_erosion, binary_closing, disk, binary_dilation

    kernel = disk(closing_kernel_size)

    def clean(mask):
        """Apply closing + small-object removal to a single binary mask."""
        if not mask.any():
            return mask
        mask = binary_closing(mask, kernel)
        # mask = binary_erosion(mask, disk(1))
        mask = remove_small_objects(mask, min_size=min_object_size)
        return mask

    # ── 1. Extract per-class binary masks from the 2-D label map ────────────
    vent_mask  = (pred_classes == 1)

    if num_classes == 4:
        nwmh_mask  = (pred_classes == 2)
        abwmh_mask = (pred_classes == 3)
    else:
        # 3-class scenario: no Normal WMH, AbWMH is class 2
        nwmh_mask  = np.zeros_like(vent_mask)
        abwmh_mask = (pred_classes == 2)

    # ── 2-3. Morphological cleaning per class ───────────────────────────────
    vent_mask  = clean(vent_mask)
    nwmh_mask  = clean(nwmh_mask)
    abwmh_mask = clean(abwmh_mask)

    # ── 4. Resolve overlaps: higher-priority mask wins ───────────────────────
    # Ventricles > Normal WMH > Abnormal WMH
    nwmh_mask  = nwmh_mask  & ~vent_mask   # NormWMH cannot overlap Vent
    abwmh_mask = abwmh_mask & ~vent_mask   # AbWMH   cannot overlap Vent
    abwmh_mask = abwmh_mask & ~nwmh_mask   # AbWMH   cannot overlap NormWMH

    # ── 5. Reconstruct the integer label map ─────────────────────────────────
    post_pred = np.zeros_like(pred_classes)   # background = 0
    post_pred[vent_mask] = 1

    if num_classes == 4:
        post_pred[nwmh_mask]  = 2
        post_pred[abwmh_mask] = 3
    else:
        post_pred[abwmh_mask] = 2

    return post_pred


###################### Main Inference Function ######################

def run_inference(config: InferenceConfig):
    """
    Main inference function
    
    Args:
        config: InferenceConfig object
    
    Returns:
        Dictionary containing all predictions and metrics
    """
    print("\n" + "="*70)
    print(f"RUNNING INFERENCE")
    print("="*70)
    
    # Initialize data loader
    data_config = DataConfig()
    data_loader = P2DataLoader(data_config)
    
    # Load test dataset
    print("Loading test data...")
    test_dataset = data_loader.create_dataset_for_fold(
        fold_id=config.fold_id,
        split='test',
        preprocessing=config.preprocessing,
        class_scenario=config.class_scenario,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Get dataset size
    test_size = tf.data.experimental.cardinality(test_dataset).numpy()
    if test_size < 0:
        test_size = sum(1 for _ in test_dataset)
        test_dataset = data_loader.create_dataset_for_fold(
            fold_id=config.fold_id, split='test',
            preprocessing=config.preprocessing,
            class_scenario=config.class_scenario,
            batch_size=config.batch_size, shuffle=False
        )
    
    print(f"Test samples: {test_size}\n")
    
    # Load model
    print(f"Loading model from: {config.model_path}")
    try:
        if config.architecture_name == 'unet':
            from unet_model import build_unet_3class as build_specific_3class # must be updated with the actual used model for traininig
        elif config.architecture_name == 'attnunet':
            from attn_unet_model import build_attention_unet_3class as build_specific_3class
        elif config.architecture_name == 'dlv3unet':
            from dlv3_unet_model_GN import build_deeplabv3_unet_3class as build_specific_3class
        elif config.architecture_name == 'transunet':
            from trans_unet_model import build_trans_unet_3class as build_specific_3class
        else:
            print(f"❌ Error loading model: Invalid Model Name")
            raise

        # Build model architecture first
        generator = build_specific_3class(
            input_shape=(256, 256, 1), 
            num_classes=config.num_classes
        )
        
        # Load weights
        generator.load_weights(str(config.model_path))
        print("✅ Model loaded successfully\n")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Initialize storage - keyed by patient ID
    patient_results = defaultdict(lambda: {
        'predictions': [],
        'ground_truths': [],
        'probabilities': [],
        'flairs': [],
        'slice_indices': []
    })
    sample_ids = []
        
    # Run inference
    print("Running inference on test set...")
    test_bar = tqdm(test_dataset, total=test_size, desc="Inference")
    
    for idx, (paired_input, target_mask, patient_id_tensor, slice_num_tensor) in enumerate(test_bar):
        
        patient_id = patient_id_tensor.numpy()[0].decode('utf-8')  # batch dim + bytes→str
        slice_num  = int(slice_num_tensor.numpy()[0])

        sample_ids.append(f"{patient_id}_slice_{slice_num:03d}")

        # Prepare input
        flair_normalized = prepare_input(paired_input)
        
        # Generate prediction
        prediction_softmax = generator(flair_normalized, training=False)
        
        # Convert to class labels
        pred_classes = tf.argmax(prediction_softmax, axis=-1).numpy()[0]
        max_prob = tf.reduce_max(prediction_softmax, axis=-1).numpy()[0]
        ground_truth = target_mask.numpy()[0]
        flair = flair_normalized.numpy()[0, :, :, 0]

        # Post-process the predictions
        # pred_classes_post = post_process_pred(pred_classes, num_classes=config.num_classes)

        # Store per-patient
        patient_results[patient_id]['predictions'].append(pred_classes)
        patient_results[patient_id]['ground_truths'].append(ground_truth)
        patient_results[patient_id]['probabilities'].append(max_prob)
        patient_results[patient_id]['flairs'].append(flair)
        patient_results[patient_id]['slice_indices'].append(slice_num)
        
        # Create visualization
        if idx % 10 == 0 or True:  # Visualize every 10th sample
            # viz_path = config.visualizations_dir / f"visualization_{idx:04d}.png"
            viz_path = config.visualizations_dir / f"{sample_ids[-1]}.png"
            visualize_prediction_short(
                flair, ground_truth, pred_classes, 
                max_prob, viz_path,
                sample_ids[-1], config.num_classes
            )
            
    print("\n✅ Inference complete!\n")
        
    # Compute overall metrics
    print("Computing metrics...")
    exclude_class = None
    per_patient_metrics = {}

    for patient_id, data in patient_results.items():
        # Sort slices by anatomical order
        order = np.argsort(data['slice_indices'])
        
        gt_volume   = np.array(data['ground_truths'])[order]    # (S, H, W)
        pred_volume = np.array(data['predictions'])[order]      # (S, H, W)
        
        per_patient_metrics[patient_id] = compute_metrics_from_predictions(
            gt_volume,
            pred_volume,
            config.num_classes
        )
        print(f"\nPatint_id : {patient_id} , Stats: {per_patient_metrics[patient_id]}\n")
        
        pm = per_patient_metrics[patient_id]
        print(f"\nPatient_id: {patient_id}")
        print(f"  Voxel  — Dice: { {k: round(v,4) for k,v in pm['dice'].items()} }")
        if 'lesion' in pm:
            for cls, ld in pm['lesion'].items():
                print(f"  Lesion [{cls}] — "
                    f"GT:{ld['n_gt_lesions']} Pred:{ld['n_pred_lesions']} "
                    f"TP:{ld['tp_lesions']} FP:{ld['fp_lesions']} FN:{ld['fn_lesions']} "
                    f"Sens:{ld['lesion_sensitivity']:.3f} Prec:{ld['lesion_precision']:.3f} "
                    f"F1:{ld['lesion_f1']:.3f}")

    # Aggregate across patients
    overall_metrics, overall_metrics_rich = aggregate_patient_metrics(
        per_patient_metrics, config.num_classes
    )
    # overall_metrics      → drop-in replacement for old overall_metrics, all print/save code unchanged
    # overall_metrics_rich → use wherever we want mean ± std reporting
    
    # Print standard metrics
    print("\n" + "="*70)
    print("STANDARD METRICS (Class vs Rest)")
    print("="*70)
    
    print("\nClass-wise Dice Scores:")
    for class_idx, class_name in enumerate(config.class_names):
        if exclude_class is not None and class_idx == exclude_class:
            continue
        key = f'class_{class_idx}'
        if key in overall_metrics['dice']:
            print(f"  {class_name}: {overall_metrics['dice'][key]:.4f}")
    print(f"  Mean Dice: {overall_metrics['dice']['mean']:.4f}")
    
    print("\nClass-wise Precision:")
    for class_idx, class_name in enumerate(config.class_names):
        if exclude_class is not None and class_idx == exclude_class:
            continue
        key = f'class_{class_idx}'
        if key in overall_metrics['precision']:
            print(f"  {class_name}: {overall_metrics['precision'][key]:.4f}")
    print(f"  Mean Precision: {overall_metrics['precision']['mean']:.4f}")
    
    print("\nClass-wise Recall:")
    for class_idx, class_name in enumerate(config.class_names):
        if exclude_class is not None and class_idx == exclude_class:
            continue
        key = f'class_{class_idx}'
        if key in overall_metrics['recall']:
            print(f"  {class_name}: {overall_metrics['recall'][key]:.4f}")
    print(f"  Mean Recall: {overall_metrics['recall']['mean']:.4f}")
    
    print("\nClass-wise IoU:")
    for class_idx, class_name in enumerate(config.class_names):
        if exclude_class is not None and class_idx == exclude_class:
            continue
        key = f'class_{class_idx}'
        if key in overall_metrics['iou']:
            print(f"  {class_name}: {overall_metrics['iou'][key]:.4f}")
    print(f"  Mean IoU: {overall_metrics['iou']['mean']:.4f}")

    print("\nClass-wise Specificity:")
    for class_idx, class_name in enumerate(config.class_names):
        if exclude_class is not None and class_idx == exclude_class:
            continue
        key = f'class_{class_idx}'
        if key in overall_metrics['specificity']:
            print(f"  {class_name}: {overall_metrics['specificity'][key]:.4f}")
    print(f"  Mean Specificity: {overall_metrics['specificity']['mean']:.4f}")
    
    print("\nClass-wise HD95 (lower is better):")
    for class_idx, class_name in enumerate(config.class_names):
        if exclude_class is not None and class_idx == exclude_class:
            continue
        key = f'class_{class_idx}'
        if key in overall_metrics['hd95']:
            print(f"  {class_name}: {overall_metrics['hd95'][key]:.4f}")
    print(f"  Mean HD95: {overall_metrics['hd95']['mean']:.4f}")

    print("="*70 + "\n")

    # Print lesion-level metrics
    print("\n" + "="*70)
    print("LESION-LEVEL METRICS (Connected-Component Analysis)")
    print("="*70)

    for class_idx, class_name in enumerate(config.class_names):
        if class_idx == 0:
            continue
        key = f'class_{class_idx}'
        if key not in overall_metrics.get('lesion', {}):
            continue
        ld = overall_metrics['lesion'][key]
        print(f"\n  [{class_name}]")
        print(f"    GT Lesions          : {ld['n_gt_lesions']}")
        print(f"    Predicted Lesions   : {ld['n_pred_lesions']}")
        print(f"    TP Lesions          : {ld['tp_lesions']}")
        print(f"    FP Lesions          : {ld['fp_lesions']}")
        print(f"    FN Lesions          : {ld['fn_lesions']}")
        print(f"    Lesion Sensitivity  : {ld['lesion_sensitivity']:.4f}")
        print(f"    Lesion Precision    : {ld['lesion_precision']:.4f}")
        print(f"    Lesion F1           : {ld['lesion_f1']:.4f}")

    print(f"\n  [Summary across foreground classes]")
    print(f"    Total GT Lesions     : {overall_metrics['lesion']['total_n_gt_lesions']}")
    print(f"    Total Pred Lesions   : {overall_metrics['lesion']['total_n_pred_lesions']}")
    print(f"    Total TP Lesions     : {overall_metrics['lesion']['total_tp_lesions']}")
    print(f"    Total FP Lesions     : {overall_metrics['lesion']['total_fp_lesions']}")
    print(f"    Total FN Lesions     : {overall_metrics['lesion']['total_fn_lesions']}")
    print(f"    Mean Lesion Sensitivity : {overall_metrics['lesion']['mean_lesion_sensitivity']:.4f}")
    print(f"    Mean Lesion Precision   : {overall_metrics['lesion']['mean_lesion_precision']:.4f}")
    print(f"    Mean Lesion F1          : {overall_metrics['lesion']['mean_lesion_f1']:.4f}")
    print("="*70 + "\n")
    
    # Save all metrics to JSON
    metrics_file = config.metrics_dir / "test_metrics_complete.json"
    
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    metrics_to_save = {
        'config': {
            'variant': int(config.variant),
            'preprocessing': config.preprocessing,
            'class_scenario': config.class_scenario,
            'fold_id': int(config.fold_id),
            'num_classes': int(config.num_classes),
            'class_names': config.class_names,
            'architecture_name': config.architecture_name,
            'model_name': config.model_name,
            'test_samples': int(test_size)
        },
        'metrics': convert_to_serializable(overall_metrics)
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"\n✅ All metrics saved to: {metrics_file}")
    # print(f"✅ Predictions saved to: {config.predictions_dir}")
    print(f"✅ Visualizations saved to: {config.visualizations_dir}")
    
    # Return results
    return {
        'patients_results': patient_results,
        'metrics': overall_metrics,
        'rich_metrics': overall_metrics_rich
    }


###################### Main Execution ######################

if __name__ == "__main__":
    # Run inference
    
    preprocess_options = ['standard']  # ['zoomed', 'standard']
    scenarios          = ['3class']  # ['3class', '4class']
    fold_numbers       = list(np.array([0, 1, 2, 3]))

    for fold_number in fold_numbers: 
        for preprocess_option in preprocess_options:
            for scenario in scenarios:

                config = InferenceConfig(
                    variant=1,
                    preprocessing=preprocess_option,
                    class_scenario=scenario,
                    fold_id=fold_number,
                    model_name='best_dice_model.h5',
                    architecture_name='unet'   # a choice from ['unet', 'attnunet', 'dlv3unet', 'transunet']
                )
                
                results = run_inference(config)

                # ── Error Analysis ──────────────────────────────────────
                error_results = run_error_analysis(
                    results=results,
                    config=config,
                    top_n_slices=300,      # visualise N hardest slices
                    top_n_patients=20,    # patient summary plots
                    fg_dice_weight=0.7,   # tunable ranking weights
                    error_rate_weight=0.2,
                    confidence_weight=0.2,
                )
                # ────────────────────────────────────────────────────────

                print("\n" + "="*70)
                print("INFERENCE + ERROR ANALYSIS COMPLETE")
                print("="*70)
