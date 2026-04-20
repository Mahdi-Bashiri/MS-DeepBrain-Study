"""
P4 - All U-Net models with Adaptive Loss (WCE + UFL)

WMH and Ventricles Segmentation with U-Net Models - Journal Paper Implementation
Three-class segmentation: Background vs Ventricles vs Abnormal WMH
Professional results saving and visualization for publication

This relates to our article:
"Deep Learning-Based Neuroanatomical Profiling Reveals Detailed Brain Changes:
A Large-Scale Multiple Sclerosis Study"

Features:
- Various U-Net architecture
- Weighted Categorical Cross-Entropy loss
- Unified Focal loss
- One-hot encoded targets
- Class weight computation per fold

Authors:
"Mahdi Bashiri Bawil, Mousa Shamsi, Abolhassan Shakeri Bavil"

Developer:
"Mahdi Bashiri Bawil"
"""

import tensorflow as tf
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# Import data loader
from p4_data_loader import DataConfig, P2DataLoader

# Import utilities from baseline
from utility_functions import (
    clear_gpu_memory,
    get_gpu_memory_info,
)

# Import class weights utility
from p4_compute_class_weights import compute_and_save_class_weights, load_class_weights

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
    print("⚠️  No GPU detected - training will be slow")

"""
GPU Memory Management for Sequential Experiments
To properly release memory between experiments
"""

###################### Target Preparation ######################

def prepare_inputs(paired_input, target_mask, num_classes):
    """
    Prepare inputs for training
    
    Args:
        paired_input: (bs, 256, 512, 1) with FLAIR + mask
        target_mask: (bs, 256, 256) with class labels [0, num_classes-1]
        num_classes: number of classes
        
    Returns:
        flair_normalized: FLAIR normalized to [-1, 1]
        target_onehot: One-hot encoded mask (bs, 256, 256, num_classes)
    """
    # Extract FLAIR, previously normalized to [-1, 1]
    flair_normalized = paired_input[:, :, :256, :]

    # One-hot encode target
    target_onehot = tf.one_hot(target_mask, depth=num_classes, dtype=tf.float32)
    
    return flair_normalized, target_onehot

###################### Metrics Calculation ######################

def compute_classwise_metrics(all_val_true, all_val_pred, num_classes, exclude_class=None):
    """
    Compute class-wise Dice, Precision, and Recall for validation predictions.
    
    Args:
        all_val_true: List of one-hot encoded ground truth tensors
        all_val_pred: List of softmax output tensors from model
        num_classes: Number of classes (3 or 4)
        exclude_class: Class to exclude from metric calculation (e.g., 2 for background)
    
    Returns:
        Dictionary containing class-wise and mean metrics
    """
    # Concatenate all batches
    y_true_concat = tf.concat(all_val_true, axis=0)  # Shape: (N, H, W, num_classes)
    y_pred_concat = tf.concat(all_val_pred, axis=0)  # Shape: (N, H, W, num_classes)
    
    # Flatten spatial dimensions: (N*H*W, num_classes)
    y_true_flat = tf.reshape(y_true_concat, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred_concat, [-1, num_classes])
    
    # Convert predictions to one-hot (argmax)
    y_pred_classes = tf.argmax(y_pred_flat, axis=-1)
    y_pred_onehot = tf.one_hot(y_pred_classes, depth=num_classes)
    
    # Convert to numpy for easier computation
    y_true_np = y_true_flat.numpy()
    y_pred_np = y_pred_onehot.numpy()
    
    metrics = {
        'dice': {},
        'precision': {},
        'recall': {}
    }
    
    classes_to_evaluate = [c for c in range(num_classes) if c != exclude_class]
    
    for class_idx in classes_to_evaluate:
        # Extract binary masks for this class
        true_class = y_true_np[:, class_idx]
        pred_class = y_pred_np[:, class_idx]
        
        # True Positives, False Positives, False Negatives
        TP = np.sum((true_class == 1) & (pred_class == 1))
        FP = np.sum((true_class == 0) & (pred_class == 1))
        FN = np.sum((true_class == 1) & (pred_class == 0))
        
        # Dice Score: 2*TP / (2*TP + FP + FN)
        dice = (2 * TP) / (2 * TP + FP + FN + 1e-7)
        
        # Precision: TP / (TP + FP)
        precision = TP / (TP + FP + 1e-7)
        
        # Recall (Sensitivity): TP / (TP + FN)
        recall = TP / (TP + FN + 1e-7)
        
        metrics['dice'][f'class_{class_idx}'] = float(dice)
        metrics['precision'][f'class_{class_idx}'] = float(precision)
        metrics['recall'][f'class_{class_idx}'] = float(recall)
    
    # Compute mean metrics (excluding the excluded class)
    metrics['dice']['mean'] = np.mean([v for v in metrics['dice'].values()])
    metrics['precision']['mean'] = np.mean([v for v in metrics['precision'].values()])
    metrics['recall']['mean'] = np.mean([v for v in metrics['recall'].values()])
    
    return metrics

###################### Experiment Configuration ######################

class ExperimentConfig:
    """Configuration for a Specific U-Net experiment"""
    
    def __init__(self, 
                 variant: int = 1,
                 preprocessing: str = 'standard',
                 class_scenario: str = '3class',
                 fold_id: int = 0,
                 architecture_name: str = 'unet'
                 ):
        
        # Experiment identification
        self.variant = variant
        self.preprocessing = preprocessing  # 'standard' or 'zoomed'
        self.class_scenario = class_scenario  # '3class' or '4class'
        self.fold_id = fold_id
        self.architecture_name = architecture_name
        
        # Experiment name
        self.exp_name = f"exp_{architecture_name}_{preprocessing}_{class_scenario}_fold{fold_id}"
        
        # Number of classes
        self.num_classes = 3 if class_scenario == '3class' else 4
        
        # Training hyperparameters
        self.batch_size = 4
        self.img_width = 256
        self.img_height = 256
        self.epochs = 60
        
        # Optimizer parameters
        self.learning_rate = 2e-4
        self.beta_1 = 0.9

        # Adaptive loss parameters
        self.focal_gamma = 0.5           # Focal loss focusing parameter
        self.beta_threshold = 0.25       # Transition at epoch 15/60
        self.beta_smoothness = 0.02      # Transition width
        self.use_focal_alpha = True      # Use class weights in focal loss

        # ReduceLROnPlateau parameters
        self.lr_patience = 5          # Wait 5 epochs before reducing
        self.lr_reduction_factor = 0.5  # Reduce LR by half
        self.lr_min = 1e-7            # Don't go below this
        self.lr_monitor = 'val_loss'  # Or 'val_dice_mean'
        
        # Paths
        self.results_dir = Path(f"results_fold_{fold_id}_var_{variant}_zscore3")
        self.models_dir = self.results_dir / "models" / f"{preprocessing}_{class_scenario}"
        self.figures_dir = self.results_dir / "figures" / f"{preprocessing}_{class_scenario}" / f"fold_{fold_id}"
        self.logs_dir = self.results_dir / "logs" / f"{preprocessing}_{class_scenario}" / f"fold_{fold_id}"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint configuration
        self.checkpoint_dir = self.models_dir / f"fold_{fold_id}"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Class weights directory
        self.weights_dir = Path("class_weights")
        self.weights_dir.mkdir(exist_ok=True)

        # Save configuration
        self.save_config()
    
    def save_config(self):
        """Save experiment configuration to JSON"""
        config_dict = {
            'variant': self.variant,
            'variant_name': f'{self.architecture_name}',
            'preprocessing': self.preprocessing,
            'class_scenario': self.class_scenario,
            'fold_id': self.fold_id,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'focal_gamma': self.focal_gamma,
            'beta_threshold': self.beta_threshold,
            'beta_smoothness': self.beta_smoothness,
            'learning_rate': self.learning_rate,
            'beta_1': self.beta_1,
            'loss': 'Phase-transitioning segmentation loss (WCE → UFD)'
        }
        
        config_file = self.checkpoint_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)


###################### Beta Scheduling ######################

def smooth_step(x, threshold=0.5, smoothness=0.1):
    """
    Smooth step function for phase transition
    
    Creates smooth transition around threshold value using sigmoid.
    
    Args:
        x: Current progress (typically epoch / total_epochs)
        threshold: Center point of transition (e.g., 0.5 for epoch 25/50)
        smoothness: Width of transition (smaller = sharper, larger = smoother)
        
    Returns:
        Value in [0, 1] representing transition progress
        - x << threshold: returns ≈ 0
        - x ≈ threshold: returns ≈ 0.5
        - x >> threshold: returns ≈ 1
        
    Example:
        epoch_progress = 0.3  # Epoch 15/50
        beta = smooth_step(0.3, threshold=0.5, smoothness=0.1)
        # beta ≈ 0.05 (mostly phase 1)
        
        epoch_progress = 0.5  # Epoch 25/50
        beta = smooth_step(0.5, threshold=0.5, smoothness=0.1)
        # beta ≈ 0.5 (equal mix)
        
        epoch_progress = 0.7  # Epoch 35/50
        beta = smooth_step(0.7, threshold=0.5, smoothness=0.1)
        # beta ≈ 0.95 (mostly phase 2)
    """
    # Sigmoid centered at threshold
    # (x - threshold) / smoothness controls steepness
    return tf.sigmoid((x - threshold) / smoothness)


def compute_beta_schedule(current_epoch, total_epochs, 
                          threshold=0.5, smoothness=0.1):
    """
    Compute beta value for current epoch
    
    Args:
        current_epoch: Current epoch number (0-indexed)
        total_epochs: Total number of epochs
        threshold: Transition center (0.5 = midpoint)
        smoothness: Transition width
        
    Returns:
        Beta value in [0, 1]
    """
    epoch_progress = tf.cast(current_epoch, tf.float32) / tf.cast(total_epochs, tf.float32)
    beta = smooth_step(epoch_progress, threshold, smoothness)
    return beta

###################### Loss Functions ######################

def unified_focal_loss(y_true, y_pred, gamma=2.0, alpha=None, exclude_class=None):
    """
    Unified Focal Loss
    
    Focal loss down-weights easy examples and focuses on hard examples.
    Particularly effective for class imbalance and boundary regions.
    
    Args:
        y_true: Ground truth labels (bs, H, W, num_classes) one-hot encoded
        y_pred: Predicted probabilities (bs, H, W, num_classes) from softmax
        gamma: Focusing parameter (default 2.0)
            - gamma=0: equivalent to cross-entropy
            - gamma>0: down-weights easy examples
            - Higher gamma = more focus on hard examples
        alpha: Per-class balancing weights (num_classes,) - optional, trainable
            - If None, no additional balancing
            - If provided, applies per-class weighting like weighted CE
            
    Returns:
        Scalar loss value
        
    Formula:
        FL = -α * (1 - p_t)^γ * log(p_t)
        where:
        - p_t is probability of correct class
        - (1 - p_t)^γ is modulating factor (focal term)
        - α is class balancing weight
    """
    # Clip predictions to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Probability of correct class at each pixel
    # y_true is one-hot, so this extracts p for the true class
    p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
    # Shape: (bs, H, W)
    
    # Focal term: (1 - p_t)^gamma
    # This is small for easy examples (p_t ≈ 1) and large for hard examples (p_t ≈ 0)
    focal_term = tf.pow(1.0 - p_t, gamma)
    # Shape: (bs, H, W)
    
    # Cross-entropy term: -log(p_t)
    ce_term = -tf.math.log(p_t)
    # Shape: (bs, H, W)
    
    # Focal loss: focal_term * ce_term
    focal_loss = focal_term * ce_term
    # Shape: (bs, H, W)
    
    # Optional: Apply alpha balancing (per-class weights)
    if alpha is not None:
        # Get weight for true class at each pixel
        weights_tensor = tf.cast(alpha, dtype=tf.float32)
        weights_tensor = tf.reshape(weights_tensor, [1, 1, 1, -1])
        alpha_map = tf.reduce_sum(y_true * weights_tensor, axis=-1)
        # Shape: (bs, H, W)
        
    # Weighted focal
    # Exclude specific class if specified
    if exclude_class is not None:
        class_mask = tf.argmax(y_true, axis=-1)  # (bs, 256, 256)
        valid_mask = tf.cast(class_mask != exclude_class, tf.float32)

        if alpha is not None:
            focal_loss = alpha_map * focal_loss * valid_mask
        else:
            focal_loss = focal_loss * valid_mask

        return tf.reduce_sum(focal_loss) / (tf.reduce_sum(valid_mask) + 1e-7)
    else:
        
        if alpha is not None:
            focal_loss = alpha_map * focal_loss

        return tf.reduce_mean(focal_loss)


def unified_focal_dice_loss(y_true, y_pred, gamma=0.5, delta=0.6, alpha=None, exclude_class=None):
    """
    Unified Focal Loss - Dice-based
    
    Combines Dice coefficient with precision-recall focal weighting.
    Best for imbalanced multi-class segmentation with small structures.
    
    Args:
        y_true: Ground truth one-hot (bs, H, W, num_classes)
        y_pred: Predicted probabilities (bs, H, W, num_classes)
        gamma: Focusing parameter for Dice component (default 0.5)
               - gamma=0: equivalent to Dice loss
               - gamma>0: focuses on hard examples
        delta: Weight for precision-recall component (0-1, default 0.6)
               - Controls emphasis on boundary regions
        alpha: Per-class weights (num_classes,) - optional
        exclude_class: Class index to exclude from loss
    
    Returns:
        Scalar loss value
        
    Formula:
        UFL = (1 - Dice)^gamma * (1 - precision * recall)^delta
        Focuses on hard examples and boundary regions
    """
    smooth = 1e-6
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    num_classes = tf.shape(y_pred)[-1]
    
    unified_losses = []
    
    for class_idx in range(num_classes if isinstance(num_classes, int) else y_pred.shape[-1]):
        # Skip excluded class
        if exclude_class is not None and class_idx == exclude_class:
            continue

        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        
        # Flatten for calculations
        y_true_f = tf.reshape(y_true_class, [-1])
        y_pred_f = tf.reshape(y_pred_class, [-1])
        
        # True positives, false positives, false negatives
        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
        fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
        
        # Precision and recall
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        
        # Dice coefficient
        dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
        
        # Unified focal loss: focuses on hard examples and boundary regions
        # (1 - dice)^gamma: focuses on classes with low Dice (hard examples)
        # (1 - precision * recall)^delta: focuses on boundary regions
        unified_loss_class = tf.pow(1.0 - dice, gamma) * tf.pow(1.0 - precision * recall, delta)
        
        # Apply class weights
        if alpha is not None:
            unified_loss_class = unified_loss_class * tf.cast(alpha[class_idx], tf.float32)
        
        unified_losses.append(unified_loss_class)
    
    # Stack and mean across classes (excluding the skipped class)
    total_loss = tf.reduce_mean(tf.stack(unified_losses))
    
    return total_loss


def weighted_categorical_crossentropy(y_true, y_pred, class_weights, exclude_class=None):
    """
    Weighted categorical cross-entropy loss
    
    Args:
        y_true: (bs, 256, 256, num_classes) one-hot encoded
        y_pred: (bs, 256, 256, num_classes) softmax probabilities
        class_weights: (num_classes,) weight per class
        exclude_class: Optional int, class index to exclude from loss (e.g., 2 for CSF)
    
    Returns:
        Scalar loss value
    """
    # Clip predictions to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Cross-entropy per pixel: -sum(y_true * log(y_pred))
    ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)  # (bs, 256, 256)
    
    # Apply class weights
    # class_weights shape: (num_classes,) -> (1, 1, 1, num_classes) for broadcasting
    weights_tensor = tf.cast(class_weights, dtype=tf.float32)
    weights_tensor = tf.reshape(weights_tensor, [1, 1, 1, -1])
    
    # Weight map: (bs, 256, 256)
    pixel_weights = tf.reduce_sum(y_true * weights_tensor, axis=-1)
    
    # Weighted cross-entropy
    # Exclude specific class if specified
    if exclude_class is not None:
        class_mask = tf.argmax(y_true, axis=-1)  # (bs, 256, 256)
        valid_mask = tf.cast(class_mask != exclude_class, tf.float32)
        weighted_ce = ce * pixel_weights * valid_mask
        return tf.reduce_sum(weighted_ce) / (tf.reduce_sum(valid_mask) + 1e-7)
    else:
        weighted_ce = ce * pixel_weights
        return tf.reduce_mean(weighted_ce)


def adaptive_segmentation_loss(y_true, y_pred, class_weights, beta, 
                               focal_gamma=0.5, use_focal_alpha=True,
                               exclude_class=None):
    """
    Adaptive segmentation loss with hard phase transition
    
    Combines weighted cross-entropy (phase 1) and focal loss (phase 2)
    based on epoch progress (beta).
    
    Args:
        y_true: Ground truth (bs, H, W, num_classes) one-hot
        y_pred: Predictions (bs, H, W, num_classes) softmax probabilities
        class_weights: Trainable class weights (num_classes,)
        beta: Transition parameter [0, 1]
            - beta=0: pure weighted CE (early training)
            - beta=1: pure focal loss (late training)
        focal_gamma: Focusing parameter for focal loss (default 0.5)
        use_focal_alpha: Whether to use class_weights as focal alpha
        
    Returns:
        seg_loss: Final loss
        wcce_loss: Weighted CE component (for monitoring)
        focal_loss: Focal loss component (for monitoring)
        
    Phase Behavior:
        Epochs 1-10: beta ≈ 0 → Weighted CE dominates
            - Learns basic class separation
            - Benefits from explicit class weighting
        
        Epochs 10-20: beta transitions 0 → 1
            - Smooth change in loss landscape
            - Gradual shift in training dynamics
        
        Epochs 20-60: beta ≈ 1 → Focal loss dominates
            - Focuses on hard examples
            - Refines boundaries and difficult regions
    """
    # Compute Phase 1 loss: Weighted Cross-Entropy
    wcce_loss = 10 * weighted_categorical_crossentropy(y_true, y_pred, class_weights, exclude_class=exclude_class)

    # Compute Phase 2 loss: Focal Loss
    focal_alpha = class_weights if use_focal_alpha else None
    focal_loss = unified_focal_dice_loss(y_true, y_pred, 
                                       gamma=focal_gamma, 
                                       alpha=focal_alpha,
                                       exclude_class=exclude_class)
    
    # Adaptive combination based on beta
    # beta=0: (1-0)*wce + 0*focal = wce (phase 1)
    # beta=1: (1-1)*wce + 1*focal = focal (phase 2)
    # beta=0.5: 0.5*wce + 0.5*focal = equal mix (transition)
    seg_loss = (1.0 - beta) * wcce_loss + beta * focal_loss
    
    return seg_loss, wcce_loss, focal_loss

###################### Training Functions ######################

@tf.function
def train_step(input_image, target_onehot, model, optimizer, 
               class_weights, beta, focal_gamma, 
               use_focal_alpha=True, exclude_class=None):
    """
    Single training step for U-Net
    
    Args:
        input_image: Input FLAIR (bs, 256, 256, 1) in [-1, 1]
        target_onehot: Target mask (bs, 256, 256, num_classes) one-hot
        model: a specific U-Net model
        optimizer: Optimizer
        class_weights: (num_classes,) weight per class
        beta: Current beta for phase transition

    
    Returns:
        loss: Training loss value
    """
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(input_image, training=True)
        
        # Compute loss
        seg_loss, wcce_loss, focal_loss = adaptive_segmentation_loss(target_onehot, predictions, class_weights, 
                                                                     beta, focal_gamma, use_focal_alpha, exclude_class)
    
    # Calculate gradients
    gradients = tape.gradient(seg_loss, model.trainable_variables)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return seg_loss, wcce_loss, focal_loss

def generate_and_save_images(model, test_input, test_target, 
                            epoch, save_path, num_classes):
    """
    Generate predictions and save visualization
    
    Args:
        model: a specific U-Net model
        test_input: Test input image (bs, 256, 512, 1)
        test_target: Test target mask (bs, 256, 256)
        epoch: Current epoch number
        save_path: Path to save figure
        num_classes: Number of classes
    """
    for ik in range(test_input.numpy().shape[0]):
        # Extract FLAIR
        flair_normalized = test_input[ik, :, :256, :]
        flair_normalized = tf.expand_dims(flair_normalized, axis=0)
        
        # Generate prediction
        prediction_softmax = model(flair_normalized, training=False)
        
        # Convert to class labels
        pred_classes = tf.argmax(prediction_softmax, axis=-1).numpy()
        target_mask = test_target[ik].numpy()
        
        # Create figure
        plt.figure(figsize=(20, 5))
        
        # Input FLAIR
        plt.subplot(1, 5, 1)
        plt.title('Input FLAIR')
        plt.imshow(flair_normalized[0, :, :, 0], cmap='gray')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(1, 5, 2)
        plt.title('Ground Truth')
        plt.imshow(target_mask, cmap='jet', vmin=0, vmax=num_classes-1)
        plt.colorbar()
        plt.axis('off')
        
        # Prediction
        plt.subplot(1, 5, 3)
        plt.title('Predicted Classes')
        plt.imshow(pred_classes[0], cmap='jet', vmin=0, vmax=num_classes-1)
        plt.colorbar()
        plt.axis('off')
        
        # Class probabilities for most confident prediction
        plt.subplot(1, 5, 4)
        plt.title('Max Probability')
        max_prob = tf.reduce_max(prediction_softmax[0], axis=-1).numpy()
        plt.imshow(max_prob, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')
        
        # Difference map
        plt.subplot(1, 5, 5)
        plt.title('Error Map (Red=Wrong)')
        error_map = (pred_classes[0] != target_mask).astype(float)
        plt.imshow(error_map, cmap='Reds', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / f'epoch_{epoch:03d}_{ik+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

###################### Main Training Function ######################

def train_net(config: ExperimentConfig):
    """
    Main training function for a Specific U-Net
    
    Args:
        config: ExperimentConfig object
    """
    print("\n" + "="*70)
    print(f"TRAINING {config.architecture_name}: {config.exp_name}")
    print("="*70)
    print(f"Variant: {config.variant}")
    print(f"Preprocessing: {config.preprocessing}")
    print(f"Class scenario: {config.class_scenario} ({config.num_classes} classes)")
    print(f"Fold: {config.fold_id}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Loss: Weighted Categorical Cross-Entropy → Unified Focal")
    print("="*70 + "\n")

    # Check initial GPU memory
    get_gpu_memory_info()
    
    # Initialize data loader
    data_config = DataConfig()
    data_loader = P2DataLoader(data_config)
    
    # Load datasets
    print("Loading training data...")
    train_dataset = data_loader.create_dataset_for_fold(
        fold_id=config.fold_id,
        split='train',
        preprocessing=config.preprocessing,
        class_scenario=config.class_scenario,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    print("Loading validation data...")
    val_dataset = data_loader.create_dataset_for_fold(
        fold_id=config.fold_id,
        split='val',
        preprocessing=config.preprocessing,
        class_scenario=config.class_scenario,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Get dataset sizes
    # Note: from_generator pipelines always report cardinality as INFINITE (-1)
    # even with .cache(), so we derive the batch count from the slice list instead.
    # We iterate once here; this also warms the in-memory cache so epoch 1 is fast.
    print("Warming dataset cache (first pass over data — subsequent epochs use RAM)...")
    train_size = sum(1 for _ in train_dataset)
    val_size   = sum(1 for _ in val_dataset)
    # ⚠️  Do NOT rebuild the datasets here — that would create new generators and
    #     throw away the cache we just populated.
    
    print(f"Training samples (batches): {train_size}")
    print(f"Validation samples (batches): {val_size}\n")
    
    # Compute or load class weights
    print("Computing class weights from training data...")
    try:
        class_weights = load_class_weights(
            config.fold_id, config.class_scenario, 
            config.preprocessing, config.weights_dir
        )
        print("✅ Loaded pre-computed class weights")
    except FileNotFoundError:
        print("Computing class weights (this may take a few minutes)...")
        results = compute_and_save_class_weights(
            config.fold_id, config.class_scenario, 
            config.preprocessing, str(config.weights_dir)
        )
        class_weights = np.array(results['class_weights'], dtype=np.float32)
    
    print(f"Class weights: {class_weights}")

    # Build model
    print(f"\n🏗️  Building {config.architecture_name} model...")
    
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

    model = build_specific_3class(input_shape=(256, 256, 1), num_classes=config.num_classes)

    print(f"Model parameters: {model.count_params():,}\n")
        
    # Optimizer (will be updated with ReduceLROnPlateau)
    optimizer = tf.keras.optimizers.legacy.Adam(
        config.learning_rate, beta_1=config.beta_1
    )

    # Initialize optimizer variables
    print("Initializing optimizer variables...")
    dummy_input = tf.zeros((1, 256, 256, 1))
    
    with tf.GradientTape() as tape:
        output = model(dummy_input, training=True)
        dummy_loss = tf.reduce_mean(output)

    # Apply dummy gradients to build optimizer variables
    grads = tape.gradient(dummy_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("✅ Optimizer variables initialized\n")
    
    # Checkpoint
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model
    )
    
    checkpoint_prefix = config.checkpoint_dir / "ckpt"
    manager = tf.train.CheckpointManager(
        checkpoint, config.checkpoint_dir, max_to_keep=1
    )
    
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"✅ Restored from checkpoint: {manager.latest_checkpoint}\n")
    else:
        print("Starting training from scratch\n")
    
    # Get example for visualization
    skip_n = 1 # min(100 // config.batch_size, val_size - 1)
    example_paired, example_target, _, _ = next(iter(val_dataset.skip(skip_n).take(20)))
    
    print("Initializing metrics computer...")
    if config.num_classes == 4:
        class_names = ['Background', 'Ventricles', 'Normal_WMH', 'Abnormal_WMH']
    elif config.num_classes == 3:
        class_names = ['Background', 'Ventricles', 'Abnormal_WMH']

    # Training history
    history = {
        'train_loss': [],
        'wce_loss': [],
        'ufd_loss': [],
        'val_loss': [],
        'val_loss_wce': [],
        'val_loss_ufd': [],
        'val_metrics': [],
        'beta_value': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    best_val_dice = float('-inf')
    exclude_class = 2 if config.num_classes == 4 else None  # Exclude class 2 only in 4-class

    try:
        for epoch in range(config.epochs):
            start_time = time.time()

            # Compute beta for this epoch
            beta_value = compute_beta_schedule(
                epoch, config.epochs, 
                config.beta_threshold, config.beta_smoothness
            )
            
            # Training metrics
            epoch_losses = []
            epoch_loss_wce = []
            epoch_loss_ufd = []
            
            # Training loop

            # Update learning rate based on epoch
            
            # y1 = 2 * np.exp(-np.log(400) * x)       # original
            # y2 = 2 * np.exp(-np.log(400) * x**2)    # milder
            # y3 = 2 * np.exp(-np.log(400) * x**3)    # even milder ✅
            # y4 = 2 * np.exp(-np.log(400) * x**5)    # very mild
            
            new_lr = config.learning_rate * np.exp(-np.log(400) * (epoch / config.epochs)**3)  # Steadily and exponentially decay from 2e-4 to 5e-7
            optimizer.learning_rate.assign(new_lr)
            
            print(f"\nEpoch {epoch+1}/{config.epochs} (β={beta_value.numpy():.4f}) (lr={new_lr*10000:.3f} 10-4)")
            train_bar = tqdm(train_dataset, total=train_size, desc="Training")

            for paired_input, target_mask, patient_id_tensor, slice_num_tensor in train_bar:
                
                patient_id = patient_id_tensor.numpy()[0].decode('utf-8')  # batch dim + bytes→str
                slice_num  = int(slice_num_tensor.numpy()[0])

                # ✅ Prepare inputs: normalize FLAIR + one-hot encode target
                flair_normalized, target_onehot = prepare_inputs(
                    paired_input, target_mask, config.num_classes
                )
                
                # Train step
                loss, wce_loss, ufd_loss = train_step(
                    flair_normalized, target_onehot,
                    model, optimizer, class_weights,
                    beta_value, config.focal_gamma
                )
                
                epoch_losses.append(loss.numpy())
                epoch_loss_wce.append(wce_loss.numpy())
                epoch_loss_ufd.append(ufd_loss.numpy())
                
                # Update progress bar
                train_bar.set_postfix({
                    'seg_loss': f"{loss.numpy():.5f}",
                    'wce_loss': f"{wce_loss.numpy():.5f}",
                    'ufd_loss': f"{ufd_loss.numpy():.5f}",
                })
            
            # Calculate epoch average
            avg_train_loss = np.mean(epoch_losses)
            avg_train_loss_wce = np.mean(epoch_loss_wce)
            avg_train_loss_ufd = np.mean(epoch_loss_ufd)

            history['train_loss'].append(avg_train_loss)
            history['wce_loss'].append(avg_train_loss_wce)
            history['ufd_loss'].append(avg_train_loss_ufd)
            history['beta_value'].append(float(beta_value.numpy()))

            # Validation
            val_losses = []
            val_losses_wce = []
            val_losses_ufd = []
            all_val_true = []
            all_val_pred = []
            
            for val_paired, val_target, patient_id_tensor, slice_num_tensor in val_dataset:
                try:

                    patient_id = patient_id_tensor.numpy()[0].decode('utf-8')  # batch dim + bytes→str
                    slice_num  = int(slice_num_tensor.numpy()[0])

                    val_flair_norm, val_target_onehot = prepare_inputs(
                        val_paired, val_target, config.num_classes
                    )

                    val_pred = model(val_flair_norm, training=False)

                    val_loss, val_wce_loss, val_ufd_loss = adaptive_segmentation_loss(
                        val_target_onehot, val_pred, class_weights, 
                        beta_value, focal_gamma=config.focal_gamma, exclude_class=exclude_class
                    )

                    # Store true and prediction values for metrics calculation
                    all_val_true.append(val_target_onehot)
                    all_val_pred.append(val_pred)
                    
                    if not tf.math.is_nan(val_loss):
                        val_losses.append(val_loss.numpy())
                        val_losses_wce.append(val_wce_loss.numpy())
                        val_losses_ufd.append(val_ufd_loss.numpy())
                except:
                    continue
            
            if len(val_losses) > 0:
                avg_val_loss = np.mean(val_losses)
                avg_val_loss_wce = np.mean(val_losses_wce)
                avg_val_loss_ufd = np.mean(val_losses_ufd)

                history['val_loss'].append(avg_val_loss)
                history['val_loss_wce'].append(avg_val_loss_wce)
                history['val_loss_ufd'].append(avg_val_loss_ufd)
                
                # Compute class-wise metrics
                val_metrics = compute_classwise_metrics(
                    all_val_true, all_val_pred, 
                    config.num_classes#, exclude_class=exclude_class
                )
                history['val_metrics'].append(val_metrics)
                
                # Print validation results
                epoch_time = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"Epoch {epoch+1}/{config.epochs} Summary (Time: {epoch_time:.2f}s)")
                print(f"{'='*70}")
                print(f"Training Loss: {avg_train_loss:.4f} | wce_loss: {avg_train_loss_wce:.4f} | ufd_loss: {avg_train_loss_ufd:.4f}")
                print(f"Validation Loss: {avg_val_loss:.4f}")
                print(f"\nClass-wise Dice Scores:")
                for class_name, dice_val in val_metrics['dice'].items():
                    if class_name != 'mean':
                        print(f"  {class_name}: {dice_val:.4f}")
                        if class_name == f"class_{config.num_classes - 1}":
                            abwmh_val_dice = dice_val
                        elif class_name == f"class_1":
                            vent_val_dice = dice_val
                print(f"  Mean Dice: {val_metrics['dice']['mean']:.4f}")
                print(f"\nClass-wise Precision:")
                for class_name, prec_val in val_metrics['precision'].items():
                    if class_name != 'mean':
                        print(f"  {class_name}: {prec_val:.4f}")
                print(f"  Mean Precision: {val_metrics['precision']['mean']:.4f}")
                print(f"\nClass-wise Recall:")
                for class_name, rec_val in val_metrics['recall'].items():
                    if class_name != 'mean':
                        print(f"  {class_name}: {rec_val:.4f}")
                print(f"  Mean Recall: {val_metrics['recall']['mean']:.4f}")
                print(f"{'='*70}\n")
                
                # Save best model based on overall validation performance
                overal_val_performance = 0.6 * abwmh_val_dice + 0.3 * vent_val_dice + 0.1 * (1 - 1*avg_val_loss)
                if overal_val_performance > best_val_dice and beta_value.numpy() > 0.9:
                    best_val_dice = overal_val_performance
                    model.save_weights(f"{config.checkpoint_dir}/best_dice_model.h5")
                    print(f"✓ Best model saved (performance: {best_val_dice:.4f})")
            else:
                print("Warning: No valid validation batches")
                history['val_loss'].append(float('nan'))
                history['val_metrics'].append({})
                
            # Save checkpoint
            if (epoch + 1) % 5 == 0 and False:
                manager.save()
                print(f"  💾 Saved checkpoint")
            
            # Generate sample images
            if ((epoch + 1) % 5 == 0 or epoch == 0) or True:
                generate_and_save_images(
                    model, example_paired, example_target,
                    epoch + 1, config.figures_dir, config.num_classes
                )
                print(f"  📊 Saved visualization")
        
        # # Save final model
        # final_model_path = config.checkpoint_dir / "final_model.h5"
        # model.save(final_model_path)
        # print(f"\n✅ Training complete! Final model saved to {final_model_path}")
        
        # Save history
        history_serializable = {
            key: [float(val) if isinstance(val, (int, float, np.number)) else val 
                  for val in values]
            for key, values in history.items()
        }
        
        history_file = config.checkpoint_dir / "history.json"
        with open(history_file, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        return history, history_file
    
    finally:
        # CRITICAL: Always cleanup, even if training fails
        print("\n🧹 Cleaning up resources...")

        # Delete models explicitly to break references
        try:
            del model
            del optimizer
            del checkpoint
            del manager
            del train_dataset
            del val_dataset
            print("✅ Deleted model objects")
        except Exception as e:
            print(f"⚠️  Error deleting objects: {e}")
        
        # Clear GPU memory
        clear_gpu_memory()
        
        # Check final GPU memory
        get_gpu_memory_info()

###################### Main Execution ######################

if __name__ == "__main__":
    
    # Example: Train a specific U-Net for 3-class, standard preprocessing, fold 0
    
    config = ExperimentConfig(
        variant=3,
        preprocessing='standard',
        class_scenario='3class',
        fold_id=0,
        architecture_name='dlv3unet'    # ['unet', 'attnunet', 'dlv3unet', transunet']
    )
    
    history, history_path = train_net(config)
    
    print("\n" + "="*70)
    print("U-NET TRAINING COMPLETE")
    print("="*70)

