"""
Enhanced WMH and Ventricles Segmentation with U-Net Models - Journal Paper Implementation
Three-class segmentation: Background vs Ventricles vs Abnormal WMH
Professional results saving and visualization for publication

This relates to our article:
"Deep Learning-Based Neuroanatomical Profiling Reveals Detailed Brain Changes:
A Large-Scale Multiple Sclerosis Study"

Authors:
"Mahdi Bashiri Bawil, Mousa Shamsi, Abolhassan Shakeri Bavil"

Developer:
"Mahdi Bashiri Bawil"
"""

###################### Libraries ######################

# General Utilities
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import cv2 as cv
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import pickle
from pathlib import Path
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.measure import label

# Deep Learning
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras import backend as K
from tensorflow.keras import layers, optimizers, callbacks
from keras.utils import to_categorical

# Analysis and Statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Models
from unet_model import build_unet_3class
from attn_unet_model import build_attention_unet_3class
from trans_unet_model import build_trans_unet_3class
from dlv3_unet_model import build_deeplabv3_unet_3class

# Loss Functions
from loss_functions_v2 import *

# Metrics Functions
from metrics_functions import *

# Check for GPU assistance
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())
print("Built with CUDA: ", tf.test.is_built_with_cuda())
print("Physical devices: ", tf.config.list_physical_devices())

# Force GPU if available
if tf.config.list_physical_devices('GPU'):
    print("\n\n\t\t\tUsing GPU\n\n")
else:
    print("\n\n\t\t\tUsing CPU\n\n")

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Set publication-ready matplotlib settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

###################### Configuration and Setup ######################

class Config:
    """Configuration class for multi-class segmentation experiment"""
    def __init__(self):
        # Model Name
        self.model_name = 'unet'  # 'unet', 'attn_unet', 'trans_unet', 'deepl3_unet'
        
        # Paths
        self.train_dir = "WMH_VENT_Article_Data/train_3L_wmh_vent_local_public/"
        self.test_dir = "WMH_VENT_Article_Data/test_3L_wmh_vent_local_public/"
        self.intended_study_dir = "multiclass_results_20251124_160949_attn_unet"  # for inference intentions

        # Model parameters
        self.input_shape = (256, 256, 1)
        self.target_size = (256, 256)
        self.num_classes = 3  # Background (0), Ventricles (1), Abnormal WMH (2)
        
        # Training parameters
        self.mode = 'inference'  # 'training' or 'inference'
        self.epochs = 100
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.validation_split = 0.1
        self.random_state = 42
        
        # Loss function
        self.loss_function = 'weighted_categorical'     # unified_focal, weighted_categorical, multiclass_dice, categorical
        
        # Model selection
        if self.model_name == 'unet':
            self.build_model = build_unet_3class
        elif self.model_name == 'attn_unet':
            self.build_model = build_attention_unet_3class
        elif self.model_name == 'trans_unet':
            self.build_model = build_trans_unet_3class
        elif self.model_name == 'deepl3_unet':
            self.build_model = build_deeplabv3_unet_3class
        
        # Create results directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.mode != 'training':
            self.results_dir = Path(f"multiclass_results_{self.timestamp}_{self.model_name}_no_training")
        else:
            self.results_dir = Path(f"multiclass_results_{self.timestamp}_{self.model_name}")
        self.create_directory_structure()
        
    def create_directory_structure(self):
        """Create professional directory structure for results"""
        subdirs = [
            'models',
            'figures',
            'tables',
            'statistics',
            'predictions',
            'logs',
            'config'
        ]
        
        self.results_dir.mkdir(exist_ok=True)
        for subdir in subdirs:
            (self.results_dir / subdir).mkdir(exist_ok=True)
            
        # Save experiment configuration
        config_dict = {
            'timestamp': self.timestamp,
            'model_name': self.model_name,
            'input_shape': self.input_shape,
            'target_size': self.target_size,
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'random_state': self.random_state,
            'loss_function': self.loss_function
        }
        
        with open(self.results_dir / 'config' / 'experiment_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

config = Config()

###################### Data Loading Functions ######################

def extract_number(filename):
    """Extract patient ID and slice number for proper sorting"""
    return int(''.join(filter(str.isdigit, filename.split('_')[0])))

def load_wmh_dataset(data_dir, target_size=(256, 256), save_info=True):
    """
    Load dataset with format: 256x512 images (FLAIR + GT mask concatenated)
    Classes: 0=Background, 1=Ventricles, 2=Abnormal WMH
    """
    images, masks = [], []
    # image_files = sorted(os.listdir(data_dir), key=extract_number)
    image_files = [f for f in os.listdir(data_dir) if (f.startswith('1') and ((int(f.split('_')[-1][:-4]) >= 8 and int(f.split('_')[-1][:-4]) <= 15))) or (f.startswith('c') and ((int(f.split('_')[-1][:-4]) >= 20 and int(f.split('_')[-1][:-4]) <= 46)))]
    # image_files = [f for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.tif')]
    # image_files = [f for f in os.listdir(data_dir)]

    dataset_info = {
        'total_files': len(image_files),
        'loaded_files': 0,
        'skipped_files': [],
        'image_shapes': [],
        'class_distributions': {'background': [], 'ventricles': [], 'abnormal_wmh': []}
    }
    
    for img_name in tqdm(image_files, desc=f"Loading from {os.path.basename(data_dir)}"):
        # Load concatenated image
        full_img = cv.imread(os.path.join(data_dir, img_name), cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE).astype(np.float32)
        
        if full_img is None or full_img.shape[1] != 512:
            dataset_info['skipped_files'].append(img_name)
            continue
        
        # Split into FLAIR and GT
        flair_img = full_img[:, :256]
        gt_mask = full_img[:, 256:]
        
        # Resize if needed
        if target_size != (256, 256):
            flair_img = cv.resize(flair_img, target_size)
            gt_mask = cv.resize(gt_mask, target_size)
        
        dataset_info['image_shapes'].append(flair_img.shape)
        
        # Normalize FLAIR image
        flair_img = flair_img.astype(np.float32)
        flair_img = (flair_img - np.mean(flair_img)) / (np.std(flair_img) + 1e-7)
        flair_img = np.expand_dims(flair_img, axis=-1)
        
        # Process ground truth masks
        gt_mask = gt_mask.astype(np.float32)
        
        # Create 3-class mask: 0=Background, 1=Ventricles, 2=Abnormal WMH
        mask_3class = np.zeros_like(gt_mask, dtype=np.uint8)
        threshold_1 = 32767 // 2
        threshold_2 = 32767 + 1000
        threshold_3 = 65535 - 32767 // 2
        mask_3class[gt_mask < threshold_1] = 0
        mask_3class[(gt_mask >= threshold_1) & (gt_mask < threshold_2)] = 1  # Ventricles
        mask_3class[gt_mask >= threshold_3] = 2  # Abnormal WMH
        
        # Record class distributions
        unique, counts = np.unique(mask_3class, return_counts=True)
        class_dist = dict(zip(unique, counts))
        dataset_info['class_distributions']['background'].append(class_dist.get(0, 0))
        dataset_info['class_distributions']['ventricles'].append(class_dist.get(1, 0))
        dataset_info['class_distributions']['abnormal_wmh'].append(class_dist.get(2, 0))
        
        images.append(flair_img)
        masks.append(mask_3class)
        dataset_info['loaded_files'] += 1
    
    # Save dataset information
    if save_info:
        dataset_info['class_distributions'] = {k: np.array(v) for k, v in dataset_info['class_distributions'].items()}
        with open(config.results_dir / 'logs' / f'dataset_info_{os.path.basename(data_dir)}.pkl', 'wb') as f:
            pickle.dump(dataset_info, f)
    
    return np.array(images), np.array(masks), dataset_info

###################### U-Net Architecture ######################
# callable from related functions saved in the main directory

###################### Loss Functions ######################
# callable from the related function saved in the main directory

###################### Metrics and Evaluation ######################
# callable from related functions saved in the main directory

###################### Post Processing ######################

def post_process_predictions(predictions, class_id, min_object_size=5, apply_opening=True, kernel_size=3):
    """
    Post-process predictions for a specific class
    
    Args:
        predictions: Multi-class prediction masks
        class_id: Class to post-process (1=Ventricles, 2=Abnormal WMH)
        min_object_size: Minimum object size in pixels
        apply_opening: Whether to apply morphological opening
        kernel_size: Size of morphological kernel
    
    Returns:
        post_processed: Cleaned binary masks for the specific class
    """
    post_processed = np.zeros_like(predictions, dtype=np.uint8)
    
    for i in range(predictions.shape[0]):
        # Extract binary mask for specific class
        mask = (predictions[i] == class_id).astype(bool)
        
        # Remove small objects
        if min_object_size > 0:
            mask = remove_small_objects(mask, min_size=min_object_size)
        
        # Apply morphological opening
        if apply_opening:
            kernel = disk(kernel_size)
            mask = binary_opening(mask, kernel)
            if min_object_size > 0:
                mask = remove_small_objects(mask, min_size=min_object_size)
        
        post_processed[i] = mask.astype(np.uint8)
    
    return post_processed

###################### Professional Visualization Functions ######################

class PublicationPlotter:
    """Professional plotting class for publication-quality figures"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / 'figures'
    
    def plot_training_curves(self, history, save_name='training_curves'):
        """Plot publication-quality training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if hasattr(history, 'history'):
            hist = history.history
        else:
            hist = history
        
        # Loss
        axes[0].plot(hist['loss'], 'b-', linewidth=2, label='Training')
        axes[0].plot(hist['val_loss'], 'r-', linewidth=2, label='Validation')
        axes[0].set_title('(a) Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(hist['accuracy'], 'b-', linewidth=2, label='Training')
        axes[1].plot(hist['val_accuracy'], 'r-', linewidth=2, label='Validation')
        axes[1].set_title('(b) Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{save_name}.png')
        plt.savefig(self.figures_dir / f'{save_name}.pdf')
        plt.close()
    
    def plot_comparison_visualization(self, images, gt_masks, pred_masks, 
                                    indices=None, save_name='comparison_visualization'):
        """
        Create publication-quality comparison visualization
        Shows: FLAIR, GT, Predictions, and separate performance for each class
        """
        if indices is None:
            indices = np.random.choice(len(images), 3, replace=False)
        # or Use manual selection:
        # indices = np.array([50, 51, 62, 74])  # Your chosen indices
        
        n_samples = len(indices)
        n_cols = 7  # FLAIR, GT, Pred, Ventricles Performance, WMH Performance, Combined, Legend
        
        fig, axes = plt.subplots(n_samples, n_cols, figsize=(21, 3 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for sample_idx, idx in enumerate(indices):
            # FLAIR Image
            axes[sample_idx, 0].imshow(images[idx].squeeze(), cmap='gray')
            axes[sample_idx, 0].set_title('FLAIR Image')
            axes[sample_idx, 0].axis('off')
            
            # Ground Truth
            axes[sample_idx, 1].imshow(gt_masks[idx], cmap='gray', vmin=0, vmax=2)
            axes[sample_idx, 1].set_title('Ground Truth')
            axes[sample_idx, 1].axis('off')
            
            # Predictions
            axes[sample_idx, 2].imshow(pred_masks[idx], cmap='gray', vmin=0, vmax=2)
            axes[sample_idx, 2].set_title('Predictions')
            axes[sample_idx, 2].axis('off')
            
            # Create RGB FLAIR for overlays
            flair_rgb = np.stack([images[idx].squeeze()] * 3, axis=-1)
            flair_rgb = (flair_rgb - flair_rgb.min()) / (flair_rgb.max() - flair_rgb.min())
            
            # Ventricles Performance (Class 1)
            gt_vent = (gt_masks[idx] == 1)
            pred_vent = (pred_masks[idx] == 1)
            tp_vent = gt_vent & pred_vent
            fp_vent = (~gt_vent) & pred_vent
            fn_vent = gt_vent & (~pred_vent)
            
            overlay_vent = flair_rgb.copy()
            overlay_vent[tp_vent, :] = [0, 1, 0]  # Green for TP
            overlay_vent[fp_vent, :] = [1, 0, 0]  # Red for FP
            overlay_vent[fn_vent, :] = [1, 1, 0]  # Yellow for FN
            
            axes[sample_idx, 3].imshow(overlay_vent)
            axes[sample_idx, 3].set_title('Ventricles Performance')
            axes[sample_idx, 3].axis('off')
            
            # Abnormal WMH Performance (Class 2)
            gt_wmh = (gt_masks[idx] == 2)
            pred_wmh = (pred_masks[idx] == 2)
            tp_wmh = gt_wmh & pred_wmh
            fp_wmh = (~gt_wmh) & pred_wmh
            fn_wmh = gt_wmh & (~pred_wmh)
            
            overlay_wmh = flair_rgb.copy()
            overlay_wmh[tp_wmh, :] = [0, 1, 0]  # Green for TP
            overlay_wmh[fp_wmh, :] = [1, 0, 0]  # Red for FP
            overlay_wmh[fn_wmh, :] = [0, 0, 1]  # Blue for FN
            
            axes[sample_idx, 4].imshow(overlay_wmh)
            axes[sample_idx, 4].set_title('Abnormal WMH Performance')
            axes[sample_idx, 4].axis('off')
            
            # Combined Performance (both classes)
            overlay_combined = flair_rgb.copy()
            # Ventricles in cyan shades
            overlay_combined[tp_vent, :] = [0, 0.8, 0.8]  # Cyan for TP Ventricles
            overlay_combined[fp_vent, :] = [1, 0, 0]  # Red for FP
            overlay_combined[fn_vent, :] = [1, 1, 0]  # Yellow for FN
            # Abnormal WMH in green/blue shades (with priority over ventricles)
            overlay_combined[tp_wmh, :] = [0, 1, 0]  # Green for TP WMH
            overlay_combined[fp_wmh, :] = [1, 0.5, 0]  # Orange for FP WMH
            overlay_combined[fn_wmh, :] = [0, 0, 1]  # Blue for FN WMH
            
            axes[sample_idx, 5].imshow(overlay_combined)
            axes[sample_idx, 5].set_title('Combined Performance')
            axes[sample_idx, 5].axis('off')
            
            # Legend
            axes[sample_idx, 6].axis('off')
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                       markersize=15, label='True Positive'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                       markersize=15, label='False Positive'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                       markersize=15, label='False Negative (WMH)'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', 
                       markersize=15, label='False Negative (Vent)')
            ]
            legend = axes[sample_idx, 6].legend(handles=legend_elements, 
                                               loc='center', fontsize=10)
            legend.get_title().set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f'{save_name}.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_metrics_per_class(self, metrics_dict, save_name='metrics_per_class'):
        """Plot metrics comparison for each class"""
        classes = ['Ventricles', 'Abnormal WMH', 'Overall']
        metrics_names = ['Precision', 'Recall', 'Dice', 'HD95']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_names):
            values = [
                metrics_dict[f'{metric}_Ventricles'],
                metrics_dict[f'{metric}_WMH'],
                metrics_dict[f'{metric}_Overall']
            ]
            
            bars = axes[idx].bar(classes, values, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
            axes[idx].set_ylabel(metric)
            axes[idx].set_title(f'({chr(97+idx)}) {metric} by Class')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if metric == 'HD95':
                    label = f'{val:.2f}' if val != np.inf else 'N/A'
                else:
                    label = f'{val:.3f}'
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              label, ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{save_name}.png')
        plt.savefig(self.figures_dir / f'{save_name}.pdf')
        plt.close()

###################### Results Saving Functions ######################

class ResultsSaver:
    """Professional results saving and documentation"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        # os.makedirs(self.results_dir)
    
    def save_model(self, model, history):
        """Save trained model and training history"""
        model.save(self.results_dir / 'models' / 'multiclass_model.h5')
        
        with open(self.results_dir / 'models' / 'training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
    
    def load_model(self, model_path, loss_func):
        """Load saved model and training history"""
        try:
            model = keras.models.load_model(model_path, compile=False)
            
            history_path = Path(model_path).parent / 'training_history.pkl'
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            
            print("Model and history loaded successfully!")
            return model, history
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    
    def save_predictions(self, test_images, test_masks, predictions, dataset_info):
        """Save predictions and test data"""
        predictions_dir = self.results_dir / 'predictions'
        
        np.save(predictions_dir / 'test_images.npy', test_images)
        np.save(predictions_dir / 'test_masks.npy', test_masks)
        np.save(predictions_dir / 'predictions.npy', predictions)
        
        with open(predictions_dir / 'dataset_info.pkl', 'wb') as f:
            pickle.dump(dataset_info, f)
    
    def save_metrics_table(self, metrics_dict, model_name):
        """Save comprehensive metrics table"""
        # Create metrics table
        data = {
            'Class': ['Ventricles', 'Abnormal WMH', 'Overall'],
            'Precision': [
                metrics_dict['Precision_Ventricles'],
                metrics_dict['Precision_WMH'],
                metrics_dict['Precision_Overall']
            ],
            'Recall': [
                metrics_dict['Recall_Ventricles'],
                metrics_dict['Recall_WMH'],
                metrics_dict['Recall_Overall']
            ],
            'Dice': [
                metrics_dict['Dice_Ventricles'],
                metrics_dict['Dice_WMH'],
                metrics_dict['Dice_Overall']
            ],
            'HD95': [
                metrics_dict['HD95_Ventricles'],
                metrics_dict['HD95_WMH'],
                metrics_dict['HD95_Overall']
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Save as CSV and Excel
        df.to_csv(self.results_dir / 'tables' / f'{model_name}_metrics.csv', index=False)
        df.to_excel(self.results_dir / 'tables' / f'{model_name}_metrics.xlsx', index=False)
        
        # Save as LaTeX
        latex_table = df.to_latex(
            index=False,
            float_format="%.4f",
            caption=f"Performance metrics for {model_name} model",
            label=f"tab:{model_name}_metrics"
        )
        
        with open(self.results_dir / 'tables' / f'{model_name}_latex_table.tex', 'w') as f:
            f.write(latex_table)
        
        return df
    
    def save_statistical_analysis(self, all_model_metrics, class_metrics_per_model):
        """
        Comprehensive statistical analysis comparing:
        1. Models against each other
        2. Classes within each model
        """
        from scipy.stats import ttest_ind, f_oneway, kruskal
        
        stats_results = {
            'model_comparison': {},
            'class_comparison': {},
            'summary': {}
        }
        
        # Extract metrics by model
        models = list(all_model_metrics.keys())
        
        # Compare models for each metric
        for metric in ['Precision_Overall', 'Recall_Overall', 'Dice_Overall']:
            values_by_model = {model: all_model_metrics[model][metric] for model in models}
            
            # One-way ANOVA to compare all models
            if len(models) > 2:
                # For overall metrics (single value per model), we can't do ANOVA
                # Instead, record the values for comparison
                stats_results['model_comparison'][metric] = {
                    'values': values_by_model,
                    'best_model': max(values_by_model, key=values_by_model.get),
                    'best_value': max(values_by_model.values())
                }
        
        # Compare classes within each model
        for model_name in models:
            metrics = all_model_metrics[model_name]
            
            # Compare Ventricles vs WMH for each metric
            for metric_base in ['Precision', 'Recall', 'Dice']:
                vent_val = metrics[f'{metric_base}_Ventricles']
                wmh_val = metrics[f'{metric_base}_WMH']
                
                stats_results['class_comparison'][f'{model_name}_{metric_base}'] = {
                    'Ventricles': vent_val,
                    'WMH': wmh_val,
                    'Difference': wmh_val - vent_val,
                    'Better_Class': 'WMH' if wmh_val > vent_val else 'Ventricles'
                }
        
        # Save statistical results
        os.makedirs(self.results_dir / 'statistics')
        with open(self.results_dir / 'statistics' / 'statistical_analysis.json', 'w') as f:
            json.dump(stats_results, f, indent=2, default=str)
        
        # Create statistical report
        report = f"""
STATISTICAL ANALYSIS REPORT
===========================

MODEL COMPARISON:
-----------------
"""
        for metric, data in stats_results['model_comparison'].items():
            report += f"\n{metric}:\n"
            report += f"  Best Model: {data['best_model']} ({data['best_value']:.4f})\n"
            for model, val in data['values'].items():
                report += f"  {model}: {val:.4f}\n"
        
        report += "\n\nCLASS COMPARISON (within each model):\n"
        report += "--------------------------------------\n"
        for key, data in stats_results['class_comparison'].items():
            report += f"\n{key}:\n"
            report += f"  Ventricles: {data['Ventricles']:.4f}\n"
            report += f"  Abnormal WMH: {data['WMH']:.4f}\n"
            report += f"  Difference: {data['Difference']:+.4f}\n"
            report += f"  Better Class: {data['Better_Class']}\n"
        
        with open(self.results_dir / 'statistics' / 'statistical_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        return stats_results
    
    def generate_summary(self, config, dataset_info_train, dataset_info_test, metrics_dict):
        """Generate comprehensive summary"""
        summary = f"""
MULTI-CLASS WMH AND VENTRICLES SEGMENTATION RESULTS
===================================================
Experiment Timestamp: {config.timestamp}
Model Architecture: {config.model_name.upper()}

DATASET INFORMATION:
--------------------
Training Images: {dataset_info_train['loaded_files']}
Test Images: {dataset_info_test['loaded_files']}
Image Size: {config.target_size}
Classes: Background (0), Ventricles (1), Abnormal WMH (2)

METHODOLOGY:
------------
Architecture: {config.model_name.upper()}
Loss Function: {config.loss_function}
Training Epochs: {config.epochs}
Batch Size: {config.batch_size}
Learning Rate: {config.learning_rate}

PERFORMANCE RESULTS:
--------------------
                            | Ventricles    | Abnormal WMH  | Overall
--------------------|---------------|---------------|-------------
Precision           | {metrics_dict['Precision_Ventricles']:.4f}        | {metrics_dict['Precision_WMH']:.4f}        | {metrics_dict['Precision_Overall']:.4f}
Recall              | {metrics_dict['Recall_Ventricles']:.4f}           | {metrics_dict['Recall_WMH']:.4f}           | {metrics_dict['Recall_Overall']:.4f}
Dice Coefficient    | {metrics_dict['Dice_Ventricles']:.4f}             | {metrics_dict['Dice_WMH']:.4f}             | {metrics_dict['Dice_Overall']:.4f}
HD95 (pixels)       | {metrics_dict['HD95_Ventricles']:.4f}             | {metrics_dict['HD95_WMH']:.4f}             | {metrics_dict['HD95_Overall']:.4f}

KEY FINDINGS:
-------------
1. Best performing class: {max([('Ventricles', metrics_dict['Dice_Ventricles']), ('Abnormal WMH', metrics_dict['Dice_WMH'])], key=lambda x: x[1])[0]}
2. Ventricles Dice: {metrics_dict['Dice_Ventricles']:.4f}
3. Abnormal WMH Dice: {metrics_dict['Dice_WMH']:.4f}
4. Overall Dice: {metrics_dict['Dice_Overall']:.4f}

FILES GENERATED:
----------------
- Model: multiclass_model.h5
- Figures: training_curves.png/.pdf, comparison_visualization.png/.pdf, metrics_per_class.png/.pdf
- Tables: {config.model_name}_metrics.csv/.xlsx, {config.model_name}_latex_table.tex
- Statistics: statistical_analysis.json, statistical_report.txt
- Predictions: All test predictions and ground truth data saved

PUBLICATION READINESS:
----------------------
✓ High-resolution figures (300 DPI, PNG/PDF)
✓ LaTeX-formatted tables
✓ Comprehensive per-class metrics
✓ Post-processing applied
✓ Reproducible results with saved model
✓ Professional documentation
"""
        
        with open(self.results_dir / f'{config.model_name}_summary.txt', 'w') as f:
            f.write(summary)
        
        print("="*80)
        print(f"{config.model_name.upper()} RESULTS SUMMARY GENERATED")
        print("="*80)
        print(summary)

###################### Metrics Calculation ######################

def calculate_class_metrics(y_true, y_pred, class_id, class_name):
    """
    Calculate metrics for a specific class
    
    Args:
        y_true: Ground truth masks (H, W) with class labels
        y_pred: Predicted masks (H, W) with class labels
        class_id: ID of the class (1 or 2)
        class_name: Name of the class for printing
    
    Returns:
        dict: Dictionary containing all metrics for this class
    """
    # Extract binary masks for this class
    y_true_binary = (y_true == class_id).astype(np.uint8)
    y_pred_binary = (y_pred == class_id).astype(np.uint8)
    
    # Flatten for sklearn metrics
    y_true_flat = y_true_binary.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    # Calculate metrics
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # Calculate Dice
    intersection = np.sum(y_true_binary * y_pred_binary)
    dice = (2. * intersection) / (np.sum(y_true_binary) + np.sum(y_pred_binary) + 1e-7)
    
    # Calculate HD95
    hd95 = hausdorff_distance_95(y_true_binary, y_pred_binary)
    
    metrics = {
        f'Precision_{class_name}': precision,
        f'Recall_{class_name}': recall,
        f'Dice_{class_name}': dice,
        f'HD95_{class_name}': hd95
    }
    
    print(f"\n{class_name} Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Dice: {dice:.4f}")
    print(f"  HD95: {hd95:.4f}")
    
    return metrics

def calculate_per_image_metrics(y_true_images, y_pred_images, class_id):
    """
    Calculate per-image metrics for statistical analysis
    
    Args:
        y_true_images: Ground truth masks array (N, H, W)
        y_pred_images: Predicted masks array (N, H, W)
        class_id: Class ID to evaluate
    
    Returns:
        Arrays of per-image metrics
    """
    dice_scores = []
    hd95_scores = []
    precision_scores = []
    recall_scores = []
    
    for i in range(len(y_true_images)):
        y_true_binary = (y_true_images[i] == class_id).astype(np.uint8)
        y_pred_binary = (y_pred_images[i] == class_id).astype(np.uint8)
        
        # Dice
        intersection = np.sum(y_true_binary * y_pred_binary)
        dice = (2. * intersection) / (np.sum(y_true_binary) + np.sum(y_pred_binary) + 1e-7)
        dice_scores.append(dice)
        
        # HD95
        hd95 = hausdorff_distance_95(y_true_binary, y_pred_binary)
        hd95_scores.append(hd95)
        
        # Precision and Recall
        y_true_flat = y_true_binary.flatten()
        y_pred_flat = y_pred_binary.flatten()
        prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        precision_scores.append(prec)
        recall_scores.append(rec)
    
    return {
        'dice': np.array(dice_scores),
        'hd95': np.array(hd95_scores),
        'precision': np.array(precision_scores),
        'recall': np.array(recall_scores)
    }

def calculate_overall_metrics(y_true, y_pred):
    """
    Calculate overall metrics (macro-average across classes)
    """
    # Ventricles metrics
    vent_metrics = calculate_class_metrics(y_true, y_pred, 1, 'Ventricles')
    
    # Abnormal WMH metrics
    wmh_metrics = calculate_class_metrics(y_true, y_pred, 2, 'WMH')
    
    # Overall (macro-average)
    overall_metrics = {
        'Precision_Overall': (vent_metrics['Precision_Ventricles'] + wmh_metrics['Precision_WMH']) / 2,
        'Recall_Overall': (vent_metrics['Recall_Ventricles'] + wmh_metrics['Recall_WMH']) / 2,
        'Dice_Overall': (vent_metrics['Dice_Ventricles'] + wmh_metrics['Dice_WMH']) / 2,
        'HD95_Overall': (vent_metrics['HD95_Ventricles'] + wmh_metrics['HD95_WMH']) / 2
    }
    
    print(f"\nOverall Metrics (Macro-Average):")
    print(f"  Precision: {overall_metrics['Precision_Overall']:.4f}")
    print(f"  Recall: {overall_metrics['Recall_Overall']:.4f}")
    print(f"  Dice: {overall_metrics['Dice_Overall']:.4f}")
    print(f"  HD95: {overall_metrics['HD95_Overall']:.4f}")
    
    # Combine all metrics
    all_metrics = {**vent_metrics, **wmh_metrics, **overall_metrics}
    
    return all_metrics

###################### Main Experiment Function ######################

def run_multiclass_experiment(config):
    """Main function to run the complete multi-class segmentation experiment"""
    
    print("="*80)
    print(f"STARTING MULTI-CLASS SEGMENTATION EXPERIMENT: {config.model_name.upper()}")
    print("="*80)
    
    # Initialize components
    plotter = PublicationPlotter(config.results_dir)
    saver = ResultsSaver(config.results_dir)
    
    # Load datasets
    print("\nLoading datasets...")
    train_images, train_masks, dataset_info_train = load_wmh_dataset(
        config.train_dir, config.target_size
    )
    
    test_images, test_masks, dataset_info_test = load_wmh_dataset(
        config.test_dir, config.target_size
    )
    
    # Split training data
    x_train, x_val, y_train, y_val = train_test_split(
        train_images, train_masks,
        test_size=config.validation_split, 
        random_state=config.random_state
    )
    
    print(f"Training: {x_train.shape[0]}, Validation: {x_val.shape[0]}, Test: {test_images.shape[0]}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train, config.num_classes)
    print(f"Class weights: {class_weights}")
    
    # Build and compile model
    print("\n" + "="*60)
    print(f"BUILDING AND TRAINING MODEL: {config.model_name.upper()}")
    print("="*60)
    
    model = config.build_model(config.input_shape, num_classes=config.num_classes)
    print(f"Model Parameters: {model.count_params():,}")
    
    # Configure loss function
    if config.loss_function == 'weighted_categorical':
        loss = weighted_categorical_crossentropy(class_weights)
    elif config.loss_function == 'unified_focal':
        loss = unified_focal_loss(class_weights)
    else:
        loss = 'categorical_crossentropy'
    
    model.compile(
        optimizer=optimizers.legacy.Adam(config.learning_rate),
        loss=loss,
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
    ]
    
    # Convert masks to categorical
    y_train_cat = to_categorical(y_train, num_classes=config.num_classes)
    y_val_cat = to_categorical(y_val, num_classes=config.num_classes)
    
    # Train model
    if config.mode == 'training':
        print("\nTraining model (Phase I: WCE Loss)...")
        history1 = model.fit(
            x_train, y_train_cat,
            validation_data=(x_val, y_val_cat),
            epochs=30, #config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        print("\nTraining model (Phase II: Unified Focal Loss)...")
        model.compile(
            optimizer=optimizers.legacy.Adam(config.learning_rate),
            loss=unified_focal_loss(class_weights),
            metrics=['accuracy']
        )
        history2 = model.fit(
            x_train, y_train_cat,
            validation_data=(x_val, y_val_cat),
            epochs=config.epochs-30,
            batch_size=config.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        # -------- Merge histories -------- #
        combined_history = {}
        for key in history1.history.keys():
            combined_history[key] = history1.history[key] + history2.history.get(key, [])

        # OPTIONAL: convert to a simple object like History
        class CombinedHistory:
            history = combined_history

        history = CombinedHistory()
                
        # Save model and history
        saver.save_model(model, history)
    else:
        print("\nLoading pre-trained model...")
        model, history = saver.load_model(config.intended_study_dir / 'models' / 'multiclass_model.h5', loss)
        if model is None:
            print("Failed to load model. Exiting...")
            return None
    
    # Generate predictions
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    test_pred = model.predict(test_images, batch_size=config.batch_size)
    test_pred_classes = np.argmax(test_pred, axis=-1)
    
    # Post-process predictions for each class
    print("\nApplying post-processing...")
    test_pred_processed = test_pred_classes.copy()
    
    # Post-process Ventricles (Class 1)
    vent_processed = post_process_predictions(
        test_pred_classes, 
        class_id=1,
        min_object_size=5, 
        apply_opening=True, 
        kernel_size=2
    )
    test_pred_processed[vent_processed == 1] = 1
    
    # Post-process Abnormal WMH (Class 2)
    wmh_processed = post_process_predictions(
        test_pred_classes,
        class_id=2,
        min_object_size=5,
        apply_opening=True,
        kernel_size=2
    )
    test_pred_processed[wmh_processed == 1] = 2
    
    # Calculate metrics
    print("\n" + "="*60)
    print("CALCULATING METRICS")
    print("="*60)
    
    # Overall metrics
    metrics_dict = calculate_overall_metrics(
        test_masks.flatten(),
        test_pred_processed.flatten()
    )
    
    # Per-image metrics for statistical analysis
    print("\nCalculating per-image metrics...")
    vent_per_image = calculate_per_image_metrics(test_masks, test_pred_processed, class_id=1)
    wmh_per_image = calculate_per_image_metrics(test_masks, test_pred_processed, class_id=2)
    
    # Filter out inf values for HD95
    vent_hd95_valid = vent_per_image['hd95'][np.isfinite(vent_per_image['hd95'])]
    wmh_hd95_valid = wmh_per_image['hd95'][np.isfinite(wmh_per_image['hd95'])]
    
    print(f"\nValid HD95 samples:")
    print(f"  Ventricles: {len(vent_hd95_valid)} / {len(test_masks)}")
    print(f"  Abnormal WMH: {len(wmh_hd95_valid)} / {len(test_masks)}")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    saver.save_predictions(test_images, test_masks, test_pred_processed, 
                          {'train': dataset_info_train, 'test': dataset_info_test})
    
    metrics_table = saver.save_metrics_table(metrics_dict, config.model_name)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plotter.plot_training_curves(history)
    plotter.plot_comparison_visualization(
        test_images, test_masks, test_pred_processed,
        indices=[0, 1, 2]  # Customize as needed
    )
    plotter.plot_metrics_per_class(metrics_dict)
    
    # Generate summary
    saver.generate_summary(config, dataset_info_train, dataset_info_test, metrics_dict)
    
    return {
        'config': config,
        'model': model,
        'history': history,
        'metrics': metrics_dict,
        'per_image_metrics': {
            'ventricles': vent_per_image,
            'wmh': wmh_per_image
        },
        'metrics_table': metrics_table
    }

###################### Multi-Model Comparison ######################

def create_multi_model_comparison(all_results, models_tested):
    """Create comparative analysis across all tested models"""
    
    comparative_dir = Path(f"multiclass_comparative_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    comparative_dir.mkdir(exist_ok=True)
    
    # Collect metrics from all models
    comparison_data = []
    
    for model_name in models_tested:
        if all_results[model_name] is not None:
            metrics = all_results[model_name]['metrics']
            
            for class_type in ['Ventricles', 'WMH', 'Overall']:
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Class': class_type,
                    'Precision': metrics[f'Precision_{class_type}'],
                    'Recall': metrics[f'Recall_{class_type}'],
                    'Dice': metrics[f'Dice_{class_type}'],
                    'HD95': metrics[f'HD95_{class_type}']
                })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save tables
    df_comparison.to_csv(comparative_dir / 'model_comparison.csv', index=False)
    df_comparison.to_excel(comparative_dir / 'model_comparison.xlsx', index=False)
    
    # Create pivot tables for better visualization
    for metric in ['Precision', 'Recall', 'Dice', 'HD95']:
        pivot = df_comparison.pivot(index='Model', columns='Class', values=metric)
        pivot.to_csv(comparative_dir / f'{metric.lower()}_comparison.csv')
    
    # Find best models
    best_models = {}
    for class_type in ['Ventricles', 'WMH', 'Overall']:
        class_data = df_comparison[df_comparison['Class'] == class_type]
        best_models[class_type] = {
            'Dice': class_data.nlargest(1, 'Dice')[['Model', 'Dice']].to_dict('records')[0],
            'HD95': class_data.nsmallest(1, 'HD95')[['Model', 'HD95']].to_dict('records')[0]
        }
    
    # Generate comparative report
    report = f"""
MULTI-MODEL COMPARATIVE ANALYSIS
=================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

MODELS TESTED: {', '.join([m.upper() for m in models_tested if all_results[m] is not None])}

PERFORMANCE COMPARISON:
-----------------------
{df_comparison.to_string(index=False)}

BEST PERFORMING MODELS BY CLASS:
---------------------------------
"""
    
    for class_type, best in best_models.items():
        report += f"\n{class_type}:"
        report += f"\n  Best Dice: {best['Dice']['Model']} ({best['Dice']['Dice']:.4f})"
        report += f"\n  Best HD95: {best['HD95']['Model']} ({best['HD95']['HD95']:.4f})"
    
    report += f"\n\nFiles saved in: {comparative_dir}\n"
    
    with open(comparative_dir / 'comparative_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + "="*80)
    print("MULTI-MODEL COMPARISON COMPLETED")
    print("="*80)
    print(report)
    
    return df_comparison, best_models

###################### Execute Experiments ######################

if __name__ == "__main__":
    
    # List of models to test
    models_to_test = ['unet', 'attn_unet', 'trans_unet', 'deepl3_unet']
    
    # Store results
    all_results = {}
    
    print("\n" + "="*80)
    print("STARTING MULTI-MODEL SEGMENTATION EXPERIMENTS")
    print("="*80)
    print(f"Models to test: {', '.join(models_to_test)}")
    
    # Run experiment for each model
    for model_idx, model_name in enumerate(models_to_test, 1):
        print("\n" + "="*80)
        print(f"EXPERIMENT {model_idx}/{len(models_to_test)}: {model_name.upper()}")
        print("="*80)
        
        # Create config for this model
        config = Config()
        config.model_name = model_name
        
        # Update model builder
        if model_name == 'unet':
            config.build_model = build_unet_3class
        elif model_name == 'attn_unet':
            config.build_model = build_attention_unet_3class
        elif model_name == 'trans_unet':
            config.build_model = build_trans_unet_3class
        elif model_name == 'deepl3_unet':
            config.build_model = build_deeplabv3_unet_3class
        
        # Recreate results directory       
        config.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.mode != 'training':
            config.results_dir = Path(f"multiclass_results_{config.timestamp}_{config.model_name}_no_training")
        else:
            config.results_dir = Path(f"multiclass_results_{config.timestamp}_{config.model_name}")
        config.create_directory_structure()

        # if we work on non-training mode, load (use) the previously trained models
        if config.model_name == 'unet':
            config.intended_study_dir = "multiclass_results_20251124_152044_unet"
        elif config.model_name == 'attn_unet':
            config.intended_study_dir = "multiclass_results_20251125_133300_attn_unet"
        elif config.model_name == 'trans_unet':
            config.intended_study_dir = "multiclass_results_20251124_171430_trans_unet"
        elif config.model_name == 'deepl3_unet':
            config.intended_study_dir = "multiclass_results_20251124_180934_deepl3_unet"

        # Set seeds for reproducibility
        np.random.seed(config.random_state)
        tf.random.set_seed(config.random_state)
        
        try:
            # Run experiment
            results = run_multiclass_experiment(config)
            all_results[model_name] = results
            
            print("\n" + "="*80)
            print(f"EXPERIMENT FOR {model_name.upper()} COMPLETED!")
            print("="*80)
            print(f"Results saved in: {config.results_dir}")
            
        except Exception as e:
            print("\n" + "="*80)
            print(f"ERROR: Experiment for {model_name.upper()} failed!")
            print("="*80)
            print(f"Error: {str(e)}")
            all_results[model_name] = None
            continue
    
    # Create comparative analysis
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    
    # Summary of all models
    print("\nSUMMARY:")
    print("-" * 80)
    for model_name in models_to_test:
        if all_results[model_name] is not None:
            metrics = all_results[model_name]['metrics']
            print(f"\n{model_name.upper()}:")
            print(f"  Overall Dice: {metrics['Dice_Overall']:.4f}")
            print(f"  Ventricles Dice: {metrics['Dice_Ventricles']:.4f}")
            print(f"  Abnormal WMH Dice: {metrics['Dice_WMH']:.4f}")
        else:
            print(f"\n{model_name.upper()}: FAILED")
    
    # Generate multi-model comparison
    if any(all_results.values()):
        df_comp, best_models = create_multi_model_comparison(all_results, models_to_test)
        
        # Also save statistical comparison across models and classes
        successful_results = {k: v for k, v in all_results.items() if v is not None}
        if len(successful_results) > 0:
            saver = ResultsSaver(Path(f"multiclass_comparative_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
            
            all_model_metrics = {k: v['metrics'] for k, v in successful_results.items()}
            class_metrics_per_model = {k: v['per_image_metrics'] for k, v in successful_results.items()}
            
            saver.save_statistical_analysis(all_model_metrics, class_metrics_per_model)
    
    print("\n" + "="*80)
    print("ALL FILES READY FOR PUBLICATION!")
    print("="*80)
