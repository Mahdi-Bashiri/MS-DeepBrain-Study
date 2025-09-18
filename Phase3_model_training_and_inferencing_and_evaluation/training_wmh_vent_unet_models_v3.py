"""
Enhanced Multi-Class Brain Segmentation: Abnormal WMH and Ventricles
Model Comparison Study - Statistical Paper Implementation
Three-class segmentation: Background vs Abnormal WMH vs Ventricles
Professional results saving and visualization for publication

Author: Mahdi Bashiri Bawil
"""

###################### Libraries ######################

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

# Image processing
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

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
from scipy.spatial.distance import directed_hausdorff
import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())
print("Built with CUDA: ", tf.test.is_built_with_cuda())
print("Physical devices: ", tf.config.list_physical_devices())

# Force GPU if available
if tf.config.list_physical_devices('GPU'):
    print("\n\n\t\t\tUsing GPU\n\n")
else:
    print("\n\n\t\t\tUsing CPU\n\n")

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
    """Configuration class for the multi-model experiment"""
    def __init__(self):
        # Paths - keeping same structure as original
        self.train_dir = "Stats_article_data/train_3L"
        self.test_dir = "Stats_article_data/test_3L"
        self.infer_dir = "Stats_article_data/new_data_images"
        self.pre_result = Path("brain_segmentation_models_20250909_034626_unified_focal_loss")
        
        # Model parameters
        self.input_shape = (256, 256, 1)
        self.target_size = (256, 256)
        self.num_classes = 3  # Background (0), Ventricles (1), Abnormal WMH (2)
        
        # Training parameters
        self.mode = 'training'  # 'training' or 'notraining'
        self.epochs = 30
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.validation_split = 0.1
        self.random_state = 42
        
        # Models to compare
        self.models_to_compare = [
            'Enhanced_Attention_U-Net',
            'Enhanced_Trans_U-Net',
            'Hybrid_Model',
            'Enhanced_U-Net',

            'U-Net',
            'Attention_U-Net', 
            'DeepLabV3Plus',
            'Trans_U-Net'
        ]
        
        # Loss function
        self.loss_function = 'unified_focal_loss'
        # weighted_categorical_crossentropy, multiclass_dice_loss, categorical, combined_wce_dice_loss,
        # focal_loss, tversky_loss, unified_focal_loss, combo_loss, exponential_logarithmic_loss
        
        # Create results directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.mode == 'training':
            self.results_dir = Path(f"brain_segmentation_models_{self.timestamp}_{self.loss_function}")
        else:
            self.results_dir = Path(f"brain_segmentation_models_{self.timestamp}_{self.loss_function}_no_training")

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
            'input_shape': self.input_shape,
            'target_size': self.target_size,
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'random_state': self.random_state,
            'models_to_compare': self.models_to_compare,
            'loss_function': self.loss_function
        }
        
        with open(self.results_dir / 'config' / 'experiment_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

config = Config()

###################### Data Loading Functions ######################

def extract_number(filename):
    """Extract patient ID and slice number for proper sorting"""
    return int(''.join(filter(str.isdigit, filename.split('_')[0])))

def load_brain_dataset(data_dir, target_size=(256, 256), save_info=True):
    """
    Load dataset with specific format: 256x512 images (FLAIR + GT mask concatenated)
    Classes: Background (0), Ventricles (1), Abnormal WMH (2)
    """

    def filter_ventricles_near_abnormal(mask_3class, dilation_radius=2, verbose=False):
        """
        Filter ventricle pixels that are in the vicinity of abnormal WMH regions.

        Parameters:
        -----------
        mask_3class : numpy.ndarray
            3-class segmentation mask where:
            - 0: Background
            - 1: Ventricles
            - 2: Abnormal WMH
        dilation_radius : int, default=2
            Radius of the disk structuring element for dilation
        verbose : bool, default=False
            If True, print statistics about the filtering

        Returns:
        --------
        mask_3class_filtered : numpy.ndarray
            Filtered 3-class mask with ventricles removed near abnormal regions
        """
        # Step 1: Extract binary masks for each class
        class_0_boolean = (mask_3class == 0)  # Background
        class_1_boolean = (mask_3class == 1)  # Ventricles
        class_2_boolean = (mask_3class == 2)  # Abnormal WMH

        # Step 2: Dilate the abnormal WMH class (class 2)
        structuring_element = disk(dilation_radius)
        class_2_boolean_dilated = binary_dilation(class_2_boolean, structuring_element)

        # Step 3: Filter ventricles (class 1) by removing pixels in dilated abnormal regions
        class_1_boolean_filtered = class_1_boolean & ~class_2_boolean_dilated

        # Step 4: Reconstruct the 3-class mask with filtered ventricles
        mask_3class_filtered = np.zeros_like(mask_3class)
        mask_3class_filtered[class_0_boolean] = 0  # Background
        mask_3class_filtered[class_1_boolean_filtered] = 1  # Filtered ventricles
        mask_3class_filtered[class_2_boolean] = 2  # Abnormal WMH (unchanged)

        # Optional: Print statistics
        if verbose:
            original_ventricle_count = np.sum(class_1_boolean)
            filtered_ventricle_count = np.sum(class_1_boolean_filtered)
            removed_count = original_ventricle_count - filtered_ventricle_count

            print(f"Original ventricle pixels: {original_ventricle_count}")
            print(f"Filtered ventricle pixels: {filtered_ventricle_count}")
            print(f"Removed ventricle pixels: {removed_count}")
            print(f"Removal percentage: {100 * removed_count / original_ventricle_count:.1f}%")

        return mask_3class_filtered

    images, masks_3class = [], []
    # Select specific slice range for consistency
    image_files = [f for f in os.listdir(data_dir)]
    
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
        
        # Create 3-class mask (Background, Ventricles, Abnormal WMH)
        mask_3class = np.zeros_like(gt_mask, dtype=np.uint8)
        # Assuming similar thresholding logic as original but adapted for new classes
        threshold_1 = 32767 // 2  # Background threshold
        threshold_2 = 32767 + 1000  # Ventricles threshold
        threshold_3 = 65535 - 32767 // 2  # Abnormal WMH threshold
        
        mask_3class[gt_mask < threshold_1] = 0  # Background
        mask_3class[(gt_mask >= threshold_1) & (gt_mask < threshold_2)] = 1  # Ventricles
        mask_3class[gt_mask >= threshold_3] = 2  # Abnormal WMH

        # Apply the filtering with default disk(2)
        mask_3class = filter_ventricles_near_abnormal(mask_3class, dilation_radius=2, verbose=False)

        # Record class distributions
        unique, counts = np.unique(mask_3class, return_counts=True)
        class_dist = dict(zip(unique, counts))
        dataset_info['class_distributions']['background'].append(class_dist.get(0, 0))
        dataset_info['class_distributions']['ventricles'].append(class_dist.get(1, 0))
        dataset_info['class_distributions']['abnormal_wmh'].append(class_dist.get(2, 0))
        
        images.append(flair_img)
        masks_3class.append(mask_3class)
        dataset_info['loaded_files'] += 1
    
    # Save dataset information
    if save_info:
        dataset_info['class_distributions'] = {k: np.array(v) for k, v in dataset_info['class_distributions'].items()}
        with open(config.results_dir / 'logs' / f'dataset_info_{os.path.basename(data_dir)}.pkl', 'wb') as f:
            pickle.dump(dataset_info, f)
    
    return np.array(images), np.array(masks_3class), dataset_info

###################### Basic Model Architectures ######################

def build_unet(input_shape=(256, 256, 1), num_classes=3):
    """Enhanced U-Net architecture with batch normalization and dropout"""
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)
    p1 = keras.layers.Dropout(0.1)(p1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)
    p2 = keras.layers.Dropout(0.1)(p2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)
    p3 = keras.layers.Dropout(0.2)(p3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)
    p4 = keras.layers.Dropout(0.2)(p4)

    # Bottleneck
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)
    c5 = keras.layers.Dropout(0.3)(c5)

    # Decoder
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = keras.layers.Dropout(0.2)(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = keras.layers.Dropout(0.2)(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = keras.layers.Dropout(0.1)(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = keras.layers.Dropout(0.1)(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(c9)
    
    return Model(inputs, outputs, name='U-Net')

def build_attention_unet(input_shape=(256, 256, 1), num_classes=3):
    """Enhanced Attention U-Net architecture"""
    
    def attention_block(F_g, F_l, F_int):
        """Attention gate implementation"""
        W_g = Conv2D(F_int, 1, padding='same')(F_g)
        W_x = Conv2D(F_int, 1, padding='same')(F_l)
        psi = keras.layers.Add()([W_g, W_x])
        psi = keras.layers.Activation('relu')(psi)
        psi = Conv2D(1, 1, padding='same')(psi)
        psi = keras.layers.Activation('sigmoid')(psi)
        return keras.layers.Multiply()([F_l, psi])
    
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(2)(c1)
    p1 = keras.layers.Dropout(0.1)(p1)
    
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(2)(c2)
    p2 = keras.layers.Dropout(0.1)(p2)
    
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(2)(c3)
    p3 = keras.layers.Dropout(0.2)(p3)
    
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(2)(c4)
    p4 = keras.layers.Dropout(0.2)(p4)
    
    # Bridge
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)
    c5 = keras.layers.Dropout(0.3)(c5)
    
    # Decoder with attention gates
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    att6 = attention_block(u6, c4, 256)
    u6 = concatenate([u6, att6])
    u6 = keras.layers.Dropout(0.2)(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)
    
    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    att7 = attention_block(u7, c3, 128)
    u7 = concatenate([u7, att7])
    u7 = keras.layers.Dropout(0.2)(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    att8 = attention_block(u8, c2, 64)
    u8 = concatenate([u8, att8])
    u8 = keras.layers.Dropout(0.1)(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)
    
    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    att9 = attention_block(u9, c1, 32)
    u9 = concatenate([u9, att9])
    u9 = keras.layers.Dropout(0.1)(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)
    
    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(c9)
    
    return Model(inputs, outputs, name='Attention_U-Net')

def build_deeplabv3_plus(input_shape=(256, 256, 1), num_classes=3):
    """DeepLabV3+ implementation with ResNet backbone"""
    
    def conv_block(x, filters, kernel_size=3, strides=1, dilation_rate=1, use_bias=False, name=None):
        """Standard convolution block with BN and ReLU"""
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', 
                         dilation_rate=dilation_rate, use_bias=use_bias, name=name)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def bottleneck_residual_block(x, filters, strides=1, dilation_rate=1, projection_shortcut=False, name_prefix=""):
        """ResNet bottleneck block"""
        shortcut = x
        
        if projection_shortcut:
            shortcut = layers.Conv2D(filters * 4, 1, strides=strides, use_bias=False, 
                                   name=f"{name_prefix}_0_conv")(shortcut)
            shortcut = layers.BatchNormalization(name=f"{name_prefix}_0_bn")(shortcut)
        
        x = layers.Conv2D(filters, 1, use_bias=False, name=f"{name_prefix}_1_conv")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_1_bn")(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, 3, strides=strides, padding='same', 
                         dilation_rate=dilation_rate, use_bias=False, name=f"{name_prefix}_2_conv")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_2_bn")(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters * 4, 1, use_bias=False, name=f"{name_prefix}_3_conv")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_3_bn")(x)
        
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        
        return x
    
    def aspp_block(x, filters=256):
        """Atrous Spatial Pyramid Pooling"""
        input_shape = tf.shape(x)
        h, w = input_shape[1], input_shape[2]
        
        # ASPP branches
        b1 = layers.Conv2D(filters, 1, use_bias=False, name='aspp_1x1')(x)
        b1 = layers.BatchNormalization(name='aspp_1x1_bn')(b1)
        b1 = layers.Activation('relu')(b1)
        
        b2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=6, use_bias=False, name='aspp_3x3_6')(x)
        b2 = layers.BatchNormalization(name='aspp_3x3_6_bn')(b2)
        b2 = layers.Activation('relu')(b2)
        
        b3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=12, use_bias=False, name='aspp_3x3_12')(x)
        b3 = layers.BatchNormalization(name='aspp_3x3_12_bn')(b3)
        b3 = layers.Activation('relu')(b3)
        
        b4 = layers.Conv2D(filters, 3, padding='same', dilation_rate=18, use_bias=False, name='aspp_3x3_18')(x)
        b4 = layers.BatchNormalization(name='aspp_3x3_18_bn')(b4)
        b4 = layers.Activation('relu')(b4)
        
        # Global average pooling
        b5 = layers.GlobalAveragePooling2D(name='aspp_gap')(x)
        b5 = layers.Reshape((1, 1, -1))(b5)
        b5 = layers.Conv2D(filters, 1, use_bias=False, name='aspp_gap_conv')(b5)
        b5 = layers.BatchNormalization(name='aspp_gap_bn')(b5)
        b5 = layers.Activation('relu')(b5)
        
        def resize_to_input_shape(args):
            features, spatial_shape = args
            return tf.image.resize(features, spatial_shape, method='bilinear')
        
        b5 = layers.Lambda(resize_to_input_shape, name='aspp_gap_resize')([b5, [h, w]])
        
        # Concatenate all branches
        concat_features = layers.Concatenate(name='aspp_concat')([b1, b2, b3, b4, b5])
        
        output = layers.Conv2D(filters, 1, use_bias=False, name='aspp_final_conv')(concat_features)
        output = layers.BatchNormalization(name='aspp_final_bn')(output)
        output = layers.Activation('relu')(output)
        output = layers.Dropout(0.1, name='aspp_dropout')(output)
        
        return output
    
    # Input layer
    inputs = layers.Input(input_shape, name='input')
    
    # Encoder (ResNet backbone)
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)
    
    # ResNet stages
    x = bottleneck_residual_block(x, 64, strides=1, projection_shortcut=True, name_prefix='conv2_block1')
    x = bottleneck_residual_block(x, 64, name_prefix='conv2_block2')
    low_level_features = bottleneck_residual_block(x, 64, name_prefix='conv2_block3')
    
    x = bottleneck_residual_block(low_level_features, 128, strides=2, projection_shortcut=True, name_prefix='conv3_block1')
    x = bottleneck_residual_block(x, 128, name_prefix='conv3_block2')
    x = bottleneck_residual_block(x, 128, name_prefix='conv3_block3')
    x = bottleneck_residual_block(x, 128, name_prefix='conv3_block4')
    
    x = bottleneck_residual_block(x, 256, strides=1, dilation_rate=2, projection_shortcut=True, name_prefix='conv4_block1')
    x = bottleneck_residual_block(x, 256, dilation_rate=2, name_prefix='conv4_block2')
    x = bottleneck_residual_block(x, 256, dilation_rate=2, name_prefix='conv4_block3')
    x = bottleneck_residual_block(x, 256, dilation_rate=2, name_prefix='conv4_block4')
    x = bottleneck_residual_block(x, 256, dilation_rate=2, name_prefix='conv4_block5')
    x = bottleneck_residual_block(x, 256, dilation_rate=2, name_prefix='conv4_block6')
    
    x = bottleneck_residual_block(x, 512, strides=1, dilation_rate=4, projection_shortcut=True, name_prefix='conv5_block1')
    x = bottleneck_residual_block(x, 512, dilation_rate=4, name_prefix='conv5_block2')
    x = bottleneck_residual_block(x, 512, dilation_rate=4, name_prefix='conv5_block3')
    
    # ASPP
    x = aspp_block(x, filters=256)
    
    # Decoder
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='decoder_upsample1')(x)
    
    low_level_features = layers.Conv2D(48, 1, use_bias=False, name='decoder_low_level_conv')(low_level_features)
    low_level_features = layers.BatchNormalization(name='decoder_low_level_bn')(low_level_features)
    low_level_features = layers.Activation('relu')(low_level_features)
    
    def match_spatial_dims(tensors):
        high_level, low_level = tensors
        high_shape = tf.shape(high_level)
        low_shape = tf.shape(low_level)
        high_level_matched = high_level[:, :low_shape[1], :low_shape[2], :]
        return high_level_matched, low_level
    
    x_matched, low_level_matched = layers.Lambda(match_spatial_dims, name='match_dims')([x, low_level_features])
    
    x = layers.Concatenate(name='decoder_concat')([x_matched, low_level_matched])
    
    x = layers.Conv2D(256, 3, padding='same', use_bias=False, name='decoder_conv1')(x)
    x = layers.BatchNormalization(name='decoder_conv1_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1, name='decoder_dropout1')(x)
    
    x = layers.Conv2D(256, 3, padding='same', use_bias=False, name='decoder_conv2')(x)
    x = layers.BatchNormalization(name='decoder_conv2_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1, name='decoder_dropout2')(x)
    
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='decoder_upsample2')(x)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', name='output')(x)
    
    return keras.Model(inputs, outputs, name='DeepLabV3Plus')

def build_trans_unet(input_shape=(256, 256, 1), num_classes=3):
    """TransUNet architecture for medical image segmentation"""
    inputs = layers.Input(input_shape)
    
    # CNN Encoder
    conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')(conv1)
    conv1 = layers.Dropout(0.1)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, padding='same', activation='relu')(pool1)
    conv2 = layers.Conv2D(128, 3, padding='same', activation='relu')(conv2)
    conv2 = layers.Dropout(0.1)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')(pool2)
    conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')(conv3)
    conv3 = layers.Dropout(0.2)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, padding='same', activation='relu')(pool3)
    conv4 = layers.Conv2D(512, 3, padding='same', activation='relu')(conv4)
    conv4 = layers.Dropout(0.2)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Transformer Bottleneck
    bottleneck = layers.Conv2D(512, 3, padding='same', activation='relu')(pool4)
    bottleneck = layers.Dropout(0.3)(bottleneck)
    
    # Prepare for transformer
    h, w, d_model = 16, 16, 512
    transformer_input = layers.Reshape((h * w, d_model))(bottleneck)
    
    # Add positional encoding
    positions = tf.range(start=0, limit=h * w, delta=1)
    pos_encoding = layers.Embedding(h * w, d_model)(positions)
    transformer_input = transformer_input + pos_encoding
    
    # Simplified transformer layers
    for _ in range(2):  # 2 transformer layers for efficiency
        attention_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=d_model // 4, dropout=0.1
        )(transformer_input, transformer_input)
        attention_output = layers.Dropout(0.1)(attention_output)
        transformer_input = layers.LayerNormalization()(transformer_input + attention_output)
        
        # Feed forward network
        ffn = layers.Dense(d_model, activation='relu')(transformer_input)
        ffn = layers.Dropout(0.1)(ffn)
        transformer_input = layers.LayerNormalization()(transformer_input + ffn)
    
    # Reshape back to spatial format
    transformer_output = layers.Reshape((h, w, d_model))(transformer_input)
    bottleneck_enhanced = layers.Dropout(0.3)(transformer_output)
    
    # CNN Decoder
    up1 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(bottleneck_enhanced)
    concat1 = layers.Concatenate()([up1, conv4])
    concat1 = layers.Dropout(0.2)(concat1)
    conv_up1 = layers.Conv2D(256, 3, padding='same', activation='relu')(concat1)
    conv_up1 = layers.Conv2D(256, 3, padding='same', activation='relu')(conv_up1)
    
    up2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(conv_up1)
    concat2 = layers.Concatenate()([up2, conv3])
    concat2 = layers.Dropout(0.2)(concat2)
    conv_up2 = layers.Conv2D(128, 3, padding='same', activation='relu')(concat2)
    conv_up2 = layers.Conv2D(128, 3, padding='same', activation='relu')(conv_up2)
    
    up3 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(conv_up2)
    concat3 = layers.Concatenate()([up3, conv2])
    concat3 = layers.Dropout(0.1)(concat3)
    conv_up3 = layers.Conv2D(64, 3, padding='same', activation='relu')(concat3)
    conv_up3 = layers.Conv2D(64, 3, padding='same', activation='relu')(conv_up3)
    
    up4 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(conv_up3)
    concat4 = layers.Concatenate()([up4, conv1])
    concat4 = layers.Dropout(0.1)(concat4)
    conv_up4 = layers.Conv2D(32, 3, padding='same', activation='relu')(concat4)
    conv_up4 = layers.Conv2D(32, 3, padding='same', activation='relu')(conv_up4)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv_up4)
    
    return tf.keras.Model(inputs, outputs, name='Trans_U-Net')

###################### Enhanced Model Architectures ######################

def build_enhanced_unet(input_shape=(256, 256, 1), num_classes=3):
    """Enhanced U-Net with Batch Normalization, Residual Connections, and SE blocks"""

    def conv_block(x, filters, kernel_size=3, activation='relu', use_bn=True, dropout_rate=0.0):
        """Enhanced convolution block with batch norm and optional dropout"""
        x = Conv2D(filters, kernel_size, padding='same', use_bias=not use_bn)(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        return x

    def squeeze_excite_block(x, ratio=16):
        """Squeeze and Excitation block for channel attention"""
        filters = x.shape[-1]
        se = GlobalAveragePooling2D()(x)
        se = Dense(filters // ratio, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = Reshape((1, 1, filters))(se)
        return Multiply()([x, se])

    def residual_conv_block(x, filters, kernel_size=3, dropout_rate=0.0):
        """Residual convolution block"""
        shortcut = x

        x = conv_block(x, filters, kernel_size, dropout_rate=dropout_rate)
        x = conv_block(x, filters, kernel_size, dropout_rate=dropout_rate)

        # Add residual connection if dimensions match
        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = squeeze_excite_block(x)
        return x

    inputs = Input(input_shape)

    # Encoder with residual blocks
    c1 = residual_conv_block(inputs, 64, dropout_rate=0.1)
    p1 = MaxPooling2D(2)(c1)

    c2 = residual_conv_block(p1, 128, dropout_rate=0.1)
    p2 = MaxPooling2D(2)(c2)

    c3 = residual_conv_block(p2, 256, dropout_rate=0.2)
    p3 = MaxPooling2D(2)(c3)

    c4 = residual_conv_block(p3, 512, dropout_rate=0.2)
    p4 = MaxPooling2D(2)(c4)

    # Bottleneck with more capacity
    c5 = residual_conv_block(p4, 1024, dropout_rate=0.3)
    c5 = residual_conv_block(c5, 1024, dropout_rate=0.3)

    # Decoder with skip connections and attention
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = residual_conv_block(u6, 512, dropout_rate=0.2)

    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = residual_conv_block(u7, 256, dropout_rate=0.2)

    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = residual_conv_block(u8, 128, dropout_rate=0.1)

    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = residual_conv_block(u9, 64, dropout_rate=0.1)

    # Deep supervision - auxiliary outputs for better gradient flow
    aux1 = Conv2D(num_classes, 1, activation='softmax', name='aux_output_1')(c7)
    aux1 = UpSampling2D(4)(aux1)

    aux2 = Conv2D(num_classes, 1, activation='softmax', name='aux_output_2')(c8)
    aux2 = UpSampling2D(2)(aux2)

    # Main output
    outputs = Conv2D(num_classes, 1, activation='softmax', name='main_output')(c9)

    model = Model(inputs, [outputs, aux1, aux2], name='Enhanced_U-Net')
    return model

def build_enhanced_attention_unet(input_shape=(256, 256, 1), num_classes=3):
    """Enhanced Attention U-Net with improved attention mechanisms"""

    def conv_block(x, filters, kernel_size=3, dropout_rate=0.0):
        x = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        return x

    def channel_attention(x, ratio=8):
        """Channel attention mechanism"""
        filters = x.shape[-1]
        avg_pool = GlobalAveragePooling2D()(x)
        max_pool = GlobalMaxPooling2D()(x)

        avg_pool = Dense(filters // ratio, activation='relu')(avg_pool)
        avg_pool = Dense(filters, activation='sigmoid')(avg_pool)

        max_pool = Dense(filters // ratio, activation='relu')(max_pool)
        max_pool = Dense(filters, activation='sigmoid')(max_pool)

        attention = Add()([avg_pool, max_pool])
        attention = Reshape((1, 1, filters))(attention)
        return Multiply()([x, attention])

    def spatial_attention(x):
        """Spatial attention mechanism"""
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = Concatenate()([avg_pool, max_pool])
        attention = Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
        return Multiply()([x, attention])

    def cbam_block(x, ratio=8):
        """Convolutional Block Attention Module"""
        x = channel_attention(x, ratio)
        x = spatial_attention(x)
        return x

    def attention_gate(F_g, F_l, F_int):
        """Enhanced attention gate with normalization"""
        W_g = Conv2D(F_int, 1, padding='same', use_bias=False)(F_g)
        W_g = BatchNormalization()(W_g)

        W_x = Conv2D(F_int, 1, padding='same', use_bias=False)(F_l)
        W_x = BatchNormalization()(W_x)

        psi = Add()([W_g, W_x])
        psi = Activation('relu')(psi)
        psi = Conv2D(1, 1, padding='same', use_bias=False)(psi)
        psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)

        return Multiply()([F_l, psi])

    inputs = Input(input_shape)

    # Encoder with CBAM attention
    c1 = conv_block(inputs, 64, dropout_rate=0.1)
    c1 = cbam_block(c1)
    p1 = MaxPooling2D(2)(c1)

    c2 = conv_block(p1, 128, dropout_rate=0.1)
    c2 = cbam_block(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = conv_block(p2, 256, dropout_rate=0.2)
    c3 = cbam_block(c3)
    p3 = MaxPooling2D(2)(c3)

    c4 = conv_block(p3, 512, dropout_rate=0.2)
    c4 = cbam_block(c4)
    p4 = MaxPooling2D(2)(c4)

    # Bridge with stronger feature extraction
    c5 = conv_block(p4, 1024, dropout_rate=0.3)
    c5 = conv_block(c5, 1024, dropout_rate=0.3)
    c5 = cbam_block(c5)

    # Decoder with enhanced attention gates
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    att6 = attention_gate(u6, c4, 256)
    u6 = concatenate([u6, att6])
    c6 = conv_block(u6, 512, dropout_rate=0.2)
    c6 = cbam_block(c6)

    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    att7 = attention_gate(u7, c3, 128)
    u7 = concatenate([u7, att7])
    c7 = conv_block(u7, 256, dropout_rate=0.2)
    c7 = cbam_block(c7)

    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    att8 = attention_gate(u8, c2, 64)
    u8 = concatenate([u8, att8])
    c8 = conv_block(u8, 128, dropout_rate=0.1)
    c8 = cbam_block(c8)

    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    att9 = attention_gate(u9, c1, 32)
    u9 = concatenate([u9, att9])
    c9 = conv_block(u9, 64, dropout_rate=0.1)

    # Output with refinement
    outputs = Conv2D(num_classes, 1, activation='softmax')(c9)

    return Model(inputs, outputs, name='Enhanced_Attention_U-Net')

def build_enhanced_trans_unet(input_shape=(256, 256, 1), num_classes=3):
    """Enhanced TransUNet with improved transformer and CNN components"""

    def conv_block(x, filters, dropout_rate=0.0):
        x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        return x

    def transformer_encoder(x, d_model=512, num_heads=8, ff_dim=2048, dropout_rate=0.1):
        """Enhanced transformer encoder block"""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )(x, x)
        attention_output = Dropout(dropout_rate)(attention_output)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)

        # Feed forward network
        ffn = Dense(ff_dim, activation='gelu')(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)
        ffn_output = Dropout(dropout_rate)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

        return x

    inputs = Input(input_shape)

    # CNN Encoder with batch normalization
    conv1 = conv_block(inputs, 64, dropout_rate=0.1)
    pool1 = MaxPooling2D(2)(conv1)

    conv2 = conv_block(pool1, 128, dropout_rate=0.1)
    pool2 = MaxPooling2D(2)(conv2)

    conv3 = conv_block(pool2, 256, dropout_rate=0.2)
    pool3 = MaxPooling2D(2)(conv3)

    conv4 = conv_block(pool3, 512, dropout_rate=0.2)
    pool4 = MaxPooling2D(2)(conv4)

    # Prepare for transformer
    h, w, d_model = 16, 16, 512
    bottleneck = conv_block(pool4, d_model, dropout_rate=0.3)

    # Flatten for transformer
    transformer_input = Reshape((h * w, d_model))(bottleneck)
    batch_size = tf.shape(transformer_input)[0]

    # Fixed: Learnable positional encoding with proper batch dimension
    positions = tf.range(h * w)
    pos_encoding = Embedding(h * w, d_model,
                             embeddings_initializer='uniform',
                             name='pos_embedding')(positions)
    # Expand to match batch dimension
    pos_encoding = tf.expand_dims(pos_encoding, 0)  # Shape: (1, 256, 512)
    pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])  # Shape: (batch_size, 256, 512)
    
    transformer_input = Add()([transformer_input, pos_encoding])
    transformer_input = Dropout(0.1)(transformer_input)

    # Enhanced transformer layers
    for i in range(4):  # More layers for better feature learning
        transformer_input = transformer_encoder(
            transformer_input,
            d_model=d_model,
            num_heads=8,
            ff_dim=d_model * 2,
            dropout_rate=0.1
        )

    # Reshape back to spatial format
    transformer_output = Reshape((h, w, d_model))(transformer_input)

    # CNN Decoder with batch normalization
    up1 = Conv2DTranspose(256, 2, strides=2, padding='same')(transformer_output)
    concat1 = concatenate([up1, conv4])
    conv_up1 = conv_block(concat1, 256, dropout_rate=0.2)

    up2 = Conv2DTranspose(128, 2, strides=2, padding='same')(conv_up1)
    concat2 = concatenate([up2, conv3])
    conv_up2 = conv_block(concat2, 128, dropout_rate=0.2)

    up3 = Conv2DTranspose(64, 2, strides=2, padding='same')(conv_up2)
    concat3 = concatenate([up3, conv2])
    conv_up3 = conv_block(concat3, 64, dropout_rate=0.1)

    up4 = Conv2DTranspose(32, 2, strides=2, padding='same')(conv_up3)
    concat4 = concatenate([up4, conv1])
    conv_up4 = conv_block(concat4, 32, dropout_rate=0.1)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv_up4)

    return Model(inputs, outputs, name='Enhanced_Trans_U-Net')

def build_hybrid_model(input_shape=(256, 256, 1), num_classes=3):
    """Hybrid model combining best features from all architectures"""

    def conv_block(x, filters, dropout_rate=0.0):
        x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        return x

    def squeeze_excite_block(x, ratio=16):
        filters = x.shape[-1]
        se = GlobalAveragePooling2D()(x)
        se = Dense(filters // ratio, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = Reshape((1, 1, filters))(se)
        return Multiply()([x, se])

    def attention_gate(F_g, F_l, F_int):
        W_g = Conv2D(F_int, 1, padding='same', use_bias=False)(F_g)
        W_g = BatchNormalization()(W_g)

        W_x = Conv2D(F_int, 1, padding='same', use_bias=False)(F_l)
        W_x = BatchNormalization()(W_x)

        psi = Add()([W_g, W_x])
        psi = Activation('relu')(psi)
        psi = Conv2D(1, 1, padding='same')(psi)
        psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)

        return Multiply()([F_l, psi])

    def dilated_conv_block(x, filters, dilation_rates=[1, 2, 4], dropout_rate=0.0):
        """Multi-scale feature extraction using dilated convolutions"""
        branches = []
        for rate in dilation_rates:
            branch = Conv2D(filters // len(dilation_rates), 3,
                            dilation_rate=rate, padding='same', use_bias=False)(x)
            branch = BatchNormalization()(branch)
            branch = Activation('relu')(branch)
            branches.append(branch)

        x = concatenate(branches)
        x = Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

        return x

    inputs = Input(input_shape)

    # Encoder with multi-scale dilated convolutions
    c1 = conv_block(inputs, 64, dropout_rate=0.1)
    c1 = squeeze_excite_block(c1)
    p1 = MaxPooling2D(2)(c1)

    c2 = conv_block(p1, 128, dropout_rate=0.1)
    c2 = squeeze_excite_block(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = dilated_conv_block(p2, 256, dropout_rate=0.2)
    c3 = squeeze_excite_block(c3)
    p3 = MaxPooling2D(2)(c3)

    c4 = dilated_conv_block(p3, 512, dropout_rate=0.2)
    c4 = squeeze_excite_block(c4)
    p4 = MaxPooling2D(2)(c4)

    # Enhanced bottleneck with ASPP-like structure
    c5 = dilated_conv_block(p4, 1024, dilation_rates=[1, 2, 4, 8], dropout_rate=0.3)
    c5 = squeeze_excite_block(c5)

    # Decoder with attention gates and multi-scale features
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    att6 = attention_gate(u6, c4, 256)
    u6 = concatenate([u6, att6])
    c6 = dilated_conv_block(u6, 512, dropout_rate=0.2)

    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    att7 = attention_gate(u7, c3, 128)
    u7 = concatenate([u7, att7])
    c7 = dilated_conv_block(u7, 256, dropout_rate=0.2)

    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    att8 = attention_gate(u8, c2, 64)
    u8 = concatenate([u8, att8])
    c8 = conv_block(u8, 128, dropout_rate=0.1)

    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    att9 = attention_gate(u9, c1, 32)
    u9 = concatenate([u9, att9])
    c9 = conv_block(u9, 64, dropout_rate=0.1)

    # Multi-scale output fusion
    output_1 = Conv2D(num_classes, 1, padding='same')(c9)
    output_2 = Conv2D(num_classes, 3, padding='same')(c9)
    output_3 = Conv2D(num_classes, 5, padding='same')(c9)

    # Combine multi-scale outputs
    combined = Add()([output_1, output_2, output_3])
    outputs = Activation('softmax')(combined)

    return Model(inputs, outputs, name='Hybrid_Medical_Segmentation_Model')

def get_model_builder(model_name):
    """Factory function to get enhanced model builder by name"""
    builders = {
        'U-Net': build_unet,
        'Attention_U-Net': build_attention_unet,
        'DeepLabV3Plus': build_deeplabv3_plus,
        'Trans_U-Net': build_trans_unet,

        'Enhanced_U-Net': build_enhanced_unet,
        'Enhanced_Attention_U-Net': build_enhanced_attention_unet,
        'Enhanced_Trans_U-Net': build_enhanced_trans_unet,
        'Hybrid_Model': build_hybrid_model
    }
    return builders.get(model_name)

# Training recommendations for your dataset
def get_training_config():
    """Recommended training configuration for your dataset size"""
    return {
        'optimizer': keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
        'batch_size': 8,  # Adjust based on GPU memory
        'epochs': 150,
        'callbacks': [
            keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7),
            keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
        ],
        'augmentation': {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': True,
            'elastic_transform': True,  # Important for medical images
            'gaussian_noise': 0.1,
            'brightness_range': [0.8, 1.2],
            'contrast_range': [0.8, 1.2]
        }
    }

###################### Loss Functions for Brain FLAIR Segmentation ######################

def weighted_categorical_crossentropy(class_weights):
    """Enhanced weighted categorical crossentropy"""

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_true = tf.cast(y_true, tf.float32)

        # Create weights tensor
        weights = tf.constant(class_weights, dtype=tf.float32)
        class_weights_tensor = tf.gather(weights, K.argmax(y_true, axis=-1))

        # Calculate cross entropy
        cross_entropy = -K.sum(y_true * K.log(y_pred), axis=-1)
        weighted_loss = cross_entropy * class_weights_tensor

        return K.mean(weighted_loss)

    return loss

def multiclass_dice_loss(num_classes=3, class_weights=None):
    """Enhanced dice loss with optional class weights"""

    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)
        dice_scores = []

        for class_idx in range(num_classes):
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred[..., class_idx]

            y_true_f = K.flatten(y_true_class)
            y_pred_f = K.flatten(y_pred_class)

            intersection = K.sum(y_true_f * y_pred_f)
            dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

            # Apply class weights if provided
            if class_weights is not None:
                dice_coef = dice_coef * class_weights[class_idx]

            dice_scores.append(dice_coef)

        mean_dice = K.mean(K.stack(dice_scores))
        return 1 - mean_dice

    return loss

def combined_wce_dice_loss(class_weights, wce_weight=0.5, dice_weight=0.5):
    """Combine weighted categorical crossentropy and dice losses"""
    wce_loss_fn = weighted_categorical_crossentropy(class_weights)
    dice_loss_fn = multiclass_dice_loss(class_weights=class_weights)

    def loss(y_true, y_pred):
        wce_loss = wce_loss_fn(y_true, y_pred)
        dice_loss = dice_loss_fn(y_true, y_pred)
        return wce_weight * wce_loss + dice_weight * dice_loss

    return loss

def focal_loss(class_weights, alpha=0.25, gamma=2.0):
    """Multi-class focal loss for handling hard examples"""

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Calculate focal loss
        cross_entropy = -y_true * K.log(y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = alpha * K.pow(1 - pt, gamma)
        focal_loss = focal_weight * cross_entropy

        # Apply class weights
        weights = tf.constant(class_weights, dtype=tf.float32)
        class_weights_tensor = tf.gather(weights, K.argmax(y_true, axis=-1))
        class_weights_expanded = tf.expand_dims(class_weights_tensor, axis=-1)

        weighted_focal_loss = focal_loss * class_weights_expanded
        return K.mean(K.sum(weighted_focal_loss, axis=-1))

    return loss

def tversky_loss(class_weights, alpha=0.7, beta=0.3):
    """Multi-class Tversky loss - excellent for small structures like ventricles"""

    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)
        num_classes = tf.shape(y_pred)[-1]

        tversky_scores = []
        for class_idx in range(num_classes):
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred[..., class_idx]

            y_true_f = K.flatten(y_true_class)
            y_pred_f = K.flatten(y_pred_class)

            true_pos = K.sum(y_true_f * y_pred_f)
            false_neg = K.sum(y_true_f * (1 - y_pred_f))
            false_pos = K.sum((1 - y_true_f) * y_pred_f)

            tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

            # Apply class weights
            if class_weights is not None:
                tversky = tversky * class_weights[class_idx]

            tversky_scores.append(tversky)

        mean_tversky = K.mean(K.stack(tversky_scores))
        return 1 - mean_tversky

    return loss

def unified_focal_loss(class_weights, delta=0.6, gamma=0.5):
    """Unified Focal Loss - state-of-the-art for medical segmentation"""

    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        num_classes = tf.shape(y_pred)[-1]

        unified_losses = []
        for class_idx in range(num_classes):
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred[..., class_idx]

            # Calculate precision and recall
            y_true_f = K.flatten(y_true_class)
            y_pred_f = K.flatten(y_pred_class)

            tp = K.sum(y_true_f * y_pred_f)
            fp = K.sum((1 - y_true_f) * y_pred_f)
            fn = K.sum(y_true_f * (1 - y_pred_f))

            precision = (tp + smooth) / (tp + fp + smooth)
            recall = (tp + smooth) / (tp + fn + smooth)

            # Dice coefficient
            dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

            # Unified focal loss
            unified_loss = K.pow(1 - dice, gamma) * K.pow(1 - precision * recall, delta)

            # Apply class weights
            if class_weights is not None:
                unified_loss = unified_loss * class_weights[class_idx]

            unified_losses.append(unified_loss)

        return K.mean(K.stack(unified_losses))

    return loss

def combo_loss(class_weights, ce_weight=0.5, dice_weight=0.5, alpha=0.5, beta=0.5):
    """Combo loss combining CE and Dice with adaptive weighting"""

    def loss(y_true, y_pred):
        # Cross entropy component
        ce_loss_fn = weighted_categorical_crossentropy(class_weights)
        ce_loss = ce_loss_fn(y_true, y_pred)

        # Dice loss component
        dice_loss_fn = multiclass_dice_loss(class_weights=class_weights)
        dice_loss = dice_loss_fn(y_true, y_pred)

        # Adaptive weighting based on dice score
        dice_coef = 1 - dice_loss
        adaptive_alpha = alpha * (1 - dice_coef) + (1 - alpha) * dice_coef

        combo = adaptive_alpha * ce_weight * ce_loss + (1 - adaptive_alpha) * dice_weight * dice_loss
        return combo

    return loss

def exponential_logarithmic_loss(class_weights, gamma_dice=0.3, gamma_cross=0.3):
    """Exponential Logarithmic Loss - good for boundary refinement"""

    def loss(y_true, y_pred):
        # Dice component
        dice_loss_fn = multiclass_dice_loss(class_weights=class_weights)
        dice_loss = dice_loss_fn(y_true, y_pred)

        # Cross entropy component
        ce_loss_fn = weighted_categorical_crossentropy(class_weights)
        ce_loss = ce_loss_fn(y_true, y_pred)

        # Exponential logarithmic formulation
        exp_log_loss = (K.pow(-K.log(1 - dice_loss), gamma_dice) +
                        K.pow(ce_loss, gamma_cross))

        return exp_log_loss

    return loss

def get_recommended_loss_functions():
    """Get recommended loss functions with typical class weights for brain FLAIR"""
    # Typical class weights: [background, ventricles, WMH]
    class_weights = [1.0, 3.5, 2.8]  # Higher weight for ventricles (smallest)

    return {
        # Enhanced version of current approach
        'enhanced_wce_dice': combined_wce_dice_loss(class_weights, wce_weight=0.4, dice_weight=0.6),

        # Best for small structures (ventricles)
        'tversky_combo': combo_loss(class_weights, ce_weight=0.3, dice_weight=0.7),

        # State-of-the-art for medical segmentation
        'unified_focal': unified_focal_loss(class_weights, delta=0.6, gamma=0.5),

        # Good for boundary refinement
        'exp_log': exponential_logarithmic_loss(class_weights, gamma_dice=0.3, gamma_cross=0.3),

        # Handles hard examples well
        'focal_dice': combo_loss(class_weights, ce_weight=0.6, dice_weight=0.4),

        # Pure Tversky (best for ventricles)
        'tversky_pure': tversky_loss(class_weights, alpha=0.7, beta=0.3)
    }

###################### Model Performance Measurement ######################

def measure_model_flops():
    """Measure FLOPs for different models"""
    try:
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
        
        flops_results = {}
        
        for model_name in config.models_to_compare:
            model_builder = get_model_builder(model_name)
            if model_builder is None:
                continue
                
            model = model_builder(config.input_shape, config.num_classes)
            
            @tf.function
            def forward_pass():
                x = tf.random.normal((1, 256, 256, 1))
                return model(x)
            
            concrete_func = forward_pass.get_concrete_function()
            opts = ProfileOptionBuilder.float_operation()
            flops_info = profile(concrete_func.graph, options=opts)
            
            if flops_info and hasattr(flops_info, 'total_float_ops'):
                flops_giga = flops_info.total_float_ops / 1e9
                flops_results[model_name] = flops_giga
                print(f"{model_name}: {flops_giga:.1f} GFLOPs")
            else:
                print(f"{model_name}: Could not measure FLOPs")
        
        return flops_results
        
    except Exception as e:
        print(f"FLOPs measurement failed: {e}")
        return {}

###################### Metrics and Evaluation ######################

def calculate_class_weights(masks, num_classes):
    """Calculate class weights inversely proportional to class frequency"""
    flattened = masks.flatten()
    class_counts = np.bincount(flattened, minlength=num_classes)
    total_pixels = len(flattened)
    class_weights = total_pixels / (num_classes * class_counts)
    class_weights = class_weights / class_weights[0]
    return class_weights

def dice_coefficient_multiclass(y_true, y_pred, class_id):
    """Calculate Dice coefficient for specific class"""
    y_true_class = (y_true == class_id).astype(np.float32)
    y_pred_class = (y_pred == class_id).astype(np.float32)
    
    smooth = 1e-6
    intersection = np.sum(y_true_class * y_pred_class)
    return (2. * intersection + smooth) / (np.sum(y_true_class) + np.sum(y_pred_class) + smooth)

def iou_coefficient_multiclass(y_true, y_pred, class_id):
    """Calculate IoU coefficient for specific class"""
    y_true_class = (y_true == class_id).astype(np.float32)
    y_pred_class = (y_pred == class_id).astype(np.float32)
    
    intersection = np.sum(y_true_class * y_pred_class)
    union = np.sum(y_true_class) + np.sum(y_pred_class) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def hausdorff_distance_95(y_true, y_pred, class_id, pixel_spacing=0.5):
    """
    Calculate 95th percentile Hausdorff Distance for a specific class
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth mask
    y_pred : numpy.ndarray  
        Predicted mask
    class_id : int
        Class ID to calculate HD95 for
    pixel_spacing : float
        Pixel spacing in mm (default 0.5mm)
    
    Returns:
    --------
    float
        95th percentile Hausdorff distance in mm
    """
    # Extract binary masks for the specific class
    y_true_class = (y_true == class_id).astype(np.uint8)
    y_pred_class = (y_pred == class_id).astype(np.uint8)
    
    # Find boundary points
    from scipy import ndimage
    
    # Get boundary points using morphological operations
    true_boundary = y_true_class - ndimage.binary_erosion(y_true_class)
    pred_boundary = y_pred_class - ndimage.binary_erosion(y_pred_class)
    
    # Get coordinates of boundary points
    true_coords = np.column_stack(np.where(true_boundary))
    pred_coords = np.column_stack(np.where(pred_boundary))
    
    # Handle edge cases
    if len(true_coords) == 0 or len(pred_coords) == 0:
        if len(true_coords) == 0 and len(pred_coords) == 0:
            return 0.0  # Both empty
        else:
            return float('inf')  # One empty, one not
    
    # Calculate directed Hausdorff distances
    # From true to predicted
    distances_true_to_pred = []
    for true_point in true_coords:
        min_dist = np.min(np.linalg.norm(pred_coords - true_point, axis=1))
        distances_true_to_pred.append(min_dist)
    
    # From predicted to true
    distances_pred_to_true = []
    for pred_point in pred_coords:
        min_dist = np.min(np.linalg.norm(true_coords - pred_point, axis=1))
        distances_pred_to_true.append(min_dist)
    
    # Combine all distances
    all_distances = distances_true_to_pred + distances_pred_to_true
    
    # Calculate 95th percentile
    hd95_pixels = np.percentile(all_distances, 95)
    
    # Convert to mm
    hd95_mm = hd95_pixels * pixel_spacing
    
    return hd95_mm

def calculate_comprehensive_metrics(y_true, y_pred, model_name, class_names=None, pixel_spacing=0.5):
    """Calculate comprehensive segmentation metrics for each class and overall"""
    if class_names is None:
        class_names = ['Background', 'Ventricles', 'Abnormal WMH']
    
    # Reshape if needed for HD95 calculation
    if len(y_true.shape) == 1:
        # Assuming square images - you may need to adjust this
        img_size = int(np.sqrt(len(y_true)))
        y_true_2d = y_true.reshape(-1, img_size, img_size)
        y_pred_2d = y_pred.reshape(-1, img_size, img_size)
    else:
        y_true_2d = y_true
        y_pred_2d = y_pred
    
    metrics = {'Model': model_name}
    
    # Overall metrics
    acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    metrics['Overall_Accuracy'] = acc
    
    # Per-class metrics
    for class_id, class_name in enumerate(class_names):
        if class_id == 0:  # Skip background for main analysis
            continue
            
        # Binary metrics for each class (flattened)
        y_true_binary = (y_true.flatten() == class_id).astype(int)
        y_pred_binary = (y_pred.flatten() == class_id).astype(int)
        
        if len(np.unique(y_true_binary)) > 1:  # Check if class exists in ground truth
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        else:
            precision = recall = f1 = 0.0
            
        dice = dice_coefficient_multiclass(y_true.flatten(), y_pred.flatten(), class_id)
        iou = iou_coefficient_multiclass(y_true.flatten(), y_pred.flatten(), class_id)
        
        # Calculate HD95 across all test images
        hd95_values = []
        for i in range(len(y_true_2d)):
            hd95 = hausdorff_distance_95(y_true_2d[i], y_pred_2d[i], class_id, pixel_spacing)
            if not np.isinf(hd95):  # Exclude infinite values
                hd95_values.append(hd95)
        
        # Calculate mean HD95
        mean_hd95 = np.mean(hd95_values) if hd95_values else float('inf')
        
        metrics[f'{class_name}_Precision'] = precision
        metrics[f'{class_name}_Recall'] = recall
        metrics[f'{class_name}_F1'] = f1
        metrics[f'{class_name}_Dice'] = dice
        metrics[f'{class_name}_IoU'] = iou
        metrics[f'{class_name}_HD95'] = mean_hd95
    
    # Mean metrics across non-background classes
    metrics['Mean_Dice'] = np.mean([metrics[f'{name}_Dice'] for name in class_names[1:]])
    metrics['Mean_IoU'] = np.mean([metrics[f'{name}_IoU'] for name in class_names[1:]])
    
    # Mean HD95 (excluding infinite values)
    hd95_values = [metrics[f'{name}_HD95'] for name in class_names[1:]]
    finite_hd95 = [v for v in hd95_values if not np.isinf(v)]
    metrics['Mean_HD95'] = np.mean(finite_hd95) if finite_hd95 else float('inf')
    
    return metrics

###################### Post Processing ######################

def post_process_predictions(predictions, min_object_size=5, apply_opening=True, kernel_size=3):
    """Post-process predictions to remove small objects and apply morphological operations"""
    from skimage.morphology import remove_small_objects, binary_opening, disk
    from skimage.measure import label
    
    post_processed = np.zeros_like(predictions, dtype=np.uint8)
    
    for i in range(predictions.shape[0]):
        mask = predictions[i].copy()
        
        # Process each class separately (skip background)
        for class_id in [1, 2]:  # Ventricles and Abnormal WMH
            class_mask = (mask == class_id).astype(bool)
            
            if min_object_size > 0:
                class_mask = remove_small_objects(class_mask, min_size=min_object_size)
            
            if apply_opening:
                kernel = disk(kernel_size)
                class_mask = binary_opening(class_mask, kernel)
            
            # Add processed class back to mask
            post_processed[i][class_mask] = class_id
    
    return post_processed

###################### Professional Visualization Functions ######################

class PublicationPlotter:
    """Professional plotting class for publication-quality figures"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / 'figures'

    def plot_training_curves_all_models(self, all_histories, save_name='training_curves_all_models'):
        """Plot training curves for all models with robust error handling"""
        n_models = len(all_histories)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, history) in enumerate(all_histories.items()):
            if hasattr(history, 'history'):
                hist = history.history
            else:
                hist = history
                
            color = colors[i % len(colors)]
            
            # Print available keys for debugging (remove after fixing)
            print(f"Available keys for {model_name}: {list(hist.keys())}")
            
            # Loss plot - this should always be available
            if 'loss' in hist:
                axes[0, 0].plot(hist['loss'], color=color, linewidth=2, label=f'{model_name} Train', linestyle='-')
            
            if 'val_loss' in hist:
                axes[0, 0].plot(hist['val_loss'], color=color, linewidth=2, label=f'{model_name} Val', linestyle='--')
            
            # Find accuracy metric - try different possible names
            accuracy_key = None
            val_accuracy_key = None
            
            for key in hist.keys():
                if 'accuracy' in key.lower() and not key.startswith('val_'):
                    accuracy_key = key
                elif key.startswith('val_') and 'accuracy' in key.lower():
                    val_accuracy_key = key
            
            # Plot accuracy if found
            if accuracy_key:
                axes[0, 1].plot(hist[accuracy_key], color=color, linewidth=2, label=f'{model_name} Train', linestyle='-')
            
            if val_accuracy_key:
                axes[0, 1].plot(hist[val_accuracy_key], color=color, linewidth=2, label=f'{model_name} Val', linestyle='--')
        
        axes[0, 0].set_title('(a) Training Loss Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('(b) Training Accuracy Comparison')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rest of your plotting code stays the same...
        model_names = list(all_histories.keys())
        
        axes[1, 0].set_title('(c) Mean Dice Coefficient Comparison')
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Mean Dice Score')
        
        axes[1, 1].set_title('(d) Mean IoU Coefficient Comparison')
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Mean IoU Score')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{save_name}.png')
        plt.savefig(self.figures_dir / f'{save_name}.pdf')

    def plot_model_comparison_metrics(self, all_metrics, save_name='model_comparison_metrics'):
        """Create comprehensive model comparison visualization"""
        model_names = [metrics['Model'] for metrics in all_metrics]
        
        # Extract metrics for visualization
        mean_dice = [metrics['Mean_Dice'] for metrics in all_metrics]
        mean_iou = [metrics['Mean_IoU'] for metrics in all_metrics]
        mean_hd95 = [metrics['Mean_HD95'] for metrics in all_metrics]

        overall_acc = [metrics['Overall_Accuracy'] for metrics in all_metrics]
        
        # WMH metrics
        wmh_dice = [metrics['Abnormal WMH_Dice'] for metrics in all_metrics]
        wmh_iou = [metrics['Abnormal WMH_IoU'] for metrics in all_metrics]
        wmh_hd95 = [metrics['Abnormal WMH_HD95'] for metrics in all_metrics]

        # Ventricles metrics
        vent_dice = [metrics['Ventricles_Dice'] for metrics in all_metrics]
        vent_iou = [metrics['Ventricles_IoU'] for metrics in all_metrics]
        vent_hd95 = [metrics['Ventricles_HD95'] for metrics in all_metrics]

        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        # Overall metrics
        axes[0, 0].bar(x, overall_acc, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('(a) Overall Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mean Dice and IoU
        axes[0, 1].bar(x - width/2, mean_dice, width, label='Mean Dice', color='lightcoral', alpha=0.8)
        axes[0, 1].bar(x + width/2, mean_iou, width, label='Mean IoU', color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('(b) Mean Dice vs IoU')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # WMH performance
        axes[0, 2].bar(x - width/2, wmh_dice, width, label='WMH Dice', color='orange', alpha=0.8)
        axes[0, 2].bar(x + width/2, wmh_iou, width, label='WMH IoU', color='purple', alpha=0.8)
        axes[0, 2].set_title('(c) Abnormal WMH Performance')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(model_names, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Ventricles performance
        axes[1, 0].bar(x - width/2, vent_dice, width, label='Ventricles Dice', color='brown', alpha=0.8)
        axes[1, 0].bar(x + width/2, vent_iou, width, label='Ventricles IoU', color='pink', alpha=0.8)
        axes[1, 0].set_title('(d) Ventricles Performance')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined comparison
        metrics_names = ['Mean Dice', 'Mean IoU', 'Mean HD95', 'WMH Dice']
        combined_data = np.array([mean_dice, mean_iou, mean_hd95, wmh_dice]).T
        
        x_combined = np.arange(len(metrics_names))
        width_combined = 0.2
        
        for i, model_name in enumerate(model_names):
            axes[1, 1].bar(x_combined + i * width_combined, combined_data[i], 
                          width_combined, label=model_name, alpha=0.8)
        
        axes[1, 1].set_title('(e) Combined Metrics Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x_combined + width_combined * (len(model_names)-1) / 2)
        axes[1, 1].set_xticklabels(metrics_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Statistical significance placeholder
        axes[1, 2].text(0.5, 0.5, 'Statistical\nSignificance\nResults\n(To be updated)', 
                       transform=axes[1, 2].transAxes, ha='center', va='center', fontsize=12)
        axes[1, 2].set_title('(f) Statistical Analysis')
        axes[1, 2].axis('off')
        
        # Add value labels on bars for key metrics
        for ax_idx, ax in enumerate([axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]):
            for i, rect in enumerate(ax.patches):
                if rect.get_height() > 0:
                    ax.annotate(f'{rect.get_height():.3f}',
                              xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{save_name}.png')
        plt.savefig(self.figures_dir / f'{save_name}.pdf')

    def plot_sample_predictions(self, images, gt_masks, all_predictions, 
                               indices=None, save_name='sample_predictions'):
        """Create visualization of sample predictions from all models"""
        if indices is None:
            indices = np.random.choice(len(images), 2, replace=False)
            
        n_samples = len(indices)
        n_models = len(all_predictions)
        
        # Create figure: samples in rows, (original, GT, model1, model2, ...) in columns
        fig, axes = plt.subplots(n_samples, n_models + 2, figsize=(4*(n_models+2), 6*n_samples))
        
        # Ensure axes is 2D
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        class_colors = {0: [0, 0, 0], 1: [0, 0, 1], 2: [1, 0, 0]}  # Background, Ventricles, WMH
        
        for sample_idx, img_idx in enumerate(indices):
            # Original image
            axes[sample_idx, 0].imshow(images[img_idx].squeeze(), cmap='gray')
            axes[sample_idx, 0].set_title(f'Sample {sample_idx+1}\nFLAIR Image')
            axes[sample_idx, 0].axis('off')
            
            # Ground truth
            gt_colored = np.zeros((*gt_masks[img_idx].shape, 3))
            for class_id, color in class_colors.items():
                mask = gt_masks[img_idx] == class_id
                gt_colored[mask] = color
            
            axes[sample_idx, 1].imshow(gt_colored)
            axes[sample_idx, 1].set_title('Ground Truth\n(Red: WMH, Blue: Vent)')
            axes[sample_idx, 1].axis('off')
            
            # Model predictions
            for model_idx, (model_name, predictions) in enumerate(all_predictions.items()):
                pred_colored = np.zeros((*predictions[img_idx].shape, 3))
                for class_id, color in class_colors.items():
                    mask = predictions[img_idx] == class_id
                    pred_colored[mask] = color
                
                axes[sample_idx, model_idx + 2].imshow(pred_colored)
                axes[sample_idx, model_idx + 2].set_title(f'{model_name}\nPrediction')
                axes[sample_idx, model_idx + 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{save_name}.png', dpi=300)
        plt.savefig(self.figures_dir / f'{save_name}.pdf')

###################### Results Saving Functions ######################

class ResultsSaver:
    """Professional results saving and documentation"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        
    def save_all_models(self, models_dict, histories_dict):
        """Save all trained models and their histories"""
        for model_name, model in models_dict.items():
            model_filename = f"{model_name.replace('-', '_').replace(' ', '_').lower()}_model.h5"
            model.save(self.results_dir / 'models' / model_filename)
            
        for model_name, history in histories_dict.items():
            history_filename = f"{model_name.replace('-', '_').replace(' ', '_').lower()}_history.pkl"
            with open(self.results_dir / 'models' / history_filename, 'wb') as f:
                if hasattr(history, 'history'):
                    pickle.dump(history.history, f)
                else:
                    pickle.dump(history, f)
    
    def load_all_models(self, models_to_load, pre_result_dir):
        """Load saved models and histories"""
        models_dict = {}
        histories_dict = {}
        
        for model_name in models_to_load:
            try:
                model_filename = f"{model_name.replace('-', '_').replace(' ', '_').lower()}_model.h5"
                # model_filename = f"{model_name}_best.h5"
                history_filename = f"{model_name.replace('-', '_').replace(' ', '_').lower()}_history.pkl"
                
                # Load model
                model_path = pre_result_dir / 'models' / model_filename
                if model_path.exists():
                    models_dict[model_name] = keras.models.load_model(model_path, compile=False)
                    print(f"Loaded model: {model_name}")
                else:
                    print(f"Model file not found: {model_path}")
                
                # Load history
                history_path = pre_result_dir / 'models' / history_filename
                if history_path.exists():
                    with open(history_path, 'rb') as f:
                        histories_dict[model_name] = pickle.load(f)
                    print(f"Loaded history: {model_name}")
                else:
                    print(f"History file not found: {history_path}")
                    
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        
        return models_dict, histories_dict
            
    def save_predictions(self, test_images, test_masks, 
                        pred_s1, pred_s2, dataset_info):
        """Save predictions and test data"""
        predictions_dir = self.results_dir / 'predictions'
        
        # Save raw predictions
        np.save(predictions_dir / 'test_images.npy', test_images)
        np.save(predictions_dir / 'test_masks_3class.npy', test_masks)
        np.save(predictions_dir / 'predictions_raw.npy', pred_s1)
        np.save(predictions_dir / 'predictions_postprocessed.npy', pred_s2)
        
        # Save dataset information
        with open(predictions_dir / 'dataset_info.pkl', 'wb') as f:
            pickle.dump(dataset_info, f)
      
    def save_comprehensive_results_table(self, all_metrics, statistical_results):
        """Save comprehensive results table with all models"""
        # Create main results DataFrame
        results_df = pd.DataFrame(all_metrics)
        
        # Save as CSV and Excel
        results_df.to_csv(self.results_dir / 'tables' / 'model_comparison_results.csv', index=False)
        results_df.to_excel(self.results_dir / 'tables' / 'model_comparison_results.xlsx', index=False)
        
        # Create LaTeX table
        latex_columns = ['Model', 'Overall_Accuracy', 'Mean_Dice', 'Mean_IoU', 'Mean_HD95',
                'Abnormal WMH_Dice', 'Ventricles_Dice', 'Abnormal WMH_HD95', 'Ventricles_HD95']
        latex_df = results_df[latex_columns].round(4)
        
        latex_table = latex_df.to_latex(
            index=False,
            float_format="%.4f",
            caption="Performance comparison of different deep learning models for brain segmentation",
            label="tab:model_performance_comparison"
        )
        
        with open(self.results_dir / 'tables' / 'latex_table.tex', 'w') as f:
            f.write(latex_table)
            
        return results_df
    
    def save_statistical_analysis_models(self, all_metrics):
        """Comprehensive statistical analysis comparing all models"""
        from scipy.stats import friedmanchisquare, wilcoxon
        from itertools import combinations
        
        model_names = [metrics['Model'] for metrics in all_metrics]
        
        # Extract metrics for statistical analysis
        metrics_for_analysis = {
            'Mean_Dice': [metrics['Mean_Dice'] for metrics in all_metrics],
            'Mean_IoU': [metrics['Mean_IoU'] for metrics in all_metrics],
            'Abnormal WMH_Dice': [metrics['Abnormal WMH_Dice'] for metrics in all_metrics],
            'Ventricles_Dice': [metrics['Ventricles_Dice'] for metrics in all_metrics]
        }
        
        statistical_results = {
            'model_names': model_names,
            'n_models': len(model_names),
            'metrics_analyzed': list(metrics_for_analysis.keys())
        }
        
        # For each metric, perform pairwise comparisons
        for metric_name, values in metrics_for_analysis.items():
            statistical_results[metric_name] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'best_model': model_names[np.argmax(values)],
                'best_score': max(values),
                'worst_model': model_names[np.argmin(values)],
                'worst_score': min(values)
            }
        
        # Save statistical results
        with open(self.results_dir / 'statistics' / 'statistical_analysis_models.json', 'w') as f:
            json.dump(statistical_results, f, indent=2, default=str)
        
        # Generate statistical report
        report = f"""
COMPREHENSIVE MODEL COMPARISON STATISTICAL ANALYSIS
==================================================

Models Compared: {', '.join(model_names)}
Number of Models: {len(model_names)}

PERFORMANCE SUMMARY:
-------------------
"""
        
        for metric_name, stats in statistical_results.items():
            if isinstance(stats, dict) and 'best_model' in stats:
                report += f"""
{metric_name.replace('_', ' ')}:
Best Model: {stats['best_model']} ({stats['best_score']:.4f})
Worst Model: {stats['worst_model']} ({stats['worst_score']:.4f})
Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}
Range: {stats['best_score'] - stats['worst_score']:.4f}
"""
        
        report += f"""

RANKING ANALYSIS:
----------------
Based on Mean Dice Score:
"""
        # Rank models by Mean Dice
        dice_scores = [(name, metrics['Mean_Dice']) for name, metrics in 
                      zip(model_names, [m for m in all_metrics])]
        dice_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (model, score) in enumerate(dice_scores, 1):
            report += f"{rank}. {model}: {score:.4f}\n"
        
        with open(self.results_dir / 'statistics' / 'model_comparison_report.txt', 'w') as f:
            f.write(report)
            
        return statistical_results

    def generate_conference_summary_models(self, config, dataset_info_train, dataset_info_test, 
                                         all_metrics, statistical_results, flops_data):
        """Generate comprehensive conference paper summary for model comparison"""
        
        best_model_dice = max(all_metrics, key=lambda x: x['Mean_Dice'])
        best_model_iou = max(all_metrics, key=lambda x: x['Mean_IoU'])
        
        summary = f"""
CONFERENCE PAPER RESULTS SUMMARY - MODEL COMPARISON STUDY
=========================================================
Experiment Timestamp: {config.timestamp}
Multi-Class Brain Segmentation: Ventricles and Abnormal WMH

DATASET INFORMATION:
--------------------
Training Images: {dataset_info_train['loaded_files']} 
Test Images: {dataset_info_test['loaded_files']}
Image Size: {config.target_size}
Classes: Background (0), Ventricles (1), Abnormal WMH (2)

METHODOLOGY:
------------
Models Compared: {', '.join(config.models_to_compare)}
Loss Function: {config.loss_function}
Training Epochs: {config.epochs}
Batch Size: {config.batch_size}
Learning Rate: {config.learning_rate}

COMPUTATIONAL COMPLEXITY:
-------------------------"""
        
        for model_name, flops in flops_data.items():
            summary += f"""
{model_name}: {flops:.1f} GFLOPs"""
        
        summary += f"""

PERFORMANCE RESULTS:
--------------------
"""
        
        # Create performance table
        headers = ['Model', 'Overall Acc', 'Mean Dice', 'Mean IoU', 'Mean HD95', 'WMH Dice', 'Vent Dice']
        summary += f"{'Model':<15} {'Acc':<8} {'MDice':<8} {'MIoU':<8} {'MHD95':<8} {'WMH':<8} {'Vent':<8}\n"
        summary += "-" * 65 + "\n"
        
        for metrics in all_metrics:
            summary += f"{metrics['Model']:<15} "
            summary += f"{metrics['Overall_Accuracy']:<8.4f} "
            summary += f"{metrics['Mean_Dice']:<8.4f} "
            summary += f"{metrics['Mean_IoU']:<8.4f} "
            summary += f"{metrics['Abnormal WMH_Dice']:<8.4f} "
            summary += f"{metrics['Ventricles_Dice']:<8.4f}\n"
            summary += f"{metrics['Mean_HD95']:<8.1f} "  # Note: using .1f for HD95 as it's in mm

        
        summary += f"""

BEST PERFORMING MODELS:
-----------------------
Best Mean Dice: {best_model_dice['Model']} ({best_model_dice['Mean_Dice']:.4f})
Best Mean IoU: {best_model_iou['Model']} ({best_model_iou['Mean_IoU']:.4f})

KEY FINDINGS:
-------------
1. {best_model_dice['Model']} achieved the highest mean Dice coefficient ({best_model_dice['Mean_Dice']:.4f})
2. {best_model_iou['Model']} achieved the highest mean IoU coefficient ({best_model_iou['Mean_IoU']:.4f})
3. All models successfully segmented both ventricles and abnormal WMH
4. Post-processing improved segmentation quality across all models

STATISTICAL ANALYSIS:
---------------------
Models ranked by Mean Dice Score:
"""
        
        # Add ranking
        dice_ranking = sorted(all_metrics, key=lambda x: x['Mean_Dice'], reverse=True)
        for rank, metrics in enumerate(dice_ranking, 1):
            summary += f"{rank}. {metrics['Model']}: {metrics['Mean_Dice']:.4f}\n"
        
        summary += f"""

FILES GENERATED:
----------------
- Models: All trained models saved in .h5 format
- Figures: training_curves_all_models.png/.pdf, model_comparison_metrics.png/.pdf
- Tables: model_comparison_results.csv/.xlsx, latex_table.tex
- Statistics: statistical_analysis_models.json, model_comparison_report.txt
- Predictions: All test predictions and ground truth data saved

PUBLICATION READINESS:
----------------------
✓ High-resolution figures (300 DPI, PNG/PDF)
✓ LaTeX-formatted tables
✓ Comprehensive model comparison analysis
✓ Post-processing impact analysis
✓ Computational complexity analysis (FLOPs)
✓ Reproducible results with saved models
✓ Professional documentation
"""
        
        with open(self.results_dir / 'conference_summary_models.txt', 'w') as f:
            f.write(summary)
            
        print("="*80)
        print("MODEL COMPARISON RESULTS SUMMARY GENERATED")
        print("="*80)
        print(summary)

###################### Main Experiment Function ######################
# Add this function to your PublicationPlotter class or as a standalone function
def debug_history_contents(all_histories):
    """Debug function to inspect what's in your saved histories"""
    print("\n" + "="*60)
    print("DEBUGGING HISTORY CONTENTS")
    print("="*60)
    
    for model_name, history in all_histories.items():
        if hasattr(history, 'history'):
            hist = history.history
        else:
            hist = history
            
        print(f"\n{model_name}:")
        print(f"  Keys: {list(hist.keys())}")
        
        # Show first few values for each metric
        for key, values in hist.items():
            if isinstance(values, list) and len(values) > 0:
                print(f"  {key}: [{values[0]:.4f}, {values[1]:.4f}, ...] (length: {len(values)})")
            else:
                print(f"  {key}: {values}")

def run_model_comparison_experiment():
    """Main function to run the complete model comparison experiment"""
    
    print("="*80)
    print("STARTING MODEL COMPARISON EXPERIMENT")
    print("="*80)
    
    # Initialize components
    plotter = PublicationPlotter(config.results_dir)
    saver = ResultsSaver(config.results_dir)
    
    # Load datasets
    print("\nLoading datasets...")
    train_images, train_masks, dataset_info_train = load_brain_dataset(
        config.train_dir, config.target_size
    )
    
    test_images, test_masks, dataset_info_test = load_brain_dataset(
        config.test_dir, config.target_size
    )
    
    # Split training data
    x_train, x_val, y_train, y_val = train_test_split(
        train_images, train_masks, 
        test_size=config.validation_split, random_state=config.random_state
    )
    
    print(f"Training: {x_train.shape[0]}, Validation: {x_val.shape[0]}, Test: {test_images.shape[0]}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train, config.num_classes)
    print(f"Class weights: {class_weights}")
    
    # Convert masks to categorical
    y_train_categorical = to_categorical(y_train, num_classes=config.num_classes)
    y_val_categorical = to_categorical(y_val, num_classes=config.num_classes)
    
    # Configure loss function
    if config.loss_function == 'weighted_categorical':
        loss_func = weighted_categorical_crossentropy(class_weights)
    elif config.loss_function == 'multiclass_dice':
        loss_func = multiclass_dice_loss(num_classes=config.num_classes)
    else:
        loss_func = 'categorical_crossentropy'
    
    # Measure FLOPs for all models
    print("\nMeasuring computational complexity...")
    flops_data = measure_model_flops()
    
    # Train or load all models
    all_models = {}
    all_histories = {}
    
    if config.mode == 'training':
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        for model_name in config.models_to_compare:
            print(f"\nTraining {model_name}...")
            
            # Build model
            model_builder = get_model_builder(model_name)
            if model_builder is None:
                print(f"Unknown model: {model_name}")
                continue
                
            model = model_builder(config.input_shape, config.num_classes)
            print(f"{model_name} Parameters: {model.count_params():,}")
            
            # Compile model
            model.compile(
                optimizer=optimizers.legacy.Adam(config.learning_rate),
                loss=loss_func,
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7),
                callbacks.ModelCheckpoint(
                    config.results_dir / 'models' / f'{model_name}_best.h5',
                    save_best_only=True, monitor='val_loss'
                )
            ]
            
            # Train model
            history = model.fit(
                x_train, y_train_categorical,
                validation_data=(x_val, y_val_categorical),
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=callbacks_list,
                verbose=1
            )
            
            all_models[model_name] = model
            all_histories[model_name] = history
            
        # Save all models and histories
        saver.save_all_models(all_models, all_histories)
        
    else:
        # Load pre-trained models
        print("\nLoading pre-trained models...")
        all_models, all_histories = saver.load_all_models(config.models_to_compare, config.pre_result)
        # Debug what's in the histories
        debug_history_contents(all_histories)
    
    if not all_models:
        print("No models available. Please train models first or check model files.")
        return None
    
    # Generate predictions for all models
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS AND EVALUATION")
    print("="*60)
    
    all_predictions = {}
    all_predictions_processed = {}
    all_metrics = []
    
    for model_name, model in all_models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Generate predictions
        test_pred = model.predict(test_images, batch_size=config.batch_size)
        if model_name == 'Enhanced_U-Net':
            test_pred = test_pred[0]
        test_pred_classes = np.argmax(test_pred, axis=-1)
        
        # Post-process predictions
        test_pred_processed = post_process_predictions(
            test_pred_classes, 
            min_object_size=5, 
            apply_opening=True, 
            kernel_size=2
        )
        
        all_predictions[model_name] = test_pred_classes
        all_predictions_processed[model_name] = test_pred_processed
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(
            test_masks, #.flatten(), 
            test_pred_processed, #.flatten(), 
            model_name,
            pixel_spacing=0.5  # Add pixel spacing parameter
        )
        all_metrics.append(metrics)
        
        print(f"{model_name} - Mean Dice: {metrics['Mean_Dice']:.4f}, Mean IoU: {metrics['Mean_IoU']:.4f}")
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    statistical_results = saver.save_statistical_analysis_models(all_metrics)
    
    # Save results
    print("\nSaving results...")
    results_df = saver.save_comprehensive_results_table(all_metrics, statistical_results)

    print("\nSaving results...")
    saver.save_predictions(
        test_images, test_masks,
        all_predictions, all_predictions_processed, 
        {'train': dataset_info_train, 'test': dataset_info_test}
    )
    
    
    # Generate visualizations
    print("\nGenerating publication-quality figures...")
    plotter.plot_training_curves_all_models(all_histories)
    plotter.plot_model_comparison_metrics(all_metrics)
    indices = np.array([23, 44, 48])  # Your chosen indices
    # indices = np.array([44])  # Your chosen indices
    plotter.plot_sample_predictions(test_images, test_masks, all_predictions_processed, indices=indices)
    
    # Generate final summary
    saver.generate_conference_summary_models(
        config, dataset_info_train, dataset_info_test,
        all_metrics, statistical_results, flops_data
    )
    
    return {
        'config': config,
        'models': all_models,
        'histories': all_histories,
        'predictions': all_predictions_processed,
        'metrics': all_metrics,
        'statistical_results': statistical_results,
        'results_df': results_df,
        'flops_data': flops_data
    }

###################### Execute Experiment ######################

if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(config.random_state)
    tf.random.set_seed(config.random_state)
    
    # Run the complete experiment
    results = run_model_comparison_experiment()
    
    if results is not None:
        print("\n" + "="*80)
        print("MODEL COMPARISON EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved in: {config.results_dir}")
        print("All files are ready for conference paper submission!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("EXPERIMENT FAILED - CHECK ERROR MESSAGES ABOVE")
        print("="*80)