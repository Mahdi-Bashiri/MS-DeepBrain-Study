"""
Brain Segmentation Inference Script
===================================
Inference script for brain FLAIR MRI segmentation using pre-trained models
Focuses on Attention U-Net with Unified Focal Loss for Abnormal WMH and Ventricles segmentation

Author: Mahdi Bashiri Bawil
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
from pathlib import Path

# Deep Learning
import tensorflow as tf
import keras
from keras.models import load_model

# Image processing
from scipy.ndimage import binary_dilation
from skimage.morphology import disk, remove_small_objects, binary_opening

print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())
print("Built with CUDA: ", tf.test.is_built_with_cuda())
print("Physical devices: ", tf.config.list_physical_devices())

# Force GPU if available
if tf.config.list_physical_devices('GPU'):
    print("\n\n\t\t\tUsing GPU\n\n")
else:
    print("\n\n\t\t\tUsing CPU\n\n")


###################### Inference Configuration ######################

class InferenceConfig:
    """Configuration class for inference"""
    def __init__(self):
        # Model and data paths
        self.header_dir = "/mnt/e/MBashiri/Thesis/p4"
        # self.header_dir = "E:\MBashiri\Thesis\p4"
        self.model_dir = os.path.join(self.header_dir, "brain_segmentation_models_20250909_034626_unified_focal_loss")  # Change this
        self.model_name = "Attention_U-Net"  # Change this to your preferred model
        self.inference_data_dir = "Stats_article_data/new_data_images"  # New data directory
        self.inference_data_dir = os.path.join(self.header_dir, "HC_COHORT_IMAGES_2")  # New data directory: HC or MS cohorts
        
        # Model parameters (should match training config)
        self.input_shape = (256, 256, 1)
        self.target_size = (256, 256)
        self.num_classes = 3  # Background (0), Ventricles (1), Abnormal WMH (2)
        
        # Inference parameters
        self.batch_size = 8
        self.save_visualizations = True
        self.save_numpy_masks = True
        self.apply_postprocessing = True
        
        # Post-processing parameters
        self.min_object_size = 5
        self.opening_kernel_size = 1
        
        # Create results directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"{self.header_dir}/inference_results_{self.model_name}_{self.timestamp}")
        
        self.create_directory_structure()
        
    def create_directory_structure(self):
        """Create directory structure for inference results"""
        subdirs = [
            'raw_predictions',
            'processed_predictions', 
            'visualizations',
            'numpy_masks',
            'logs'
        ]
        
        self.results_dir.mkdir(exist_ok=True)
        for subdir in subdirs:
            (self.results_dir / subdir).mkdir(exist_ok=True)
            
        # Save inference configuration
        config_dict = {
            'timestamp': self.timestamp,
            'model_dir': str(self.model_dir),
            'model_name': self.model_name,
            'inference_data_dir': self.inference_data_dir,
            'input_shape': self.input_shape,
            'target_size': self.target_size,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'min_object_size': self.min_object_size,
            'opening_kernel_size': self.opening_kernel_size
        }
        
        with open(self.results_dir / 'logs' / 'inference_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

###################### Inference Data Loading ######################

def load_inference_dataset(data_dir, target_size=(256, 256)):
    """
    Load inference dataset (new data without ground truth masks)
    Returns images, filenames, and metadata
    """
    images = []
    filenames = []
    
    # Filter files by slice range (matching your training data selection)
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.tif')]
    image_files = [f for f in all_files]
    
    print(f"Found {len(image_files)} inference images in slice range 8-15")
    
    dataset_info = {
        'total_files_in_dir': len(all_files),
        'selected_files': len(image_files),
        'loaded_files': 0,
        'failed_files': [],
        'image_shapes': [],
        'filenames': []
    }
    
    for img_name in tqdm(image_files, desc="Loading inference images"):
        try:
            # Load image
            full_img = cv.imread(os.path.join(data_dir, img_name), cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE).astype(np.float32)
            
            if full_img is None:
                dataset_info['failed_files'].append(img_name)
                continue
                
            # For inference data, we expect single images (not concatenated with GT)
            # If your inference images are concatenated like training data, extract FLAIR part:
            if full_img.shape[1] == 512:
                flair_img = full_img[:, :256]  # Extract FLAIR part
                print(f"Note: {img_name} appears to be concatenated format, extracting FLAIR part")
            else:
                flair_img = full_img  # Single FLAIR image
            
            # Resize if needed
            if target_size != (256, 256):
                flair_img = cv.resize(flair_img, target_size)
            
            dataset_info['image_shapes'].append(flair_img.shape)
            
            # Normalize FLAIR image (same as training)
            flair_img = flair_img.astype(np.float32)
            flair_img = (flair_img - np.mean(flair_img)) / (np.std(flair_img) + 1e-7)
            flair_img = np.expand_dims(flair_img, axis=-1)
            
            images.append(flair_img)
            filenames.append(img_name)
            dataset_info['loaded_files'] += 1
            dataset_info['filenames'].append(img_name)
            
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            dataset_info['failed_files'].append(img_name)
    
    return np.array(images), filenames, dataset_info

###################### Model Loading ######################

def load_inference_model(model_dir, model_name):
    """Load pre-trained model for inference"""
    model_path = Path(model_dir) / 'models'
    
    # Try different possible model file naming conventions
    possible_names = [
        f"{model_name.replace('-', '_').replace(' ', '_').lower()}_model.h5",
        f"{model_name}_best.h5",
        f"{model_name.replace(' ', '_')}_model.h5"
    ]
    
    for model_filename in possible_names:
        full_path = model_path / model_filename
        if full_path.exists():
            try:
                print(f"Loading model from: {full_path}")
                
                # Load with custom objects if needed
                custom_objects = {}
                
                # ADD YOUR CUSTOM LOSS FUNCTIONS TO custom_objects HERE
                # custom_objects['unified_focal_loss'] = unified_focal_loss([1.0, 3.5, 2.8])
                # etc.
                
                model = keras.models.load_model(full_path, custom_objects=custom_objects, compile=False)
                print(f"Successfully loaded {model_name}")
                print(f"Model input shape: {model.input_shape}")
                print(f"Model output shape: {model.output_shape}")
                return model
                
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                continue
    
    # If no model found, try to build and load weights
    print(f"Could not find saved model. Trying to build {model_name} and load weights...")
    
    # ADD MODEL BUILDING CODE HERE IF NEEDED
    # model_builder = get_model_builder(model_name)
    # if model_builder:
    #     model = model_builder(input_shape, num_classes)
    #     # Try to load weights
    #     weights_path = model_path / f"{model_name}_weights.h5"
    #     if weights_path.exists():
    #         model.load_weights(weights_path)
    #         return model
    
    raise FileNotFoundError(f"Could not load model {model_name} from {model_dir}")

###################### Visualization Functions ######################

def create_segmentation_visualization(original_image, raw_prediction, processed_prediction, filename):
    """Create visualization showing original image and both predictions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Class colors for visualization
    class_colors = {0: [0, 0, 0], 1: [0, 0, 1], 2: [1, 0, 0]}  # Background, Ventricles, WMH
    
    # Original image
    axes[0].imshow(original_image.squeeze(), cmap='gray')
    axes[0].set_title('Original FLAIR')
    axes[0].axis('off')
    
    # Raw prediction
    raw_colored = np.zeros((*raw_prediction.shape, 3))
    for class_id, color in class_colors.items():
        mask = raw_prediction == class_id
        raw_colored[mask] = color
    
    axes[1].imshow(raw_colored)
    axes[1].set_title('Raw Prediction')
    axes[1].axis('off')
    
    # Processed prediction
    processed_colored = np.zeros((*processed_prediction.shape, 3))
    for class_id, color in class_colors.items():
        mask = processed_prediction == class_id
        processed_colored[mask] = color
    
    axes[2].imshow(processed_colored)
    axes[2].set_title('Post-processed\n(Red: WMH, Blue: Ventricles)')
    axes[2].axis('off')
    
    plt.suptitle(f'Segmentation Results - {filename}', fontsize=14)
    plt.tight_layout()
    
    return fig

def save_prediction_masks(raw_prediction, processed_prediction, filename, save_dir):
    """Save prediction masks as image files"""
    # Create colored masks for visualization
    class_colors = {0: 0, 1: 127, 2: 255}  # Background, Ventricles, WMH (grayscale values)
    
    # Raw prediction mask
    raw_mask = np.zeros_like(raw_prediction, dtype=np.uint8)
    for class_id, color in class_colors.items():
        mask = raw_prediction == class_id
        raw_mask[mask] = color
    
    # Processed prediction mask  
    processed_mask = np.zeros_like(processed_prediction, dtype=np.uint8)
    for class_id, color in class_colors.items():
        mask = processed_prediction == class_id
        processed_mask[mask] = color
    
    # Save masks
    base_name = filename.replace('.png', '').replace('.tif', '')
    
    raw_path = save_dir / f"{base_name}_raw_mask.png"
    processed_path = save_dir / f"{base_name}_processed_mask.png"
    
    cv.imwrite(str(raw_path), raw_mask)
    cv.imwrite(str(processed_path), processed_mask)
    
    return raw_path, processed_path

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
                if class_id==1:
                    pass           # Skip the ventricles from the operatrion
                else:
                    kernel = disk(kernel_size)
                    class_mask = binary_opening(class_mask, kernel)
            
            # Add processed class back to mask
            post_processed[i][class_mask] = class_id
    
    return post_processed

###################### Main Inference Function ######################

def run_inference():
    """Main inference function"""
    
    print("="*80)
    print("BRAIN SEGMENTATION INFERENCE")
    print("="*80)
    
    # Initialize configuration
    config = InferenceConfig()
    
    print(f"Model: {config.model_name}")
    print(f"Model Directory: {config.model_dir}")
    print(f"Inference Data: {config.inference_data_dir}")
    print(f"Results Directory: {config.results_dir}")
    
    # Load inference dataset
    print("\nLoading inference dataset...")
    inference_images, filenames, dataset_info = load_inference_dataset(
        config.inference_data_dir, config.target_size
    )
    
    print(f"Loaded {len(inference_images)} images for inference")
    
    if len(inference_images) == 0:
        print("No images found for inference. Check your data directory and file format.")
        return
    
    # Load model
    print(f"\nLoading model: {config.model_name}")
    model = load_inference_model(config.model_dir, config.model_name)
    
    # Run inference
    print(f"\nRunning inference on {len(inference_images)} images...")
    raw_predictions = model.predict(inference_images, batch_size=config.batch_size, verbose=1)
    
    # Handle multiple outputs (e.g., Enhanced U-Net has auxiliary outputs)
    if isinstance(raw_predictions, list):
        raw_predictions = raw_predictions[0]  # Take main output
    
    # Convert to class predictions
    raw_pred_classes = np.argmax(raw_predictions, axis=-1)
    
    # Apply post-processing if requested
    if config.apply_postprocessing:
        print("Applying post-processing...")
        processed_predictions = post_process_predictions(
            raw_pred_classes,
            min_object_size=config.min_object_size,
            apply_opening=True,
            kernel_size=config.opening_kernel_size
        )
    else:
        processed_predictions = raw_pred_classes.copy()
    
    # Save results
    print("\nSaving results...")
    results_summary = []
    
    for i, (image, raw_pred, proc_pred, filename) in enumerate(
        zip(inference_images, raw_pred_classes, processed_predictions, filenames)):
        
        # Calculate basic statistics
        unique_raw, counts_raw = np.unique(raw_pred, return_counts=True)
        unique_proc, counts_proc = np.unique(proc_pred, return_counts=True)
        
        stats = {
            'filename': filename,
            'image_index': i,
            'total_pixels': raw_pred.size,
            'raw_background_pixels': counts_raw[unique_raw == 0][0] if 0 in unique_raw else 0,
            'raw_ventricles_pixels': counts_raw[unique_raw == 1][0] if 1 in unique_raw else 0,
            'raw_wmh_pixels': counts_raw[unique_raw == 2][0] if 2 in unique_raw else 0,
            'proc_background_pixels': counts_proc[unique_proc == 0][0] if 0 in unique_proc else 0,
            'proc_ventricles_pixels': counts_proc[unique_proc == 1][0] if 1 in unique_proc else 0,
            'proc_wmh_pixels': counts_proc[unique_proc == 2][0] if 2 in unique_proc else 0,
        }
        
        results_summary.append(stats)
        
        # Save numpy masks if requested
        if config.save_numpy_masks:
            base_name = filename.replace('.png', '').replace('.tif', '')
            np.save(config.results_dir / 'numpy_masks' / f"{base_name}_raw_prediction.npy", raw_pred)
            np.save(config.results_dir / 'numpy_masks' / f"{base_name}_processed_prediction.npy", proc_pred)
        
        # Save image masks
        save_prediction_masks(raw_pred, proc_pred, filename, config.results_dir / 'processed_predictions')
        
        # Create and save visualizations if requested
        if config.save_visualizations:
            fig = create_segmentation_visualization(image, raw_pred, proc_pred, filename)
            viz_name = filename.replace('.png', '').replace('.tif', '')
            fig.savefig(config.results_dir / 'visualizations' / f"{viz_name}_segmentation.png", 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # Save results summary
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(config.results_dir / 'inference_results_summary.csv', index=False)
    results_df.to_excel(config.results_dir / 'inference_results_summary.xlsx', index=False)
    
    # Save dataset info
    with open(config.results_dir / 'logs' / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Generate summary report
    total_pixels = results_df['total_pixels'].sum()
    total_ventricles = results_df['proc_ventricles_pixels'].sum()
    total_wmh = results_df['proc_wmh_pixels'].sum()
    
    summary_report = f"""
BRAIN SEGMENTATION INFERENCE SUMMARY
====================================
Timestamp: {config.timestamp}
Model: {config.model_name}
Model Directory: {config.model_dir}
Inference Data: {config.inference_data_dir}

DATASET STATISTICS:
------------------
Total images processed: {len(inference_images)}
Failed to load: {len(dataset_info['failed_files'])}
Image size: {config.target_size}

SEGMENTATION RESULTS:
--------------------
Total pixels analyzed: {total_pixels:,}
Ventricles pixels detected: {total_ventricles:,} ({100*total_ventricles/total_pixels:.2f}%)
Abnormal WMH pixels detected: {total_wmh:,} ({100*total_wmh/total_pixels:.2f}%)

POST-PROCESSING APPLIED:
-----------------------
Minimum object size: {config.min_object_size} pixels
Morphological opening: Yes (kernel size: {config.opening_kernel_size})

OUTPUT FILES GENERATED:
----------------------
- Processed prediction masks: {len(filenames)} files in /processed_predictions/
- Raw prediction masks: {len(filenames)} files in /processed_predictions/
- Visualization images: {len(filenames)} files in /visualizations/
- Numpy masks: {len(filenames)*2} files in /numpy_masks/
- Results summary: inference_results_summary.csv/xlsx

FILES WITH DETECTED ABNORMALITIES:
---------------------------------"""
    
    # List files with detected abnormalities
    abnormal_files = results_df[results_df['proc_wmh_pixels'] > 0]['filename'].tolist()
    if abnormal_files:
        summary_report += f"\nFiles with WMH detected ({len(abnormal_files)}):\n"
        for filename in abnormal_files:
            wmh_pixels = results_df[results_df['filename'] == filename]['proc_wmh_pixels'].iloc[0]
            summary_report += f"  - {filename}: {wmh_pixels} WMH pixels\n"
    else:
        summary_report += "\nNo abnormal WMH detected in any image.\n"
    
    # Save summary report
    with open(config.results_dir / 'inference_summary.txt', 'w') as f:
        f.write(summary_report)
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved in: {config.results_dir}")
    print(f"Processed {len(inference_images)} images")
    print(f"Detected abnormalities in {len(abnormal_files)} images")
    print("="*80)
    
    return {
        'config': config,
        'results_summary': results_df,
        'dataset_info': dataset_info,
        'raw_predictions': raw_pred_classes,
        'processed_predictions': processed_predictions,
        'filenames': filenames
    }

###################### Execute Inference ######################

if __name__ == "__main__":
    # Run inference
    results = run_inference()
    
    if results is not None:
        print(f"\nInference results available in: {results['config'].results_dir}")
        print("Check the visualizations folder to review model performance!")
