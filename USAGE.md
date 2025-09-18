# Usage Tutorial

## Overview

This tutorial provides step-by-step instructions for using the Deep Learning-Based Neuroanatomical Profiling Reveals Detailed Brain Changes: A Large-Scale Multiple Sclerosis Study. The pipeline consists of four main phases:

1. **Phase 1**: Data Preprocessing
2. **Phase 2**: Data Preparation for Model Training
3. **Phase 3**: Model Training/Inference/Evaluation
4. **Phase 4**: Core Data Processing
4. **Phase 5**: Comprehensive Statistical Analysis

## Quick Start (Inference Only)

### Prerequisites
- FLAIR MRI images in NIfTI format (.nii or .nii.gz)
- Completed installation (see Installation Guide)

### Basic Usage

```python
import sys
sys.path.append('src')

from preprocessing.pre_processing_flair import main as preprocess_flair
from models.inferring import main as run_inference

# Step 1: Preprocess FLAIR image
input_path = "path/to/your/flair.nii.gz"
output_dir = "path/to/preprocessed/output"

preprocessed_image, brain_info = preprocess_flair(input_path, output_dir)

# Step 2: Run inference
model_path = "src/models/pix2pix_generator_4L"
results_dir = "path/to/results"

segmentation_results = run_inference(
    preprocessed_image,
    brain_info, 
    model_path, 
    results_dir
)
```

## Detailed Phase-by-Phase Tutorial

### Phase 1: Data Preprocessing

**Purpose**: Clean and normalize raw FLAIR MRI images for processing.

**Location**: `Phase1_data_preprocessing/`

#### Step 1.1: Prepare Raw Data

```bash
# Place your FLAIR images in the raw_data directory
mkdir -p Phase1_data_preprocessing/raw_data/subjects_flair
cp your_flair_images/* Phase1_data_preprocessing/raw_data/subjects_flair/
```

#### Step 1.2: Run Preprocessing

```python
# Navigate to Phase1 directory
cd Phase1_data_preprocessing

# Run preprocessing script
python pre_processing_flair.py
```

**What this does**:
- Noise reduction using median and Gaussian filters
- Brain extraction with morphology-based approach
- Intensity normalization (slice-based adaptive)
- Generates .npz files with brain masks and metadata

**Expected Output**:
```
preprocessed_output/
├── patient_001.nii.gz     # Preprocessed FLAIR
├── patient_001.npz        # Brain mask + metadata
├── patient_002.nii.gz
├── patient_002.npz
└── ...
```

### Phase 2: Data Preparation for Model Training

**Purpose**: Generate paired images for pix2pix model training (only needed for training new models).

**Location**: `Phase2_data_preparation_for_model_training/`

#### Step 2.1: Organize Manual Segmentations (Training Only)

If you have ground truth segmentations:

```bash
# Copy segmentations to appropriate directories
cp ventricle_masks/* vent_manual_segmentations/
cp abnormal_wmh_masks/* abWMH_manual_segmentations/
cp preprocessed_flairs/* Original_FLAIRs_prep/
```

#### Step 2.2: Generate Training Data

```python
# Generate 4-level masks and paired images
python generating_4L_masks.py
```

**Output**: Creates paired 256×512 composite images for pix2pix training.

### Phase 3: Model Training, Inference, and Evaluation

**Purpose**: Train models, run inference, and evaluate performance.

**Location**: `Phase3_model_training_and_inferencing_and_evaluation/`

#### Step 3.1: Inference with Pre-trained Model

Use the Python script:

```python
# Run inference
python inferencing_wmh_vent_unet_models_v3.py 
                --input_dir "path/to/preprocessed/data" \
                --model_path "pix2pix_generator_4L" \
                --output_dir "path/to/results"
```

#### Step 3.2: Model Training (Optional)

```python
# If training a new model
training_wmh_vent_unet_models_v3.py
```

**Training Parameters**:
- Batch size: 8
- Epochs: 30
- Optimizer: Adam (lr=0.0001)
- Loss: Unified Focal

#### Step 3.3: Evaluation

Use training script with training mode switched off through config class.

```python
# Run evaluation with parallelization
python training_wmh_vent_unet_models_v3.py 
                                --predictions_dir "path/to/predictions" \
                                --ground_truth_dir "path/to/ground_truth" \
                                --output_dir "path/to/evaluation_results"
```

**Evaluation Metrics**:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Hausdorff Distance (HD95)
- Precision
- Recall

### Phase 4: Data Processing

**Purpose**: Perform lesion/ventricle characterization.

**Location**: `Phase4_data_processing/`

#### Step 4.1: Core mask processing

```python
# Produce detailed information regarding lesions and ventricles alongside subtype lesion masks 
python core_processing.py
```

#### Step 4.2: Prepare Excel Data

```python
# Produce Final Excel or CSV data file for the cohort
python excel_extractor.py
```

#### Step 4.3: Add more information to the Excel file

```python
# Incorporate brain intracranial area data into extracted excel file
python excel_filler_brain_TIA.py
```

### Phase 5: Comprehensive Statistical Analysis

**Purpose**: Compare results against baseline methods.

**Location**: `Phase4_statistical_analysis/`

#### Step 5.1: Excel Data

```bash
# Ensure your data is in the correct format
ls Phase4_data_processing/  # Should contain Excel or CSV file
```

#### Step 5.2: Run Statistical Analysis

```python
# Run comprehensive analysis
cd Phase5_statistical_analysis
python comprehensive_statistical_analysis_v3.py
```

## Input Data Requirements

### FLAIR MRI Specifications
- **Format**: NIfTI (.nii or .nii.gz)
- **Orientation**: Standard radiological orientation
- **Resolution**: Compatible with anisotropic voxels (tested on 0.9 × 0.9 × 6 mm)
- **Scanner**: Any 1.5T or 3T scanner (tested on 1.5T TOSHIBA Vantage)

### File Naming Convention
```
patient_001_FLAIR.nii.gz
patient_002_FLAIR.nii.gz
...
```

## Output Interpretation

### Segmentation Results

The model outputs a 4-class segmentation:

- **Class 0**: Background
- **Class 1**: Brain ventricles
- **Class 2**: Normal white matter hyperintensities
- **Class 3**: Abnormal white matter hyperintensities (MS lesions)

### Result Files

```
results/
# Original FLAIR
# Preprocessed FLAIR
# 3-class segmentation
# Ventricles mask
# Abnormal WMH mask
# Performance metrics
# Processed Excel data
# Statistical Results
...
```

## Processing Time

**Expected processing times per patient**:
- Preprocessing: ~4 seconds
- Inference: ~1 seconds
- Total: ~5 seconds

## Clinical Interpretation

### Ventricle Segmentation
- **Volume measurements**: Quantify ventricular enlargement
- **Asymmetry assessment**: Compare left vs right ventricles
- **Longitudinal tracking**: Monitor atrophy progression

### White Matter Hyperintensity Classification
- **Abnormal WMH**: MS lesions, pathological hyperintensities
- **Clinical significance**: Distinguish pathology from normal aging

## Visualization

```python
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def visualize_results(flair_path, segmentation_path, slice_idx=None):
    """Visualize FLAIR image with segmentation overlay"""
    
    # Load images
    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(segmentation_path).get_fdata()
    
    # Select middle slice if not specified
    if slice_idx is None:
        slice_idx = flair.shape[2] // 2
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # FLAIR image
    axes[0].imshow(flair[:, :, slice_idx], cmap='gray')
    axes[0].set_title('FLAIR Image')
    axes[0].axis('off')
    
    # Segmentation
    axes[1].imshow(seg[:, :, slice_idx], cmap='viridis')
    axes[1].set_title('4-Class Segmentation')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(flair[:, :, slice_idx], cmap='gray', alpha=0.7)
    axes[2].imshow(seg[:, :, slice_idx], cmap='jet', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
visualize_results("results/patient_001/preprocessed_flair.nii.gz", 
                 "results/patient_001/segmentation_4class.nii.gz")
```

## Performance Monitoring

Monitor processing with progress tracking:

```python
from tqdm import tqdm
import time

def process_with_progress(file_list, process_func):
    """Process files with progress bar"""
    
    results = []
    for file_path in tqdm(file_list, desc="Processing files"):
        start_time = time.time()
        result = process_func(file_path)
        processing_time = time.time() - start_time
        
        results.append({
            'file': file_path,
            'result': result,
            'time': processing_time
        })
        
        tqdm.write(f"Processed {file_path} in {processing_time:.2f}s")
    
    return results
```

## Next Steps

After completing this tutorial:

1. **Analyze Results**: Review segmentation quality and metrics
2. **Clinical Validation**: Have expert review clinical accuracy
3. **Comparison Analysis**: Compare with baseline methods
4. **Integration**: Integrate into clinical workflow
5. **Troubleshooting**: Refer to troubleshooting guide for issues

## Support

For additional help:
- Check the **Troubleshooting Guide** for common issues
- Review example outputs in the `results/` directory
- Create GitHub issues for specific problems