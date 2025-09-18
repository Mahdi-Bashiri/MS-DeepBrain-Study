# Repository Structure Documentation

## Overview

This document describes the implementation details and organizational structure of our second research article, which focuses on the **deep learning-based neuroanatomical profiling reveals detailed brain chnages: a large-scale multiple sclerosis study**.

Detailed methodology and results are provided in the accompanying files and can be referenced in our published article. This document specifically outlines the implementation structure and organization of the Python-based codebase for developing and maintaining this GitHub repository.

## Repository Structure

The main directory contains **7 folders** organized as follows:

```
├── Article_Figures/
├── Article_Tables/
├── Phase1_data_preprocessing/
├── Phase2_data_preparation_for_model_training/
├── Phase3_model_training_and_inferencing_and_evaluation/
├── Phase4_data_processing/
└── Phase5_statistical_analysis/
```

## Detailed Directory Structure

### Article_Figures/
Contains all figures used in the research article: **7 main figures**.

```
├── Figure_1.tif
├── Figure_2.tif
├── Figure_3.tif
├── Figure_4.tif
├── Figure_5.tif
├── Figure_6.tif
└── Figure_7.tif
```

### Article_Tables/
Contains all tables used in the research article: **3 main tables**.

```
├── Table_1.docx
├── Table_2.docx
└── Table_3.docx
```

The `raw_data` directory contains sample data from **5 patients** with both FLAIR and T1 sequences. Each method directory includes necessary documentation and explanatory files as needed.

### Phase1_data_preprocessing/
Contains preprocessing scripts for raw input files.

```
├── raw_data/
└── pre_processing_flair.py
```

The `raw_data` directory contains the same sample data as mentioned above. The Python script handles preprocessing of raw input files.

### Phase2_data_preparation_for_model_training/
Contains **4 data directories** and **1 Python script** for generating input images for the pix2pix (cGAN) model.

```
├── Original_FLAIRs_prep/
├── abWMH_manual_segmentations/
├── vent_manual_segmentations/
├── manual_3l_masks/
└── generating_3L_masks.py
```

**Directory descriptions:**
- **Original_FLAIRs_prep/**: Contains preprocessed patient data files. Each patient has one NIfTI file and one NPZ file containing the FLAIR image, brain mask, and mask metadata.
- **abWMH_manual_segmentations/**: Contains manual segmentation masks for abnormal lesions.
- **vent_manual_segmentations/**: Contains manual segmentation masks for brain ventricles.
- **manual_3l_masks/**: Contains generated 3-level masks from the above segmentations.

The Python script generates 4-level masks and creates paired images for pix2pix model training.

### Phase3_model_training_and_inferencing_and_evaluation/
Contains **3 directories** and **3 Python scripts**.

```
├── dataset_3l_man/
│   ├── test/
│   └── train/
├── chosen_model_performance/
├── trained_models/
├── inferencing_wmh_vent_unet_models_v3.py
└── training_wmh_vent_unet_models_v3.py
```

**Directory descriptions:**
- **test/** and **train/**: Contains training data for the model, with paired images in PNG format.
- **chosen_model_performance/**: Contains performance metrics and evaluation results.
- **trained_models/**: Contains the trained and saved model alongside all training config, log, models, predictions, statistics, figures, and tables.

The Python scripts handle model training, inference, and evaluation processes.

### Phase4_data_processing/
Contains **3 Python script** and **1 Excel file**.

```
├── brain_mri_analysis_results_PROCESSED_updated.xlsx
├── core_processing.py
├── excel_extractor.py
└── excel_filler_brain_TIA.py
```

**File descriptions:**
- **core_processing.py**: Produce detailed information regarding lesions and ventricles alongside subtype lesion masks.

- **excel_extractor.py**: Produce Final Excel or CSV data file for the cohort. 

- **excel_filler_brain_TIA.py**: Incorporate brain intracranial area data into extracted excel file.

- **brain_mri_analysis_results_PROCESSED_updated.xlsx**: Produced final Excel file containing all extracted information from ventricle and lesion segmentation masks and additional processed masks.

### Phase5_statistical_analysis/
Contains **2 directories** and **1 Python script**.

```
├── csv_analysis_outputs_no_outlier_v3/
├── csv_analysis_outputs_outlier_v3/
└── comprehensive_statistical_analysis_v3.py
```

**File descriptions:**
- **comprehensive_statistical_analysis_v3.py**: Performs comprehensive statistical analysis and handles visualizations, tables, and documenings. 

**Directory descriptions:**
- **csv_analysis_outputs_no_outlier_v3/**: Contains all produced results in different file formats based on not using outlier detection and removal.
- **csv_analysis_outputs_outlier_v3/**: Contains all produced results in different file formats based on using outlier detection and removal.

## Root Files

### our_article_DOI.md
Contains the BibTeX citation format for referencing our article and this repository.

### repo_explanation.docx
This explanatory document that describes the repository structure and organization.

## Implementation Framework

The entire implementation is developed in **Python** using the following key technologies:
- **Deep Learning Framework**: TensorFlow/Keras for pix2pix implementation
- **Image Processing**: OpenCV, Scikit-image
- **Data Handling**: NumPy, NIfTI processing libraries
- **Visualization**: Matplotlib, Seaborn
- **Statistical Analysis**: Scipy, Pandas

## Usage Instructions

1. **Data Preprocessing**: Start with Phase1 to preprocess raw FLAIR images
2. **Data Preparation**: Use Phase2 to generate training data for model training or inference
3. **Model Training**: Execute Phase3 scripts to train and evaluate the model
4. **Comparative Analysis**: Run Phase4 scripts to conduct comprehensive statistical analysis

## Citation

Please refer to `our_article_DOI.md` for proper citation format when using this repository or referencing our work.