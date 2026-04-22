# Deep Learning-Based Neuroanatomical Profiling Reveals Population-Specific Brain Changes in Multiple Sclerosis: A Large-Scale Middle Eastern Study

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.11+](https://img.shields.io/badge/TensorFlow-2.11+-orange.svg)](https://tensorflow.org/)
[![Medical Imaging](https://img.shields.io/badge/domain-Medical%20Imaging-green.svg)](https://github.com/topics/medical-imaging)
[![Multiple Sclerosis](https://img.shields.io/badge/application-Multiple%20Sclerosis-red.svg)](https://github.com/topics/multiple-sclerosis)
[![Models on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/Bawil/MS-DeepBrain-Study/upload/main)


## 🧠 Overview

This repository implements a large-scale neuroanatomical profiling study of Multiple Sclerosis using deep learning-based automated segmentation. Our comprehensive analysis of **1,381 subjects** from Northwest Iran provides detailed statistical characterization of brain structural changes in an underrepresented Middle Eastern population.

### 🎯 Key Contributions

- **🔬 Large-Scale MS Neuroimaging Study**: 1,381 participants (381 MS patients, 1,000 healthy controls)
- **🌍 Population-Specific Research**: Addresses gap in Middle Eastern MS neuroimaging research
- **🤖 Comparative Architecture Evaluation**: Four deep learning models evaluated via 4-fold cross-validation (U-Net, Attention U-Net, Trans-U-Net, DeepLabV3Plus)
- **⚡ Optimal Model Selection**: U-Net achieved superior performance (DSC=88.8%±0.5%, HD95=2.8±0.0mm, L-F1=77.4±1.4%)
- **📊 Comprehensive Statistics**: Multi-dimensional analysis across age, gender, and anatomical regions
- **🎯 Clinical Translation**: Population-specific normative values for MS biomarkers in Middle Eastern populations

### 📈 Key Clinical Findings

| Metric | MS Patients | Healthy Controls | Statistical Significance |
|--------|-------------|------------------|-------------------------|
| **WMH Lesion Burden** | ~3-fold higher (age-stratified) | Baseline | p<0.001 |
| **Periventricular Predominance** | 53.91±20.62% | Minimal | Consistent across all age groups |
| **Age-Related WMH Progression** | 0.315% → 0.749% (18–29 to 50–59 yrs) | Stable (r=0.085) | r=0.224 (MS) vs r=0.085 (HC) |
| **Ventricular Area** | 3,812.7±1,791.1 mm² | 2,559.6±946.9 mm² | Cohen's d=1.012 |
| **Gender Differences (Lesion Subtypes)** | No significant difference | — | n.s. after Bonferroni correction |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (RTX 3060 or equivalent)
- 64GB+ RAM (recommended for large-scale processing)
- FLAIR MRI sequences in NIfTI format

### Installation

```bash
# Clone the repository
git clone https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study.git
cd MS-DeepBrain-Study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from Phase3_model_training_and_inferencing_and_evaluation import inferencing_wmh_vent_unet_models_v3
from Phase4_data_processing import core_processing
from Phase5_statistical_analysis import comprehensive_statistical_analysis_v3

# Run automated segmentation with U-Net (optimal model)
results = inferencing_wmh_vent_unet_models_v3.main(
    input_dir="path/to/flair/images",
    model_path="trained_models/unet_model",
    output_dir="results/"
)

# Process and extract quantitative metrics
processed_data = core_processing.main(results)

# Perform comprehensive statistical analysis
statistical_results = comprehensive_statistical_analysis_v3.main(processed_data)
```

---

## 🗂️ Repository Structure

The repository follows a **5-phase pipeline architecture**:

```
├── 📁 Article_Figures/              # 7 main figures from published article
│   ├── Figure_1.tif                 # Sample FLAIR annotations
│   ├── Figure_2.tif                 # Study population demographics
│   ├── Figure_3.tif                 # Ventricular burden analysis
│   ├── Figure_4.tif                 # Age-related burden analysis
│   ├── Figure_5.tif                 # Lesion burden analysis
│   ├── Figure_6.tif                 # Anatomical lesion distribution
│   └── Figure_7.tif                 # Statistical correlation matrices
├── 📁 Article_Tables/               # 3 comprehensive tables
│   ├── Table_1.docx                # Demographic characteristics
│   ├── Table_2.docx                # Deep learning model performance
│   └── Table_3.docx                # Lesion subtype statistical comparison
├── 📁 Phase1_data_preprocessing/    # FLAIR image preprocessing
│   ├── 📁 raw_data/                # Sample data (5 patients)
│   └── pre_processing_flair.py     # 4-step preprocessing pipeline
├── 📁 Phase2_data_preparation_for_model_training/
│   ├── 📁 Original_FLAIRs_prep/    # Preprocessed FLAIR images
│   ├── 📁 abWMH_manual_segmentations/ # Manual abnormal WMH masks
│   ├── 📁 vent_manual_segmentations/  # Manual ventricle masks
│   ├── 📁 manual_3l_masks/         # Generated 3-class masks
│   └── generating_3L_masks.py      # Training data generation
├── 📁 Phase3_model_training_and_inferencing_and_evaluation/
│   ├── 📁 dataset_3l_man/          # Training/validation/testing datasets
│   ├── 📁 model_performance/       # Performance metrics for all 4 architectures
│   ├── 📁 trained_models/          # Pre-trained models (all 4 architectures)
│   │   ├── unet_model/             # ⭐ Optimal model (DSC=88.8%, L-F1=77.4%)
│   │   ├── attention_unet_model/
│   │   ├── trans_unet_model/
│   │   └── deeplabv3plus_model/
│   ├── training_wmh_vent_unet_models_v3.py    # Model training
│   └── inferencing_wmh_vent_unet_models_v3.py    # Automated inference
├── 📁 Phase4_data_processing/       # Quantitative analysis
│   ├── brain_mri_analysis_results_ALL.xlsx
│   ├── core_processing.py          # Neuroanatomical lesion classification
│   ├── excel_extractor.py          # Data extraction
│   └── excel_filler_brain_TIA.py    # Brain area normalization
├── 📁 Phase5_statistical_analysis/    # Comprehensive statistics
│   ├── 📁 csv_analysis_outputs_no_outlier_v4/    # Statistical results
│   ├── brain_mri_analysis_results_ALL.csv          # Extracted Data from All Cohort for Analyses
│   ├── Patient_Flair_data_HC.csv   # Demography of Healthy Controls 
│   ├── Patient_Flair_data_MS.csv   # Demography of MS Patients
│   └── p4_excel_analysis_developed.py    # Statistical pipeline
├── 📄 our_article_DOI.md           # Citation information
├── 📄 repo_explanation.docx        # Detailed methodology
└── 📄 README.md                    # This file
```

---

## 🔬 Methodology

### Study Population

- **Total Participants**: 1,381 subjects from Northwest Iran (2021-2024)
  - **MS Patients**: 381 subjects (71.7% female, 28.3% male)
  - **Healthy Controls**: 1,000 subjects (67.4% female, 32.6% male)
- **Age Range**: 18-74 years (HC), 18-68 years (MS)
- **Mean Age**: HC males: 34.8±10.8 years, HC females: 34.7±9.8 years; MS: 37.4±10.2 years
- **Location**: Golghasht Medical Imaging Center, Tabriz, Iran
- **Ethics Approval**: Tabriz University of Medical Sciences Research Ethics Committee (IR.TBZMED.REC.1402.902)

### Deep Learning Architecture Evaluation

**Comparative Analysis of 4 Architectures (4-fold cross-validation):**

| Model | Overall DSC (%) | HD95 (mm) | L-F1 (%) | Selection |
|-------|----------------|-----------|----------|-----------|
| **U-Net** | **88.8±0.5** | **2.8±0.0** | **77.4±1.4** | ⭐ **Optimal** |
| **Attention U-Net** | — | — | — | Evaluated |
| **Trans-U-Net** | — | — | — | Evaluated |
| **DeepLabV3Plus** | — | — | — | Evaluated |

> All architectures demonstrated comparable performance. U-Net achieved superior overall DSC and HD95 and was selected for population-level analysis of the complete 1,381-subject cohort.

**U-Net Implementation:**
- **Architecture**: Symmetric encoder-decoder with skip connections
- **Input**: Single-modality FLAIR sequences (256×256 pixels)
- **Output**: 3-class segmentation (background, ventricles, WMH)
- **Processing Speed**: 38ms per image (inference)
- **Training Details**: 100 epochs with hybrid loss function strategy

### Preprocessing Pipeline

1. **Noise Reduction**: 3×3 median filter + Gaussian smoothing (σ=1.0)
2. **Intensity Normalization**: Slice-wise z-score standardization
3. **Dimension Standardization**: Isotropic resampling to 1×1 mm² pixels
4. **Matrix Standardization**: Resizing to 256×256 pixels

### Neuroanatomical Lesion Classification

**Distance-Based Classification Criteria:**
- **PEWMH** (Periventricular): ≤3mm from ventricular surface
- **JCWMH** (Juxtacortical): ≤3mm from gray-white matter junction, area ≤20mm²
- **DWMH** (Deep White Matter): Remaining pathological hyperintensities not meeting periventricular or juxtacortical criteria (deep white matter parenchyma)

---

## 📊 Key Clinical Findings

### Model Performance

**U-Net Achieved Optimal Performance (4-fold cross-validation):**
- **Dice Similarity Coefficient**: 88.8±0.5% (overall)
- **Hausdorff Distance (95th percentile)**: 2.8±0.0mm
- **Lesion-wise F1-score**: 77.4±1.4%

### Lesion Burden Analysis

- **MS vs HC Comparison**: MS patients exhibited substantially elevated WMH area relative to healthy controls
- **Age-Related Progression**:
  - MS WMH ratio: 0.315% (18–29 years) → 0.749% (50–59 years) (~2.4-fold increase)
  - HC: Stable across age groups (r=0.085, negligible correlation)
- **MS age correlation**: r=0.224 (moderate), HC: r=0.085 (negligible)

### Anatomical Distribution

- **Periventricular Predominance**: 53.91±20.62% of total lesion burden in MS patients
  - Age-group means rising from 49.15% (18–29 years) to 63.49% (50–59 years)
- **Gender-Specific Patterns**:
  - No statistically significant differences in lesion localization for any subtype after Bonferroni correction (α=0.017)
- **Deep White Matter (DWMH)**: Second-largest contribution
- **Juxtacortical (JCWMH)**: Smallest contribution

### Ventricular Burden

- **MS patients showed significantly higher ventricular area than HC**: 3,812.7±1,791.1 mm² vs. 2,559.6±946.9 mm² (Cohen's d=1.012)
- **Brain-normalized ventricular ratio**: ~1.7-fold higher in MS vs. HC
- **Age-related correlations**:
  - HC: r=0.207 (weak positive)
  - MS: r=0.403 (moderate positive)
- **Pattern**: Progressive enlargement with age in both groups, accelerated in MS

### Statistical Correlations (Spearman)

- **HC (n=1,000)**: Age significantly correlated only with ventricular ratio (r=0.199, p=2.18×10⁻¹⁰) after Bonferroni correction
- **MS (n=381)**: Age correlated significantly with ventricular ratio (r=0.403, p=2.50×10⁻¹⁶), total WMH ratio (r=0.266), and periventricular WMH (positive); deep WMH showed significant negative correlation with age
- Juxtacortical WMH showed no significant age association (r=−0.046, p=0.37)

### Population-Specific Context

- **Regional Prevalence**: Iran shows elevated MS prevalence compared to global average
- **Study Contribution**: Large-scale neuroanatomical characterization in Middle Eastern population providing preliminary population-specific reference ranges
- **Clinical Utility**: Population-specific normative WMH and ventricular ratio values for MS monitoring in this underrepresented group

---

## 🛠️ Technical Specifications

### Hardware Requirements

- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or equivalent
- **CPU**: Intel Core i7-7700K (8 cores) or equivalent
- **RAM**: 64GB DDR4 (recommended for large-scale processing)
- **Storage**: 100GB+ free space for datasets and processing

### Software Stack

```python
# Core Dependencies
tensorflow==2.11.0
cuda==11.8
nibabel>=3.2.0
scikit-learn>=1.0.0
scipy>=1.7.0
scikit-image>=0.19.0
opencv-python>=4.5.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### MRI Acquisition Specifications

- **Scanner**: 1.5-Tesla TOSHIBA Vantage (Canon Medical Systems, Japan)
- **T2-FLAIR Sequence Parameters**:
  - TR = 10,000 ms
  - TE = 100 ms
  - TI = 2,500 ms
  - Flip angle = 90°
  - Field of view = 230 × 230 mm²
  - Slice thickness = 6 mm
  - Voxel size = 0.9 × 0.9 mm²
  - Acquisition matrix = [0, 256, 192, 0]

---

## 📋 Usage Pipeline

### Phase 1: Data Preprocessing

```bash
cd Phase1_data_preprocessing
python pre_processing_flair.py \
    --input_dir raw_data/subjects_flair \
    --output_dir preprocessed_output
```

**Preprocessing Steps:**
1. Noise reduction (median + Gaussian filtering)
2. Slice-wise z-score normalization
3. Isotropic resampling (1×1 mm²)
4. Matrix standardization (256×256 pixels)

### Phase 2: Training Data Preparation

```bash
cd Phase2_data_preparation_for_model_training
python generating_3L_masks.py \
    --flair_dir Original_FLAIRs_prep \
    --vent_dir vent_manual_segmentations \
    --wmh_dir abWMH_manual_segmentations \
    --output_dir manual_3l_masks
```

**Dataset Composition:**
- 100 MS patients from local dataset
- 15 MS patients from MSSEG 2016 dataset
- Split: 80% training, 10% validation, 10% testing
- Total: 2,050 training images, 350 validation, 350 testing

### Phase 3: Model Training/Inference

```bash
cd Phase3_model_training_and_inferencing_and_evaluation

# For inference with optimal model (U-Net)
python inferencing_wmh_vent_unet_models_v3.py \
    --input_dir ../Phase1_data_preprocessing/preprocessed_output \
    --model_path trained_models/unet_model \
    --output_dir inference_results

# For training new model (optional)
python training_wmh_vent_unet_models_v3.py \
    --architecture unet \
    --config config/training_config.yaml
```

**Training Configuration:**
- Optimizer: Adam (lr=2×10⁻⁴, β₁=0.9, β₂=0.999)
- Batch size: 8
- Epochs: 100 (early stopping patience=10)
- Loss: Hybrid strategy (weighted categorical cross-entropy for initial 15 epochs → unified focal loss)
- Learning rate schedule: ReduceLROnPlateau
- Validation: 4-fold cross-validation

### Phase 4: Neuroanatomical Processing

```bash
cd Phase4_data_processing

# Classify lesions into anatomical subtypes
python core_processing.py \
    --input_dir ../Phase3_*/inference_results \
    --output_dir processed_results

# Extract comprehensive metrics
python excel_extractor.py \
    --processed_dir processed_results \
    --output_file brain_mri_analysis_results.xlsx

# Add brain area normalization
python excel_filler_brain_TIA.py \
    --excel_file brain_mri_analysis_results.xlsx \
    --output_file brain_mri_analysis_results_PROCESSED_updated.xlsx
```

**Extracted Metrics:**
- Total ventricular area and ratio
- Total WMH area and ratio
- PEWMH, DWMH, JCWMH areas and proportions
- Age and gender stratification
- Normalized values (% of total brain area)

### Phase 5: Statistical Analysis

```bash
cd Phase5_statistical_analysis
python comprehensive_statistical_analysis_v3.py \
    --data_file ../Phase4_data_processing/brain_mri_analysis_results_PROCESSED_updated.xlsx \
    --output_dir csv_analysis_outputs_v3
```

**Statistical Methods:**
- Age stratification: 5 groups (18-29, 30-39, 40-49, 50-59, 60+ years)
- Normality testing: Shapiro-Wilk
- Group comparisons: Independent t-tests (where CLT applies and CV<1.0) / Mann-Whitney U tests
- Effect sizes: Cohen's d / rank-biserial correlation (r)
- Correlation analysis: Spearman correlation matrices
- Multiple comparison correction: Bonferroni adjustment

---

## 📈 Performance Benchmarks

### Computational Efficiency

| Operation | Time | Hardware |
|-----------|------|----------|
| **Training (per epoch)** | 40-45 seconds | RTX 3060 |
| **Inference (per image)** | 38 milliseconds | RTX 3060 |
| **Full cohort processing** | ~2 hours | 1,381 subjects |
| **Statistical analysis** | ~10 minutes | i7-7700K |

### Segmentation Accuracy

**U-Net on Test Set (4-fold cross-validation):**
- Overall DSC: 88.8±0.5%
- HD95: 2.8±0.0mm
- Lesion-wise F1-score: 77.4±1.4%

**Clinical Acceptability:**
- ✅ Suitable for population-level analysis
- ✅ Consistent performance via cross-validation
- ✅ Robust to clinical imaging variability

---

## 🌍 Clinical Impact

### Research Contributions

- **Population-Specific Data**: Large-scale MS neuroimaging study in Middle Eastern population
- **Normative Values**: Preliminary reference ranges for WMH and ventricular ratios in Iranian/Middle Eastern MS patients
- **Methodological Framework**: Scalable approach for large-scale neuroimaging studies
- **Open-Source Tools**: Reproducible pipeline for global MS research community

### Clinical Applications

- **Diagnostic Support**: Automated lesion quantification and classification
- **Disease Monitoring**: Longitudinal tracking of lesion burden
- **Treatment Assessment**: Quantitative biomarkers for therapeutic response
- **Risk Stratification**: Age and gender-specific lesion patterns

### Future Research Directions

1. **Longitudinal Studies**: Track individual patient trajectories
2. **Higher-Resolution Protocols**: Volumetric analysis with ≤3mm slice thickness / isotropic acquisition
3. **Multi-Modal Integration**: Combine FLAIR with DTI, T1-weighted sequences
4. **Clinical Correlation**: Integrate with EDSS scores and disease duration
5. **Genetic Association**: Link imaging biomarkers with genetic profiles (e.g., HLA-DRB1*1501)
6. **Multi-Center Expansion**: Validate across diverse Iranian regions

---

## 📚 Documentation

### Available Resources
- **[Article Manuscript](p4_Manuscript_BAWIL2025_R1.pdf)**: Complete research article (Revised)
- **[Repository Explanation](repo_explanation.docx)**: Detailed methodology
- **[Figure Collection](Article_Figures/)**: All 7 publication-quality figures
- **[Statistical Tables](Article_Tables/)**: Comprehensive demographic and performance tables

### Key Figures
- **Figure 1**: Expert manual annotation examples (FLAIR with ground truth overlays)
- **Figure 2**: Study population age distribution histograms by group and gender
- **Figure 3**: Age-stratified ventricular area analysis (stacked area plots)
- **Figure 4**: Age-related scatter plots for ventricular and WMH ratios by gender
- **Figure 5**: Age-stratified lesion (WMH) area analysis
- **Figure 6**: Anatomical lesion distribution in MS patients (PEWMH, DWMH, JCWMH)
- **Figure 7**: Spearman correlation matrices (HC vs MS)

---

## 🤝 Contributing

We welcome contributions to advance MS neuroimaging research! Areas for contribution:

- **Algorithm Improvements**: Enhanced architectures or training strategies
- **Population Expansion**: Extension to other geographic regions
- **Clinical Validation**: Real-world deployment studies
- **Multi-Modal Analysis**: Integration with advanced imaging sequences

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code quality checks
black src/
flake8 src/
mypy src/
```

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@article{bashiri2026deeplearning,
    title={Deep Learning-Based Neuroanatomical Profiling Reveals Population-Specific Brain Changes in Multiple Sclerosis: A Large-Scale Middle Eastern Study},
    author={Bashiri Bawil, Mahdi and Shamsi, Mousa and Shakeri Bavil, Abolhassan},
    journal={Under Review - BMC Medical Imaging},
    year={2026},
    note={Ethics Approval: IR.TBZMED.REC.1402.902},
    url={https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study}
}
```

---

## 🏥 Study Information

### Clinical Site
**Golghasht Medical Imaging Center**  
Tabriz, Iran  
- Data collection: 2021-2024
- Expert neuroradiologist validation (20+ years experience)
- Standardized clinical MRI protocols

### Ethics & Compliance
- **IRB Approval**: Tabriz University of Medical Sciences Research Ethics Committee
- **Approval Number**: IR.TBZMED.REC.1402.902
- **Compliance**: 1964 Helsinki Declaration and amendments
- **Patient Consent**: Written informed consent obtained
- **Data Protection**: Comprehensive anonymization protocols

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Golghasht Medical Imaging Center** for clinical dataset and imaging resources (Eng. Mehrdad Rahbarpour, Eng. Azad Ahmadi, Dr. Aydin Asadi)
- **Expert neuroradiologist** for manual annotations and validation
- **Study participants** (381 MS patients, 1,000 healthy volunteers)
- **Tabriz University of Medical Sciences** for institutional support and ethics oversight
- **MSSEG 2016 Challenge** for public validation dataset
- **Open-source community** for foundational deep learning tools

---

## 📞 Contact & Support

### Repository
- **GitHub**: [https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study](https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study)
- **Issues**: [GitHub Issues](https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study/discussions)

### Corresponding Author
**Mousa Shamsi** — shamsi@sut.ac.ir  
Biomedical Engineering Faculty, Sahand University of Technology, Tabriz, Iran

---

## 🌟 Impact Statement

This research contributes to understanding Multiple Sclerosis through neuroimaging in an underrepresented population. By providing:

- **Open-source tools** for automated MS lesion analysis
- **Preliminary population-specific normative data** for Middle Eastern populations
- **Methodological framework** for large-scale neuroimaging studies
- **Clinical translation pathway** from research to practice

We aim to support improved patient care and advance global MS research initiatives.

---

[![Star History Chart](https://api.star-history.com/svg?repos=Mahdi-Bashiri/MS-DeepBrain-Study&type=Date)](https://star-history.com/#Mahdi-Bashiri/MS-DeepBrain-Study&Date)

---

*Version: 3.0 (Revised) | Last Updated: April 2026 | Status: Manuscript Under Review (R1)*
