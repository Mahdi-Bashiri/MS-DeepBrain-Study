# Deep Learning-Based Neuroanatomical Profiling of Multiple Sclerosis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.11+](https://img.shields.io/badge/TensorFlow-2.11+-orange.svg)](https://tensorflow.org/)
[![Medical Imaging](https://img.shields.io/badge/domain-Medical%20Imaging-green.svg)](https://github.com/topics/medical-imaging)
[![Multiple Sclerosis](https://img.shields.io/badge/application-Multiple%20Sclerosis-red.svg)](https://github.com/topics/multiple-sclerosis)

## 🧠 Overview

This repository implements a **large-scale neuroanatomical profiling study** of Multiple Sclerosis using deep learning-based automated segmentation. Our comprehensive analysis of **1,381 subjects** from Northwest Iran provides the first detailed statistical characterization of brain structural changes in an underrepresented Middle Eastern population.

### 🎯 Key Contributions

- **🔬 Largest MS Neuroimaging Study**: 1,381 participants (381 MS patients, 1,000 healthy controls)
- **🌍 Population-Specific Research**: First large-scale study addressing Middle Eastern MS populations
- **🤖 Automated Pipeline**: Attention U-Net for clinically-acceptable automated segmentation
- **📊 Comprehensive Statistics**: Multi-dimensional analysis across age, gender, and anatomical regions
- **⚡ Clinical Translation**: Reduced manual segmentation time with reproducible quantitative metrics

### 📈 Key Clinical Findings

| Metric | MS Patients | Healthy Controls | Effect Size |
|--------|-------------|------------------|-------------|
| **Lesion Burden Disparity** | 5.5x higher | Baseline | Large (r=0.82) |
| **Periventricular Involvement** | 58.02±28.35% | - | - |
| **Age-Related Progression** | 0.13% → 0.71% | - | - |
| **Gender Differences** | ♂: 61.85% vs ♀: 56.45% | - | p=0.018 |

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

# Run automated segmentation
results = inferencing_wmh_vent_unet_models_v3.main(
    input_dir="path/to/flair/images",
    model_path="trained_models/attention_unet_model",
    output_dir="results/"
)

# Process and extract quantitative metrics
processed_data = core_processing.main(results)

# Perform comprehensive statistical analysis
statistical_results = comprehensive_statistical_analysis_v3.main(processed_data)
```

---

## 🏗️ Repository Structure

The repository follows a **5-phase pipeline architecture**:

```
├── 📁 Article_Figures/              # 7 main figures from published article
│   ├── Figure_1.tif                 # Sample FLAIR annotations images
│   ├── Figure_2.tif                 # Study population demographics
│   ├── Figure_3.tif                 # Venticular burden analysis
│   ├── Figure_4.tif                 # Burden Analysis
│   ├── Figure_5.tif                 # Lesion burden analysis
│   ├── Figure_6.tif                 # Subtype lesion analysis
│   └── Figure_7.tif                 # Statistical correlation matrices
├── 📁 Article_Tables/               # 3 comprehensive tables
│   ├── Table_1.docx                # Population demographics
│   ├── Table_2.docx                # Performance metrics
│   └── Table_3.docx                # Statistical analysis results
├── 📁 Phase1_data_preprocessing/    # FLAIR image preprocessing
│   ├── 📁 raw_data/                # Sample data (5 patients)
│   └── pre_processing_flair.py     # 5-step preprocessing pipeline
├── 📁 Phase2_data_preparation_for_model_training/
│   ├── 📁 Original_FLAIRs_prep/    # Preprocessed FLAIR images
│   ├── 📁 abWMH_manual_segmentations/ # Manual abnormal WMH masks
│   ├── 📁 vent_manual_segmentations/  # Manual ventricle masks
│   ├── 📁 manual_3l_masks/         # Generated 3-level masks
│   └── generating_3L_masks.py      # Training data generation
├── 📁 Phase3_model_training_and_inferencing_and_evaluation/
│   ├── 📁 dataset_3l_man/          # Training/testing datasets
│   ├── 📁 chosen_model_performance/ # Performance metrics
│   ├── 📁 trained_models/          # Pre-trained all models containing Attention U-Net
│   ├── training_wmh_vent_unet_models_v3.py    # Model training
│   └── inferencing_wmh_vent_unet_models_v3.py # Automated inference
├── 📁 Phase4_data_processing/       # Quantitative analysis
│   ├── brain_mri_analysis_results_PROCESSED_updated.xlsx
│   ├── core_processing.py          # Lesion characterization
│   ├── excel_extractor.py          # Data extraction
│   └── excel_filler_brain_TIA.py   # Brain area incorporation
├── 📁 Phase5_statistical_analysis/  # Comprehensive statistics
│   ├── 📁 csv_analysis_outputs_no_outlier_v3/  # Results without outlier removal
│   ├── 📁 csv_analysis_outputs_outlier_v3/     # Results with outlier removal
│   └── comprehensive_statistical_analysis_v3.py # Statistical pipeline
├── 📄 our_article_DOI.md           # Citation information
├── 📄 repo_explanation.docx        # Detailed methodology
└── 📄 README.md                    # This file
```

---

## 🔬 Methodology

### Study Population

- **Total Participants**: 1,381 subjects from Northwest Iran (2021-2024)
  - **MS Patients**: 381 subjects
  - **Healthy Controls**: 1,000 subjects
- **Demographics**: 68.6% female, 31.4% male
- **Age Range**: 18-74 years
- **Location**: Golghasht Medical Imaging Center, Tabriz, Iran

### Deep Learning Architecture

**Attention U-Net Implementation:**
- **Network Type**: Encoder-decoder with attention mechanisms
- **Input**: Single-modality FLAIR sequences
- **Performance**: DSC=0.749, IoU=0.610 (clinically acceptable)
- **Processing Speed**: 40-45 seconds/epoch training, 5ms/image inference

### Preprocessing Pipeline

1. **Noise Reduction**: Advanced filtering techniques
2. **Skull Stripping**: Automated brain extraction
3. **Normalization**: Intensity standardization
4. **Resampling**: Consistent spatial resolution
5. **Matrix Standardization**: Unified coordinate systems

### Lesion Classification

**Neuroanatomically-Informed Categories:**
- **PEWMH**: Periventricular White Matter Hyperintensities
- **PAWMH**: Paraventricular Hyperintensities  
- **JCWMH**: Juxtacortical White Matter Hyperintensities

---

## 📊 Key Clinical Findings

### Lesion Burden Analysis

- **Dramatic Disparity**: MS patients showed **5.5-fold higher** lesion burden
- **Age-Related Progression**: Normalized lesion ratios increased from **0.13% to 0.71%**
- **Large Clinical Effect**: r=0.82, p<0.001

### Anatomical Distribution

- **Periventricular Predominance**: 58.02±28.35% of total lesion burden
- **Gender Differences**: Males showed higher periventricular involvement (61.85% vs 56.45%, p=0.018)
- **Age-Related Changes**: Periventricular involvement increased from 44.97% to 71.43% with age

### Population-Specific Insights

- **Regional Relevance**: Iran shows elevated MS prevalence (100 per 100,000) vs global rates (35.9 per 100,000)
- **Normative Baselines**: First population-specific reference values for Middle Eastern populations
- **Clinical Translation**: Automated pipeline reduces manual segmentation time and variability

---

## 🛠️ Technical Specifications

### Hardware Requirements

- **GPU**: NVIDIA RTX 3060 or equivalent
- **CPU**: Intel Core i7-7700K or equivalent
- **RAM**: 64GB (recommended for large-scale processing)
- **Storage**: 100GB+ free space for processing

### Software Stack

```python
# Core Dependencies
tensorflow==2.11.0
nibabel>=3.2.0
scikit-learn>=1.0.0
scipy>=1.7.0
scikit-image>=0.19.0
opencv-python>=4.5.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### Scanner Specifications

- **MRI Scanner**: 1.5-Tesla TOSHIBA Vantage
- **Sequence**: FLAIR (Fluid-Attenuated Inversion Recovery)
- **Compatibility**: Tested on anisotropic voxels (clinical protocols)

---

## 📋 Usage Pipeline

### Phase 1: Data Preprocessing

```bash
cd Phase1_data_preprocessing
python pre_processing_flair.py --input_dir raw_data/subjects_flair --output_dir preprocessed_output
```

### Phase 2: Training Data Preparation (if training new models)

```bash
cd Phase2_data_preparation_for_model_training
python generating_3L_masks.py --flair_dir Original_FLAIRs_prep --output_dir dataset_3l_man
```

### Phase 3: Model Training/Inference

```bash
cd Phase3_model_training_and_inferencing_and_evaluation

# For inference with pre-trained model
python inferencing_wmh_vent_unet_models_v3.py \
    --input_dir ../Phase1_data_preprocessing/preprocessed_output \
    --model_path trained_models/attention_unet_model \
    --output_dir inference_results

# For training new model
python training_wmh_vent_unet_models_v3.py --config config/training_config.yaml
```

### Phase 4: Data Processing

```bash
cd Phase4_data_processing

# Core lesion processing
python core_processing.py --input_dir ../Phase3_*/inference_results

# Extract comprehensive data
python excel_extractor.py --processed_dir processed_masks

# Add brain area information
python excel_filler_brain_TIA.py --excel_file brain_mri_analysis_results.xlsx
```

### Phase 5: Statistical Analysis

```bash
cd Phase5_statistical_analysis
python comprehensive_statistical_analysis_v3.py \
    --data_file ../Phase4_data_processing/brain_mri_analysis_results_PROCESSED_updated.xlsx \
    --output_dir csv_analysis_outputs_no_outlier_v3 \
    --remove_outliers False
```

---

## 📈 Performance Metrics

### Segmentation Performance

| Metric | Attention U-Net | Clinical Acceptability |
|--------|----------------|-------------------------|
| **Dice Coefficient** | 0.749 | ✅ Clinically Acceptable |
| **IoU** | 0.610 | ✅ Clinically Acceptable |
| **Processing Time** | 5ms/image | ⚡ Real-time Capable |
| **Training Time** | 40-45s/epoch | 🚀 Efficient |

### Statistical Analysis Coverage

- **Age Stratification**: 5 age groups (18-29, 30-39, 40-49, 50-59, 60+ years)
- **Gender Analysis**: Male vs female lesion patterns
- **Anatomical Regions**: Periventricular, paraventricular, juxtacortical
- **Correlation Matrices**: Multi-dimensional relationship analysis

---

## 🌍 Clinical Impact

### Diagnostic Applications

- **MS Diagnosis**: Enhanced accuracy in lesion characterization
- **Disease Monitoring**: Quantitative progression tracking
- **Treatment Planning**: Biomarker-guided therapy decisions
- **Population Studies**: Reference values for Middle Eastern populations

### Research Contributions

- **Methodological**: Single-modality FLAIR-only approach for clinical accessibility
- **Technical**: Attention mechanisms for scattered MS lesion distribution
- **Clinical**: Population-specific normative data for underrepresented regions
- **Global**: Contributing to worldwide MS imaging databases

---

## 📚 Documentation

### Detailed Guides
- **[Usage Tutorial](USAGE.md)**: Step-by-step processing pipeline
- **[Repository Structure](article_repo_structure.md)**: Detailed organization explanation
- **[Article Summary](p4_article_summary.docx)**: Comprehensive research overview

### Clinical Resources
- **Population Demographics**: Age, gender, and regional characteristics
- **Normative Values**: Reference ranges for clinical interpretation
- **Statistical Interpretation**: Clinical significance guidelines

---

## 🎯 Future Directions

### Immediate Next Steps
- **Longitudinal Studies**: Track individual disease progression over time
- **Multi-Center Validation**: Expand across diverse Iranian regions
- **Multi-Modal Integration**: Combine with DTI and advanced sequences

### Long-term Goals
- **Clinical Implementation**: Real-world deployment and validation
- **Genetic Integration**: Combine imaging with genetic markers
- **Global Expansion**: Extend to other underrepresented populations

---

## 🤝 Contributing

We welcome contributions to advance MS neuroimaging research! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Areas for Contribution
- **Algorithm Improvements**: Enhance attention mechanisms
- **Population Expansion**: Extend to other demographics
- **Clinical Validation**: Real-world deployment studies
- **Multi-Modal Integration**: Advanced imaging sequences

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

If you use this work in your research, please cite our article:

```bibtex
@article{bashiri2025deeplearning,
    title={Deep Learning-Based Neuroanatomical Profiling Reveals Detailed Brain Changes: A Large-Scale Multiple Sclerosis Study},
    author={Bashiri Bawil, Mahdi and Shamsi, Mousa and Shakeri Bavil, Abolhassan},
    journal={[Journal Name]},
    year={2025},
    volume={[Volume]},
    pages={[Pages]},
    doi={[DOI]},
    url={https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study}
}
```

See [our_article_DOI.md](our_article_DOI.md) for complete citation details.

---

## 🏥 Clinical Collaboration

### Study Institution
**Golghasht Medical Imaging Center**  
Tabriz, Iran  
- Data collection period: 2021-2024
- Expert neuroradiologist validation
- Clinical protocol standardization

### Ethical Considerations
- IRB approval obtained
- Patient consent protocols
- Data anonymization procedures
- GDPR compliance measures

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Golghasht Medical Imaging Center** for providing the comprehensive clinical dataset
- **Expert neuroradiologists** for manual annotations and clinical validation  
- **Iranian MS patients and healthy volunteers** for their participation
- **Northwest Iran medical community** for supporting this research initiative
- **Open-source community** for foundational tools and methodological guidance

---

## 📞 Contact & Support

### Repository Maintenance
- **GitHub Repository**: [https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study](https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study)
- **Issues & Bugs**: [GitHub Issues](https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/Mahdi-Bashiri/MS-DeepBrain-Study/discussions)

### Research Collaboration
For research collaborations, clinical implementations, or dataset access inquiries, please use the appropriate GitHub communication channels.

---

## 🌟 Impact & Recognition

### Research Significance
- **First comprehensive study** of MS in Middle Eastern populations at this scale
- **Clinical translation** of attention-based deep learning to MS neuroimaging
- **Population-specific insights** for precision medicine approaches
- **Open-source contribution** to global MS research community

### Community Engagement
If you find this research valuable for MS understanding and patient care, please consider:
- ⭐ **Starring** the repository
- 🍴 **Forking** for your own research
- 🐛 **Reporting** issues or bugs
- 💡 **Contributing** improvements and extensions

[![Star History Chart](https://api.star-history.com/svg?repos=Mahdi-Bashiri/MS-DeepBrain-Study&type=Date)](https://star-history.com/#Mahdi-Bashiri/MS-DeepBrain-Study&Date)

---

*This repository represents a significant step forward in understanding Multiple Sclerosis through advanced neuroimaging and population-specific research, contributing to better patient outcomes and global MS research initiatives.*
