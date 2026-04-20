"""
P1 & P4 Articles - Data Loading System

Complete implementation for brain segmentation experiments

Specialized Gray Matter (GM) Segmentation with U-Net Models - Journal Paper Implementation
Binary segmentation: Background vs Specialized GM
Professional results saving and visualization for publication

This relates to our articles:
"Specialized gray matter segmentation via a generative adversarial network: 
application on brain white matter hyperintensities classification"

"Deep Learning-Based Neuroanatomical Profiling Reveals Detailed Brain Changes:
A Large-Scale Multiple Sclerosis Study"

Features:
- Load FLAIR images and individual mask files from Cohort directory
- Support both Local_SAI_GM_sp dataset
- Handle standard and zoomed preprocessing variants
- Combine masks into 2-class format
- Create paired inputs: [FLAIR | mask] concatenated (256x512)
- Patient-stratified K-fold cross-validation
- TensorFlow dataset creation with proper batching

Authors:
"Mahdi Bashiri Bawil, Mousa Shamsi, Abolhassan Shakeri Bavil"

Developer:
"Mahdi Bashiri Bawil"
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
from sklearn.model_selection import KFold
from tqdm import tqdm
import cv2 as cv

# Deep Learning
import tensorflow as tf


###################### Configuration ######################

class DataConfig:
    """Data configuration for P4 experiments"""
    
    def __init__(self):
        # Base paths
        self.cohort_dir = Path("/mnt/e/MBashiri/ours_articles/Paper#2/Data/Cohort")  # CHANGE THIS to your actual path of Data Cohort
        
        # Dataset configurations
        self.datasets = {
            'Local_SAI_GM_sp': {
                'base_path': self.cohort_dir / 'Local_SAI_GM_sp',
                'slice_range': (1, 20),  # inclusive range 9,15
                'patient_prefix_length': 6  # "101228"
            }
        }
        
        # Preprocessing variants
        self.preprocessing_types = ['standard', 'zoomed']
        
        # Class scenarios
        self.class_scenarios = {
            'binary': {
                'num_classes': 2,
                'class_names': ['Background', 'Specialized GM'],
                'description': 'Binary: Background, Specialized GM',
                'class_mapping': {
                    'background': 0,
                    'specialized_gm': 1,
                }
            }
        }
        
        # K-fold parameters
        self.k_folds = 5
        self.test_split = 0.1  # 10% for test set
        self.random_state = 42
        
        # Image parameters
        self.target_size = (256, 256)
        self.paired_width = 512  # FLAIR (256) + mask (256)
        
        # Paths for splits
        self.splits_dir = Path("data_splits_sp_gm")
        self.splits_file = self.splits_dir / "SP_GM_fold_assignments.json"


###################### Helper Functions ######################

def extract_patient_id(filename: str, prefix_length: int = 6) -> str:
    """
    Extract patient ID from filename
    
    Args:
        filename: e.g., "101228_5.npy" or "c01p01_25.png"
        prefix_length: Number of characters in patient ID
        
    Returns:
        Patient ID: e.g., "101228" or "c01p01"
    """
    return filename.split('_')[0][:prefix_length]


def extract_slice_number(filename: str) -> int:
    """
    Extract slice number from filename
    
    Args:
        filename: e.g., "101228_5.npy" or "c01p01_25.png"
        
    Returns:
        Slice number as integer
    """
    # Get the part before file extension
    basename = filename.split('.')[0]
    # Get the last part after splitting by '_'
    slice_num = basename.split('_')[-1]
    return int(slice_num)


def load_flair_image(flair_path: Path, normalize: bool = False, of_z_score: bool = False) -> np.ndarray:
    """
    Load FLAIR image (.png format)
    
    Args:
        flair_path: Path to .png file
        normalize: Whether to apply z-score normalization
        
    Returns:
        FLAIR image (256, 256, 1) as float32
    """
    if of_z_score:
        # Load NPY: the already z-scored FLAIR image data
        flair = np.load(str(flair_path).replace('.png','.npy')).astype(np.float32)
    else:
        # Load PNG as grayscale
        flair = cv.imread(str(flair_path), cv.IMREAD_GRAYSCALE).astype(np.float32)

        # Normalize to [-1, 1]:
        flair = (flair - np.min(flair)) / (np.max(flair) - np.min(flair))
        flair = (2 * flair) - 1
    
    # Ensure correct shape
    if len(flair.shape) == 2:
        flair = np.expand_dims(flair, axis=-1)
    
    # Additional normalization if needed (should already be normalized)
    if normalize and (np.std(flair) > 2.0 or np.abs(np.mean(flair)) > 1.0):
        # Re-normalize if values seem off
        flair = (flair - np.mean(flair)) / (np.std(flair) + 1e-7)
    
    return flair


def load_mask_image(mask_path: Path) -> np.ndarray:
    """
    Load mask image (.png format)
    
    Args:
        mask_path: Path to .png file
        
    Returns:
        Binary mask (256, 256) as uint8
    """
    # Load PNG as grayscale
    mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    
    # Binarize (any non-zero value becomes 1)
    mask = (mask > 0).astype(np.uint8)
    
    return mask


def combine_masks(gm_mask: np.ndarray, 
                  class_scenario: str,
                  preprocess: bool = False) -> np.ndarray:
    """
    Combine individual masks into multi-class format
    
    Args:
        gm_mask: Ventricles mask (256, 256)
        class_scenario: 'binary'
        preprocess: Boolean turning the morphological preprocessing on or off
        
    Returns:
        Combined mask (256, 256) with class labels
    """
    if preprocess:
        from skimage.morphology import remove_small_objects, binary_erosion, binary_closing, binary_opening, disk, binary_dilation
        min_object_size = 5
        closing_kernel_size = 2
        dilation_kernel_size = 1

        gm_mask = gm_mask > 0

        gm_mask = binary_closing(gm_mask, disk(closing_kernel_size))
        gm_mask = binary_erosion(gm_mask, disk(dilation_kernel_size))
        gm_mask = remove_small_objects(gm_mask, min_size=min_object_size)

    # Class 0: Background (default)
    # Class 1: Specialized GM
    combined = np.zeros_like(gm_mask, dtype=np.uint8)
    combined[gm_mask>0] = 1
    
    return combined


def is_valid_slice(gm_mask: np.ndarray) -> bool:
    """
    Check if slice has at least one non-empty mask
    
    Args:
        gm_mask: Specialized GM mask (256, 256)
        
    Returns:
        True if at least one mask has non-zero pixels
    """
    has_specialized_gm = np.sum(gm_mask) > 50
    
    # Valid if ANY mask has content
    return True #  or has_specialized_gm 


def create_paired_input(flair: np.ndarray, 
                                 mask: np.ndarray,
                                 brain_mask: np.ndarray,
                                 num_classes: np.ndarray,
                                 if_bet=False) -> np.ndarray:
    """
    Create paired input: [FLAIR | mask] concatenated horizontally
    
    Args:
        flair: FLAIR image (256, 256, 1) float32
        mask: Combined mask (256, 256) uint8
        
    Returns:
        Paired image (256, 512, 1) float32
    """
    # Binarize (any non-zero value becomes 1)
    brain_mask = brain_mask > 0

    # Brain extraction
    if if_bet:
        # print("\n\t Doing THEEEEEEEEE BET")
        flair[~brain_mask] = np.min(flair)
        mask[~brain_mask] = 0

    # Ensure flair is 3D
    if len(flair.shape) == 2:
        flair = np.expand_dims(flair, axis=-1)
    
    # Convert mask to float and normalize to [0, 1] range for consistency

    max_class = num_classes
    mask_normalized = mask.astype(np.float32)
    if max_class > 0:
        mask_normalized = mask_normalized / max_class
        mask_normalized = (2 * mask_normalized) - 1
    
    mask_3d = np.expand_dims(mask_normalized, axis=-1)
    
    # Concatenate horizontally: [FLAIR | mask]
    paired = np.concatenate([flair, mask_3d], axis=1)  # (256, 512, 1)
    
    return paired, mask


###################### Patient Stratified Splitting ######################

class PatientStratifiedSplitter:
    """
    Create patient-stratified train/val/test splits
    Similar to P6 implementation but adapted for P1 data structure
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.config.splits_dir.mkdir(exist_ok=True)
    
    def collect_all_patients(self) -> Dict[str, List[str]]:
        """
        Collect all unique patient IDs from both datasets
        
        Returns:
            Dictionary mapping dataset_name -> list of patient IDs
        """
        all_patients = {}
        
        for dataset_name, dataset_config in self.config.datasets.items():
            patients = set()
            
            # Path to FLAIR images (standard preprocessing)
            flair_dir = dataset_config['base_path'] / 'FLAIR' / 'Preprocessed' / 'images'
            
            if not flair_dir.exists():
                print(f"Warning: {flair_dir} does not exist. Skipping {dataset_name}.")
                continue
            
            # Collect all .png files
            for flair_file in flair_dir.glob('*.png'):
                patient_id = extract_patient_id(
                    flair_file.name, 
                    dataset_config['patient_prefix_length']
                )
                patients.add(patient_id)
            
            all_patients[dataset_name] = sorted(list(patients))
            print(f"{dataset_name}: {len(all_patients[dataset_name])} patients")
        
        return all_patients
    
    def create_patient_stratified_splits(self, 
                                        save: bool = True) -> Dict:
        """
        Create patient-stratified K-fold splits
        
        Returns:
            Dictionary containing fold assignments
        """
        all_patients = self.collect_all_patients()
        
        # Combine patients from both datasets
        combined_patients = []
        for dataset_name, patients in all_patients.items():
            combined_patients.extend(patients)
        
        combined_patients = np.array(combined_patients)
        total_patients = len(combined_patients)
        
        print(f"\nTotal unique patients: {total_patients}")
        
        # Step 1: Split into train+val (80%) and test (20%)
        np.random.seed(self.config.random_state)
        test_size = int(total_patients * self.config.test_split)
        
        test_indices = np.random.choice(
            total_patients, 
            size=test_size, 
            replace=False
        )
        
        test_patients = combined_patients[test_indices]
        train_val_indices = np.setdiff1d(np.arange(total_patients), test_indices)
        train_val_patients = combined_patients[train_val_indices]
        
        print(f"Test patients: {len(test_patients)}")
        print(f"Train+Val patients: {len(train_val_patients)}")
        
        # Step 2: Create K-fold splits on train+val patients
        kfold = KFold(
            n_splits=self.config.k_folds, 
            shuffle=True, 
            random_state=self.config.random_state
        )
        
        fold_assignments = {
            'metadata': {
                'total_patients': total_patients,
                'test_patients': len(test_patients),
                'trainval_patients': len(train_val_patients),
                'n_folds': self.config.k_folds,
                'random_seed': self.config.random_state,
                'datasets': list(all_patients.keys())
            },
            'test_set': {
                'patients': test_patients.tolist(),
                'n_patients': len(test_patients)
            },
            'folds': {}
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_val_patients)):
            train_patients_fold = train_val_patients[train_idx]
            val_patients_fold = train_val_patients[val_idx]
            
            fold_assignments['folds'][f'fold_{fold_idx}'] = {
                'train_patients': train_patients_fold.tolist(),
                'val_patients': val_patients_fold.tolist(),
                'n_train': len(train_patients_fold),
                'n_val': len(val_patients_fold)
            }
            
            print(f"Fold {fold_idx}: Train={len(train_patients_fold)}, Val={len(val_patients_fold)}")
        
        # Save to JSON
        if save:
            with open(self.config.splits_file, 'w') as f:
                json.dump(fold_assignments, f, indent=2)
            print(f"\nâœ… Fold assignments saved to: {self.config.splits_file}")
        
        return fold_assignments
    
    def load_fold_assignments(self) -> Dict:
        """Load existing fold assignments from JSON"""
        if not self.config.splits_file.exists():
            raise FileNotFoundError(
                f"Fold assignments not found: {self.config.splits_file}\n"
                f"Run create_patient_stratified_splits() first."
            )
        
        with open(self.config.splits_file, 'r') as f:
            fold_assignments = json.load(f)
        
        return fold_assignments
    
    def verify_patient_separation(self, fold_assignments: Dict) -> bool:
        """
        Verify no patient appears in multiple folds or in both train/val
        Similar to P6's verification logic
        """
        print("\n" + "="*60)
        print("VERIFYING PATIENT SEPARATION")
        print("="*60)
        
        all_issues = []
        test_patients = set(fold_assignments['test_set']['patients'])
        
        # Check 1: No patient in both test and train/val
        for fold_name, fold_data in fold_assignments['folds'].items():
            train_patients = set(fold_data['train_patients'])
            val_patients = set(fold_data['val_patients'])
            
            test_train_overlap = test_patients.intersection(train_patients)
            test_val_overlap = test_patients.intersection(val_patients)
            
            if test_train_overlap:
                issue = f"{fold_name}: Test-Train overlap: {test_train_overlap}"
                all_issues.append(issue)
                print(f"❌ {issue}")
            
            if test_val_overlap:
                issue = f"{fold_name}: Test-Val overlap: {test_val_overlap}"
                all_issues.append(issue)
                print(f"❌ {issue}")
        
        # Check 2: No patient in both train and val within same fold
        for fold_name, fold_data in fold_assignments['folds'].items():
            train_patients = set(fold_data['train_patients'])
            val_patients = set(fold_data['val_patients'])
            
            train_val_overlap = train_patients.intersection(val_patients)
            if train_val_overlap:
                issue = f"{fold_name}: Train-Val overlap: {train_val_overlap}"
                all_issues.append(issue)
                print(f"❌ {issue}")
        
        # Check 3: Each patient in validation exactly once
        all_val_patients = []
        for fold_data in fold_assignments['folds'].values():
            all_val_patients.extend(fold_data['val_patients'])
        
        val_patient_counts = {}
        for patient in all_val_patients:
            val_patient_counts[patient] = val_patient_counts.get(patient, 0) + 1
        
        for patient, count in val_patient_counts.items():
            if count != 1:
                issue = f"Patient {patient} in validation {count} times (should be 1)"
                all_issues.append(issue)
                print(f"❌ {issue}")
        
        if not all_issues:
            print("âœ… All patient separation checks passed")
            print("âœ… No data leakage detected")
            return True
        else:
            print(f"\n❌ Found {len(all_issues)} issues")
            return False


###################### Data Loader ######################

class P1DataLoader:
    """
    Main data loader for P1 experiments
    Handles loading FLAIR and masks, creating paired inputs, TensorFlow datasets
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def get_file_paths(self, 
                       patient_id: str, 
                       slice_num: int,
                       dataset_name: str,
                       preprocessing: str) -> Dict[str, Path]:
        """
        Construct file paths for a given patient-slice
        
        Args:
            patient_id: e.g., "101228" or "c01p01"
            slice_num: Slice number
            dataset_name: 'Local_SAI_GM_sp'
            preprocessing: 'standard' or 'zoomed'
            
        Returns:
            Dictionary with paths to FLAIR and mask files
        """
        dataset_config = self.config.datasets[dataset_name]
        base_path = dataset_config['base_path']
        
        # Determine subdirectory based on preprocessing
        if preprocessing == 'standard':
            flair_subdir = 'images'
            gt_subdir = 'images'
        else:  # zoomed
            flair_subdir = 'zoomed/images'
            gt_subdir = 'zoomed/images'
        
        # Construct paths
        flair_path = base_path / 'FLAIR' / 'Preprocessed' / flair_subdir / f'{patient_id}_{slice_num}.png'
        gm_path = base_path / 'GroundTruth' / gt_subdir / 'GM_Masks' / f'{patient_id}_{slice_num}.png'
        brain_path = base_path / 'GroundTruth' / gt_subdir / 'Brain_Masks' / f'{patient_id}_{slice_num}.png'
        
        # Optional: zooming factors (only for zoomed preprocessing)
        zoom_factors_path = None
        if preprocessing == 'zoomed':
            zoom_factors_path = base_path / 'FLAIR' / 'Preprocessed' / 'zoomed' / 'images' / f'{patient_id}_zooming_factors.npy'
        
        return {
            'flair': flair_path,
            'gm_mask': gm_path,
            'brain_mask': brain_path,
            'zoom_factors': zoom_factors_path
        }
    
    def load_single_slice(self,
                         patient_id: str,
                         slice_num: int,
                         dataset_name: str,
                         preprocessing: str,
                         class_scenario: str,
                         of_z_score: bool = True,
                         if_bet: bool = True,
                         pre_morph: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single patient-slice and create paired input
        
        Args:
            patient_id: Patient identifier
            slice_num: Slice number
            dataset_name: 'Local_SAI_GM_sp'
            preprocessing: 'standard' or 'zoomed'
            class_scenario: 'binary'
            
        Returns:
            Tuple of (paired_input, combined_mask)
            - paired_input: (256, 512, 1) FLAIR + mask concatenated
            - combined_mask: (256, 256) multi-class labels
        """
        # Class number
        num_classes = 1 # int(class_scenario[0]) - 1

        # Get file paths
        paths = self.get_file_paths(patient_id, slice_num, dataset_name, preprocessing)
        
        # Load FLAIR
        flair = load_flair_image(paths['flair'], of_z_score=of_z_score)
        
        # Load masks
        gm_mask = load_mask_image(paths['gm_mask'])
        brain_mask = load_mask_image(paths['brain_mask'])
        
        # Combine masks
        combined_mask = combine_masks(gm_mask, class_scenario, preprocess=pre_morph)
        
        # Create paired input
        paired_input, combined_mask = create_paired_input(flair, combined_mask, brain_mask, num_classes=num_classes, if_bet=if_bet)
        
        return paired_input, combined_mask
        
    def collect_patient_slices(self, 
                            patient_list: List[str],
                            dataset_name: str,
                            preprocessing: str) -> List[Tuple[str, int, str]]:
        """
        Collect all valid slice files for given patients
        FILTERS OUT SLICES WITH ALL EMPTY MASKS
        
        Args:
            patient_list: List of patient IDs
            dataset_name: 'Local_SAI_GM_sp'
            preprocessing: 'standard' or 'zoomed'
            
        Returns:
            List of tuples (patient_id, slice_num, dataset_name)
        """
        dataset_config = self.config.datasets[dataset_name]
        slice_min, slice_max = dataset_config['slice_range']
        
        patient_slices = []
        skipped_empty = 0
        
        for patient_id in patient_list:
            # Check which dataset this patient belongs to
            # Try to find patient in current dataset
            for slice_num in range(slice_min, slice_max + 1):
                paths = self.get_file_paths(patient_id, slice_num, dataset_name, preprocessing)
                
                # Check if all required files exist
                if (paths['flair'].exists() and 
                    paths['gm_mask'].exists() and 
                    paths['brain_mask'].exists()):
                    
                    # VALIDATION: Check if masks are not all empty
                    try:
                        gm_mask = load_mask_image(paths['gm_mask'])
                        brain_mask = load_mask_image(paths['brain_mask'])
                        
                        # Only add if at least one mask has content
                        if is_valid_slice(gm_mask):
                            patient_slices.append((patient_id, slice_num, dataset_name))
                        else:
                            skipped_empty += 1
                            
                    except Exception as e:
                        print(f"Warning: Could not validate {patient_id}_{slice_num}: {e}")
                        skipped_empty += 1
        
        if skipped_empty > 0:
            print(f"  ⚠️  Skipped {skipped_empty} slices with empty masks")
        
        return patient_slices
        
    def create_dataset_for_fold(self,
                                fold_id: int,
                                split: str,
                                preprocessing: str,
                                class_scenario: str,
                                batch_size: int = 1,
                                shuffle: bool = True,
                                use_z_scored: bool = True,
                                bet: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset for a specific fold and split
        
        Args:
            fold_id: Fold number (0-4)
            split: 'train', 'val', or 'test'
            preprocessing: 'standard' or 'zoomed'
            class_scenario: 'binary'
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            tf.data.Dataset yielding (paired_input, combined_mask) batches
        """
        # Load fold assignments
        splitter = PatientStratifiedSplitter(self.config)
        fold_assignments = splitter.load_fold_assignments()
        
        # Get patient list for this split
        if split == 'test':
            patient_list = fold_assignments['test_set']['patients']
        else:
            fold_key = f'fold_{fold_id}'
            if split == 'train':
                patient_list = fold_assignments['folds'][fold_key]['train_patients']
            elif split == 'val':
                patient_list = fold_assignments['folds'][fold_key]['val_patients']
            else:
                raise ValueError(f"Unknown split: {split}")
        
        print(f"\nCreating dataset for fold {fold_id}, split '{split}'")
        print(f"Patients: {len(patient_list)}")
        
        # Collect all patient-slices from both datasets
        all_patient_slices = []
        
        for dataset_name in self.config.datasets.keys():
            # Filter patient list to only include patients from this dataset
            # This is done by checking patient ID prefix
            dataset_patients = [p for p in patient_list]
            
            patient_slices = self.collect_patient_slices(
                dataset_patients, 
                dataset_name, 
                preprocessing
            )
            all_patient_slices.extend(patient_slices)
        
        print(f"Total slices: {len(all_patient_slices)}")
        
        if len(all_patient_slices) == 0:
            raise ValueError(f"No data found for fold {fold_id}, split '{split}'")
        
        # Create TensorFlow dataset
        def data_generator():
            """Generator function for tf.data.Dataset"""
            for patient_id, slice_num, dataset_name in all_patient_slices:
                try:
                    paired_input, combined_mask = self.load_single_slice(
                        patient_id, slice_num, dataset_name, 
                        preprocessing, class_scenario
                    )
                    yield paired_input, combined_mask, patient_id, slice_num
                except Exception as e:
                    print(f"Error loading {patient_id}_{slice_num}: {e}")
                    continue
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(256, 512, 1), dtype=tf.float32),   # concatenated image
                tf.TensorSpec(shape=(256, 256), dtype=tf.uint8),        # multi-level mask
                tf.TensorSpec(shape=(),            dtype=tf.string),    # patient_id
                tf.TensorSpec(shape=(),            dtype=tf.int32)      # slice_num
            )
        )

        # ── Cache BEFORE shuffle/batch ──────────────────────────────────────
        # On epoch 1 the generator runs once and all samples are stored
        # in RAM (~1 GB).  From epoch 2 onward no disk I/O occurs at all.
        # Placing cache HERE (on unbatched, unshuffled samples) means:
        #   • The expensive load/decode/combine step is paid only once.
        #   • Shuffle re-randomises the order freshly each epoch (because
        #     reshuffle_each_iteration=True is the default).
        #   • Batch composition therefore differs every epoch as desired.
        dataset = dataset.cache()

        # Shuffle if training  (acts on the in-RAM cache every epoch)
        if shuffle and split == 'train':
            dataset = dataset.shuffle(
                buffer_size=len(all_patient_slices),
                reshuffle_each_iteration=True   # new random order each epoch
            )
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


###################### Testing & Validation Functions ######################

def test_data_loading():
    """Test data loading functionality"""
    print("\n" + "="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    config = DataConfig()
    
    # Test 1: Create fold assignments
    print("\n[TEST 1] Creating patient stratified splits...")
    splitter = PatientStratifiedSplitter(config)
    fold_assignments = splitter.create_patient_stratified_splits(save=True)
    
    # Verify patient separation
    is_valid = splitter.verify_patient_separation(fold_assignments)
    
    if not is_valid:
        print("❌ Patient separation verification failed!")
        return False
    
    # Test 2: Load a single slice
    print("\n[TEST 2] Loading single slice...")
    loader = P1DataLoader(config)
    
    # Get a test patient from fold 0 train set
    test_patient = fold_assignments['folds']['fold_0']['train_patients'][0]
    
    # Determine which dataset this patient belongs to
    if test_patient.startswith('1'):
        test_dataset = 'Local_SAI_GM_sp'
        test_slice = 10  # Middle of 8-15 range
    else:
        raise ValueError

    
    try:
        paired_input, combined_mask = loader.load_single_slice(
            test_patient, test_slice, test_dataset, 
            'standard', 'binary'
        )
        
        print(f"âœ… Loaded slice {test_patient}_{test_slice}")
        print(f"   Paired input shape: {paired_input.shape}")
        print(f"   Combined mask shape: {combined_mask.shape}")
        print(f"   Mask unique values: {np.unique(combined_mask)}")
        
    except Exception as e:
        print(f"❌ Failed to load slice: {e}")
        return False
    
    # Test 3: Create TensorFlow dataset
    print("\n[TEST 3] Creating TensorFlow dataset...")
    try:
        dataset = loader.create_dataset_for_fold(
            fold_id=0,
            split='train',
            preprocessing='standard',
            class_scenario='binary',
            batch_size=2,
            shuffle=True
        )
        
        # Get first batch
        for batch_paired, batch_masks, _, _ in dataset.take(1):
            print(f"âœ… Created dataset")
            print(f"   Batch paired input shape: {batch_paired.shape}")
            print(f"   Batch masks shape: {batch_masks.shape}")
            print(f"   Paired input dtype: {batch_paired.dtype}")
            print(f"   Masks dtype: {batch_masks.dtype}")
            
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False
    
    print("\n" + "="*60)
    print("âœ… ALL TESTS PASSED")
    print("="*60)
    
    return True


###################### Main Execution ######################

if __name__ == "__main__":
    # Run tests
    success = test_data_loading()
    
    if success:
        print("\n" + "="*60)
        print("DATA LOADER READY FOR USE")
        print("="*60)
        print("\nNext steps:")
        print("1. Verify fold_assignments.json created in data_splits/")
        print("2. Check that all file paths are correct for your system")
        print("3. Proceed to model implementation")
    else:
        print("\n" + "="*60)
        print("❌ DATA LOADER TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before proceeding")

