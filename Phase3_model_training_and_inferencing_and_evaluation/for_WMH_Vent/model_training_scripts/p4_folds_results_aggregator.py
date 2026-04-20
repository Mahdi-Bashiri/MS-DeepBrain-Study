"""
P4 - All U-Net models with Adaptive Loss (WCE + UFL)

WMH and Ventricles Segmentation with U-Net Models - Journal Paper Implementation
Three-class segmentation: Background vs Ventricles vs Abnormal WMH
Professional results saving and visualization for publication

This relates to our article:
"Deep Learning-Based Neuroanatomical Profiling Reveals Detailed Brain Changes:
A Large-Scale Multiple Sclerosis Study"

Features:
- Aggregatation of all inferenced results
- Includes lesion-level (connected-component) metrics: sensitivity, precision,
  F1, TP/FP/FN lesion counts (added to address reviewer R1C7)

Authors:
"Mahdi Bashiri Bawil, Mousa Shamsi, Abolhassan Shakeri Bavil"

Developer:
"Mahdi Bashiri Bawil"
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ResultsAggregator:
    """
    Aggregates segmentation results across multiple variants and folds.
    """
    
    def __init__(self, base_dir='./'):
        """
        Initialize the aggregator.
        
        Args:
            base_dir: Base directory containing all results folders
        """
        self.base_dir = Path(base_dir)
        self.variants = {
            1: "unet",
            2: "attnunet",
            3: "dlv3unet",
            4: "transunet"
        }
        self.class_names = ["Background", "Ventricles", "Abnormal_WMH"]
        self.num_variants = 4
        self.num_folds = 4
        
    def find_results_folders(self):
        """Find all results folders matching the naming pattern."""
        results_folders = []
        for variant in range(self.num_variants):
            for fold in range(self.num_folds):
                folder_pattern = f"results_fold_{fold}_var_{variant+1}_zscore2"
                folder_path = self.base_dir / folder_pattern
                if folder_path.exists():
                    results_folders.append({
                        'variant': variant+1,
                        'fold': fold,
                        'path': folder_path
                    })
        return results_folders
    
    def load_test_metrics(self, results_folder):
        """Load test metrics from JSON file."""
        metrics_path = results_folder['path'] / 'inference_all_test' / 'standard_3class' / 'metrics' / 'test_metrics_complete.json'
        
        if not metrics_path.exists():
            print(f"Warning: Metrics file not found at {metrics_path}")
            return None
            
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def load_training_summary(self, results_folder):
        """Load training summary from JSON file (new format)."""
        summary_path = results_folder['path'] / 'models' / 'standard_3class' / f"fold_{results_folder['fold']}" / 'training_summary.json'
        
        if not summary_path.exists():
            # Fallback to history.json if training_summary doesn't exist
            return self.load_training_history(results_folder)
            
        with open(summary_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def load_training_history(self, results_folder):
        """Load training history from JSON file (legacy support)."""
        history_path = results_folder['path'] / 'models' / 'standard_3class' / f"fold_{results_folder['fold']}" / 'history.json'
        
        if not history_path.exists():
            print(f"Warning: History file not found at {history_path}")
            return None
            
        with open(history_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def load_best_epoch_analysis(self, results_folder):
        """Load best epoch analysis from JSON file (new format)."""
        analysis_path = results_folder['path'] / 'models' / 'standard_3class' / f"fold_{results_folder['fold']}" / 'best_epoch_analysis.json'
        
        if not analysis_path.exists():
            return None
            
        with open(analysis_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def extract_test_metrics_row(self, results_folder, metrics_data):
        """
        Extract a row of test metrics for the summary dataframe.
        Includes both voxel-level and lesion-level metrics.
        """
        if metrics_data is None:
            return None
        
        row = {
            'Variant': results_folder['variant'],
            'Variant_Name': self.variants[results_folder['variant']],
            'Fold': results_folder['fold'],
            'Test_Samples': metrics_data['config']['test_samples']
        }
        
        # ── Voxel-level metrics (unchanged) ─────────────────────────────────
        for metric_name in ['dice', 'precision', 'recall', 'iou', 'specificity', 'hd95']:
            metric_data = metrics_data['metrics'][metric_name]
            
            for class_idx in range(3):
                if class_idx != 0:
                    row[f'{metric_name.upper()}_class_{class_idx}'] = metric_data.get(f'class_{class_idx}')
            
            row[f'{metric_name.upper()}_mean'] = metric_data.get('mean')
        
        # ── Lesion-level metrics (new — R1C7) ────────────────────────────────
        lesion_data = metrics_data['metrics'].get('lesion', None)
        if lesion_data is not None:
            for class_idx in range(2):   # foreground classes only
                key = f'class_{class_idx}'
                cls = lesion_data.get(key, {})

                # Scalar rates (averaged across patients in inference script)
                for sk in ['lesion_sensitivity', 'lesion_precision', 'lesion_f1']:
                    col = f'LESION_{sk.upper()}_class_{class_idx}'
                    row[col] = cls.get(sk)

                # Integer counts (summed across patients in inference script)
                for ck in ['n_gt_lesions', 'n_pred_lesions', 'tp_lesions', 'fn_lesions', 'fp_lesions']:
                    col = f'LESION_{ck.upper()}_class_{class_idx}'
                    row[col] = cls.get(ck)

            # Cross-class summary keys produced by aggregate_patient_metrics()
            for sk in ['lesion_sensitivity', 'lesion_precision', 'lesion_f1']:
                row[f'LESION_{sk.upper()}_mean'] = lesion_data.get(f'mean_{sk}')
            for ck in ['n_gt_lesions', 'n_pred_lesions', 'tp_lesions', 'fn_lesions', 'fp_lesions']:
                row[f'LESION_{ck.upper()}_total'] = lesion_data.get(f'total_{ck}')
        
        return row
    
    def extract_training_info_row(self, results_folder, training_data, best_epoch_analysis):
        """Extract training information including best epoch details."""
        if training_data is None:
            return None
        
        row = {
            'Variant': results_folder['variant'],
            'Variant_Name': self.variants[results_folder['variant']],
            'Fold': results_folder['fold']
        }
        
        # Try to extract from training_summary.json first
        if isinstance(training_data, dict) and 'best_epoch_selection' in training_data:
            row['Best_Epoch'] = training_data['best_epoch_selection']['overall_best_epoch']
            row['Composite_Score'] = training_data['best_epoch_selection']['composite_score']
            row['Total_Epochs'] = training_data['training_config']['total_epochs']
            # Handle valid_epochs (only for Pix2Pix variants with beta scheduling)
            if 'valid_epochs' in training_data['best_epoch_selection']:
                row['First_Valid_Epoch'] = training_data['best_epoch_selection']['valid_epochs']['first_valid_epoch']
                row['Total_Valid_Epochs'] = training_data['best_epoch_selection']['valid_epochs']['total_valid_epochs']
            else:
                row['First_Valid_Epoch'] = 1
                row['Total_Valid_Epochs'] = training_data['training_config']['total_epochs']

            # Best epoch metrics
            best_metrics = training_data['best_epoch_metrics']
            row['Best_Epoch_Val_Loss'] = best_metrics['val_loss']
            row['Best_Epoch_Dice_Ventricles'] = best_metrics['dice']['class_1']
            row['Best_Epoch_Dice_Abnormal_WMH'] = best_metrics['dice'].get('class_2', None)
            row['Best_Epoch_Dice_Mean'] = best_metrics['dice']['mean']
            
            # Priority metrics
            row['Best_Abnormal_Epoch'] = training_data['priority_metrics']['abnormal_wmh']['best_epoch']
            row['Best_Abnormal_Dice'] = training_data['priority_metrics']['abnormal_wmh']['best_dice']
            row['Best_Ventricles_Epoch'] = training_data['priority_metrics']['ventricles']['best_epoch']
            row['Best_Ventricles_Dice'] = training_data['priority_metrics']['ventricles']['best_dice']
            
        # Fallback to best_epoch_analysis.json
        elif best_epoch_analysis is not None:
            row['Best_Epoch'] = best_epoch_analysis['best_overall_epoch']
            row['Composite_Score'] = best_epoch_analysis['composite_score']
            row['Total_Epochs'] = best_epoch_analysis['total_epochs']
            row['First_Valid_Epoch'] = best_epoch_analysis['first_valid_epoch']
            row['Total_Valid_Epochs'] = best_epoch_analysis['total_valid_epochs']
            
            # Best epoch metrics
            best_metrics = best_epoch_analysis['best_epoch_metrics']
            row['Best_Epoch_Val_Loss'] = best_metrics['val_loss']
            row['Best_Epoch_Dice_Ventricles'] = best_metrics['dice']['class_1']
            row['Best_Epoch_Dice_Abnormal_WMH'] = best_metrics['dice'].get('class_2', None)
            row['Best_Epoch_Dice_Mean'] = best_metrics['dice']['mean']
            
            # Priority metrics
            row['Best_Abnormal_Epoch'] = best_epoch_analysis['best_abnormal_epoch']
            row['Best_Abnormal_Dice'] = best_epoch_analysis['best_abnormal_dice']
            row['Best_Ventricles_Epoch'] = best_epoch_analysis['best_ventricles_epoch']
            row['Best_Ventricles_Dice'] = best_epoch_analysis['best_ventricles_dice']
            
        # Legacy fallback to history.json
        elif isinstance(training_data, dict) and 'val_metrics' in training_data:
            if 'best_epoch_analysis' in training_data:
                analysis = training_data['best_epoch_analysis']
                row['Best_Epoch'] = analysis['best_overall_epoch']
                row['Composite_Score'] = analysis.get('composite_score', None)
            else:
                # Find best validation dice
                val_dice_list = [m['dice']['mean'] for m in training_data['val_metrics']]
                row['Best_Epoch'] = val_dice_list.index(max(val_dice_list)) + 1
                row['Composite_Score'] = max(val_dice_list)
            
            row['Total_Epochs'] = len(training_data['val_metrics'])
        
        return row
    
    def create_test_metrics_summary(self):
        """Create a comprehensive summary of test metrics."""
        results_folders = self.find_results_folders()
        
        if not results_folders:
            print("No results folders found!")
            return None
        
        rows = []
        for folder in results_folders:
            metrics_data = self.load_test_metrics(folder)
            row = self.extract_test_metrics_row(folder, metrics_data)
            if row is not None:
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['Variant', 'Fold']).reset_index(drop=True)
        
        return df
    
    def create_training_summary(self):
        """Create a comprehensive summary of training information."""
        results_folders = self.find_results_folders()
        
        if not results_folders:
            print("No results folders found!")
            return None
        
        rows = []
        for folder in results_folders:
            training_data = self.load_training_summary(folder)
            best_epoch_analysis = self.load_best_epoch_analysis(folder)
            row = self.extract_training_info_row(folder, training_data, best_epoch_analysis)
            if row is not None:
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['Variant', 'Fold']).reset_index(drop=True)
        
        return df
    
    def create_per_class_summary(self, test_metrics_df):
        """
        Create per-class summary statistics across folds for each variant.
        Includes both voxel-level and lesion-level metrics.
        """
        summaries = []
        
        for variant in range(self.num_variants +1):
            variant_data = test_metrics_df[test_metrics_df['Variant'] == variant]
            
            if len(variant_data) == 0:
                continue
            
            for class_idx in range(3):
                if class_idx == 0:
                    continue

                class_summary = {
                    'Variant': variant,
                    'Variant_Name': self.variants[variant],
                    'Class': class_idx,
                    'Class_Name': self.class_names[class_idx]
                }
                
                # Voxel-level metrics
                for metric in ['DICE', 'PRECISION', 'RECALL', 'IOU', 'SPECIFICITY', 'HD95']:
                    col_name = f'{metric}_class_{class_idx}'
                    if col_name in variant_data.columns:
                        values = variant_data[col_name].dropna().values
                        class_summary[f'{metric}_mean'] = np.mean(values)
                        class_summary[f'{metric}_std']  = np.std(values)
                        class_summary[f'{metric}_min']  = np.min(values)
                        class_summary[f'{metric}_max']  = np.max(values)
                
                # Lesion-level scalar metrics (mean ± std across folds)
                for sk in ['LESION_SENSITIVITY', 'LESION_PRECISION', 'LESION_F1']:
                    col_name = f'LESION_{sk}_class_{class_idx}'
                    if col_name in variant_data.columns:
                        values = variant_data[col_name].dropna().values
                        class_summary[f'{sk}_mean'] = np.mean(values) if len(values) else np.nan
                        class_summary[f'{sk}_std']  = np.std(values)  if len(values) else np.nan

                # Lesion-level count metrics (sum across folds — total pool)
                for ck in ['N_GT_LESIONS', 'N_PRED_LESIONS', 'TP_LESIONS', 'FN_LESIONS', 'FP_LESIONS']:
                    col_name = f'LESION_{ck}_class_{class_idx}'
                    if col_name in variant_data.columns:
                        values = variant_data[col_name].dropna().values
                        class_summary[f'LESION_{ck}_total'] = int(np.sum(values)) if len(values) else 0

                summaries.append(class_summary)
        
        df = pd.DataFrame(summaries)
        return df
    
    def create_variant_comparison(self, test_metrics_df):
        """
        Create a variant comparison table with mean ± std across folds.
        Includes both voxel-level and lesion-level metrics.
        """
        comparisons = []
        
        for variant in range(self.num_variants + 1):
            variant_data = test_metrics_df[test_metrics_df['Variant'] == variant]
            
            if len(variant_data) == 0:
                continue
            
            comparison = {
                'Variant': variant,
                'Variant_Name': self.variants[variant],
                'N_Folds': len(variant_data)
            }
            
            # ── Voxel-level metrics ──────────────────────────────────────────
            for metric in ['DICE', 'PRECISION', 'RECALL', 'IOU', 'SPECIFICITY', 'HD95']:
                # Overall mean across classes
                col_name = f'{metric}_mean'
                if col_name in variant_data.columns:
                    values = variant_data[col_name].dropna().values
                    comparison[f'{metric}_Mean'] = np.mean(values)
                    comparison[f'{metric}_Std']  = np.std(values)
                
                # Per-class (Ventricles=1, Abnormal_WMH=2)
                for class_idx in [1, 2]:
                    col_name = f'{metric}_class_{class_idx}'
                    if col_name in variant_data.columns:
                        values = variant_data[col_name].dropna().values
                        comparison[f'{metric}_Class{class_idx}_Mean'] = np.mean(values)
                        comparison[f'{metric}_Class{class_idx}_Std']  = np.std(values)
            
            # ── Lesion-level scalar metrics (mean ± std across folds) ────────
            for sk_suffix in ['LESION_SENSITIVITY', 'LESION_PRECISION', 'LESION_F1']:
                # Cross-class mean
                col_name = f'LESION_{sk_suffix}_mean'
                if col_name in variant_data.columns:
                    values = variant_data[col_name].dropna().values
                    comparison[f'{sk_suffix}_Mean'] = np.mean(values) if len(values) else np.nan
                    comparison[f'{sk_suffix}_Std']  = np.std(values)  if len(values) else np.nan

                # Per-class
                for class_idx in [2]:
                    col_name = f'LESION_{sk_suffix}_class_{class_idx}'
                    if col_name in variant_data.columns:
                        values = variant_data[col_name].dropna().values
                        comparison[f'{sk_suffix}_Class{class_idx}_Mean'] = np.mean(values) if len(values) else np.nan
                        comparison[f'{sk_suffix}_Class{class_idx}_Std']  = np.std(values)  if len(values) else np.nan

            # ── Lesion-level count metrics (sum across folds) ────────────────
            for ck in ['N_GT_LESIONS', 'N_PRED_LESIONS', 'TP_LESIONS', 'FN_LESIONS', 'FP_LESIONS']:
                # Total across all classes
                col_name = f'LESION_{ck}_total'
                if col_name in variant_data.columns:
                    values = variant_data[col_name].dropna().values
                    comparison[f'LESION_{ck}_Total'] = int(np.sum(values)) if len(values) else 0

                # Per-class totals
                for class_idx in [2]:
                    col_name = f'LESION_{ck}_class_{class_idx}'
                    if col_name in variant_data.columns:
                        values = variant_data[col_name].dropna().values
                        comparison[f'LESION_{ck}_Class{class_idx}_Total'] = int(np.sum(values)) if len(values) else 0

            comparisons.append(comparison)
        
        df = pd.DataFrame(comparisons)
        return df
    
    def create_training_comparison(self, training_df):
        """Create training comparison showing convergence patterns."""
        if training_df is None:
            return None
        
        comparisons = []
        
        for variant in range(self.num_variants + 1):
            variant_data = training_df[training_df['Variant'] == variant]
            
            if len(variant_data) == 0:
                continue
            
            comparison = {
                'Variant': variant,
                'Variant_Name': self.variants[variant],
                'N_Folds': len(variant_data)
            }
            
            # Best epoch statistics
            if 'Best_Epoch' in variant_data.columns:
                comparison['Best_Epoch_Mean'] = np.mean(variant_data['Best_Epoch'].values)
                comparison['Best_Epoch_Std']  = np.std(variant_data['Best_Epoch'].values)
                comparison['Best_Epoch_Min']  = np.min(variant_data['Best_Epoch'].values)
                comparison['Best_Epoch_Max']  = np.max(variant_data['Best_Epoch'].values)
            
            # Composite score statistics
            if 'Composite_Score' in variant_data.columns:
                comparison['Composite_Score_Mean'] = np.mean(variant_data['Composite_Score'].dropna().values)
                comparison['Composite_Score_Std']  = np.std(variant_data['Composite_Score'].dropna().values)
            
            # Validation metrics at best epoch
            for metric_col in ['Best_Epoch_Val_Loss', 'Best_Epoch_Dice_Mean', 
                              'Best_Epoch_Dice_Ventricles', 'Best_Epoch_Dice_Abnormal_WMH']:
                if metric_col in variant_data.columns:
                    values = variant_data[metric_col].dropna().values
                    if len(values) > 0:
                        comparison[f'{metric_col}_Mean'] = np.mean(values)
                        comparison[f'{metric_col}_Std']  = np.std(values)
            
            comparisons.append(comparison)
        
        df = pd.DataFrame(comparisons)
        return df
    
    def generate_all_summaries(self, output_dir='./folds_results'):
        """Generate all summary CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("RESULTS AGGREGATION STARTED")
        print("=" * 80)
        
        # 1. Test Metrics Summary (all variants, all folds)
        print("\n1. Generating test metrics summary...")
        test_metrics_df = self.create_test_metrics_summary()
        if test_metrics_df is not None:
            output_file = output_path / 'test_metrics_all_variants_folds.csv'
            test_metrics_df.to_csv(output_file, index=False)
            print(f"   ✓ Saved: {output_file}")
            print(f"   - Shape: {test_metrics_df.shape}")
        
        # 2. Training Summary
        print("\n2. Generating training summary...")
        training_df = self.create_training_summary()
        if training_df is not None:
            output_file = output_path / 'training_info_all_variants_folds.csv'
            training_df.to_csv(output_file, index=False)
            print(f"   ✓ Saved: {output_file}")
            print(f"   - Shape: {training_df.shape}")
        
        # 3. Per-Class Summary
        print("\n3. Generating per-class summary...")
        per_class_df = None
        if test_metrics_df is not None:
            per_class_df = self.create_per_class_summary(test_metrics_df)
            output_file = output_path / 'per_class_summary.csv'
            per_class_df.to_csv(output_file, index=False)
            print(f"   ✓ Saved: {output_file}")
            print(f"   - Shape: {per_class_df.shape}")
        
        # 4. Variant Comparison (Test Metrics)
        print("\n4. Generating variant comparison (test metrics)...")
        variant_comparison_df = None
        if test_metrics_df is not None:
            variant_comparison_df = self.create_variant_comparison(test_metrics_df)
            output_file = output_path / 'variant_comparison_test.csv'
            variant_comparison_df.to_csv(output_file, index=False)
            print(f"   ✓ Saved: {output_file}")
            print(f"   - Shape: {variant_comparison_df.shape}")
        
        # 5. Variant Comparison (Training)
        print("\n5. Generating variant comparison (training)...")
        training_comparison_df = None
        if training_df is not None:
            training_comparison_df = self.create_training_comparison(training_df)
            if training_comparison_df is not None:
                output_file = output_path / 'variant_comparison_training.csv'
                training_comparison_df.to_csv(output_file, index=False)
                print(f"   ✓ Saved: {output_file}")
                print(f"   - Shape: {training_comparison_df.shape}")
        
        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        
        return {
            'test_metrics': test_metrics_df,
            'training_info': training_df,
            'per_class': per_class_df,
            'variant_comparison_test': variant_comparison_df,
            'variant_comparison_training': training_comparison_df
        }
    
    def print_summary_statistics(self, dfs):
        """Print summary statistics to console."""
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        if dfs['variant_comparison_test'] is not None:

            # ── Voxel-level Dice ─────────────────────────────────────────────
            print("\n📊 TEST DICE SCORES (Mean ± Std) across folds:")
            print("-" * 80)
            for _, row in dfs['variant_comparison_test'].iterrows():
                print(f"\nVariant {row['Variant']}: {row['Variant_Name']}")
                print(f"  Overall:        {row['DICE_Mean']:.4f} ± {row['DICE_Std']:.4f}")
                print(f"  Ventricles:     {row['DICE_Class1_Mean']:.4f} ± {row['DICE_Class1_Std']:.4f}")
                print(f"  Abnormal WMH:   {row['DICE_Class2_Mean']:.4f} ± {row['DICE_Class2_Std']:.4f}")

            # ── Lesion-level metrics ─────────────────────────────────────────
            lesion_cols_present = any(
                col.startswith('LESION_') for col in dfs['variant_comparison_test'].columns
            )
            if lesion_cols_present:
                print("\n\n🔬 LESION-LEVEL METRICS (Mean ± Std) across folds:")
                print("-" * 80)
                for _, row in dfs['variant_comparison_test'].iterrows():
                    print(f"\nVariant {row['Variant']}: {row['Variant_Name']}")

                    # Per-class
                    for class_idx, class_name in [(2, 'Abnormal WMH')]:
                        sens_col  = f'LESION_LESION_SENSITIVITY_Class{class_idx}_Mean'
                        prec_col  = f'LESION_LESION_PRECISION_Class{class_idx}_Mean'
                        f1_col    = f'LESION_LESION_F1_Class{class_idx}_Mean'
                        tp_col    = f'LESION_TP_LESIONS_Class{class_idx}_Total'
                        fp_col    = f'LESION_FP_LESIONS_Class{class_idx}_Total'
                        fn_col    = f'LESION_FN_LESIONS_Class{class_idx}_Total'
                        gt_col    = f'LESION_N_GT_LESIONS_Class{class_idx}_Total'

                        print(f"  [{class_name}]")
                        if sens_col in row:
                            s_m  = f"{row[sens_col]:.4f}" if pd.notna(row.get(sens_col)) else 'N/A'
                            s_s  = f"{row.get(f'LESION_LESION_SENSITIVITY_Class{class_idx}_Std', float('nan')):.4f}"
                            p_m  = f"{row[prec_col]:.4f}" if pd.notna(row.get(prec_col)) else 'N/A'
                            p_s  = f"{row.get(f'LESION_LESION_PRECISION_Class{class_idx}_Std', float('nan')):.4f}"
                            f_m  = f"{row[f1_col]:.4f}"  if pd.notna(row.get(f1_col))   else 'N/A'
                            f_s  = f"{row.get(f'LESION_LESION_F1_Class{class_idx}_Std', float('nan')):.4f}"
                            print(f"    Sensitivity : {s_m} ± {s_s}")
                            print(f"    Precision   : {p_m} ± {p_s}")
                            print(f"    F1          : {f_m} ± {f_s}")
                        if gt_col in row:
                            print(f"    GT Lesions  : {int(row.get(gt_col, 0))}   "
                                  f"TP: {int(row.get(tp_col, 0))}   "
                                  f"FP: {int(row.get(fp_col, 0))}   "
                                  f"FN: {int(row.get(fn_col, 0))}")

        if dfs['variant_comparison_training'] is not None:
            print("\n\n🏆 TRAINING CONVERGENCE:")
            print("-" * 80)
            for _, row in dfs['variant_comparison_training'].iterrows():
                print(f"\nVariant {row['Variant']}: {row['Variant_Name']}")
                if 'Best_Epoch_Mean' in row:
                    print(f"  Best Epoch:     {row['Best_Epoch_Mean']:.1f} ± {row['Best_Epoch_Std']:.1f}")
                if 'Best_Epoch_Dice_Abnormal_WMH_Mean' in row:
                    print(f"  Val Abnormal:   {row['Best_Epoch_Dice_Abnormal_WMH_Mean']:.4f} ± {row['Best_Epoch_Dice_Abnormal_WMH_Std']:.4f}")


# Main execution
if __name__ == "__main__":
    # Initialize aggregator
    aggregator = ResultsAggregator(base_dir='./')
    
    # Generate all summaries
    dfs = aggregator.generate_all_summaries(output_dir='./folds_results_zscore2_all')
    
    # Print summary statistics
    aggregator.print_summary_statistics(dfs)
    
    print("\n✓ All CSV files have been generated in './folds_results_zscore2_all' directory")
    print("\nGenerated files:")
    print("  1. test_metrics_all_variants_folds.csv - Complete test metrics (voxel + lesion level)")
    print("  2. training_info_all_variants_folds.csv - Training convergence info")
    print("  3. per_class_summary.csv - Per-class statistics (voxel + lesion level)")
    print("  4. variant_comparison_test.csv - Test metrics comparison (voxel + lesion level)")
    print("  5. variant_comparison_training.csv - Training comparison")