"""
P4 Article - Run Multiple Variant Experiments
Updated runner script supporting all models

Supports:
- Variant 1: Baseline U-Net
- Variant 2: Attention U-Net
- Variant 3: DeepLabV3+ U-Net
- Variant 4: Trans U-Net

Usage:
    # Single experiment
    python p4_run_experiments_all.py --variant 2 --fold 0 --scenario standard_3class
    
    # All scenarios for one variant+fold
    python p4_run_experiments_all.py --variant 2 --fold 0
    
    # All scenarios for one variant (all folds)
    python p4_run_experiments_all.py --variant 2
    
    # All scenarios (all folds and all variants)
    python p4_run_experiments_all.py
"""

import sys
import argparse
import subprocess
from pathlib import Path
import tensorflow as tf
import gc
from tensorflow.keras import backend as K

import p4_unet_viz


def clear_gpu_memory():
    """Comprehensive GPU memory cleanup between experiments"""
    print("\n" + "="*70)
    print("CLEANING UP GPU MEMORY")
    print("="*70)
    
    # Clear Keras session
    K.clear_session()
    print("✅ Cleared Keras session")
    
    # Force garbage collection
    gc.collect()
    print("✅ Ran garbage collection")
    
    # Reset TensorFlow graphs
    tf.compat.v1.reset_default_graph()
    print("✅ Reset default graph")
    
    # Additional cleanup for TF 2.x
    try:
        # Clear any cached tensors
        tf.config.experimental.reset_memory_stats('GPU:0')
        print("✅ Reset GPU memory stats")
    except:
        pass
    
    print("="*70 + "\n")


def run_single_experiment(variant: int, 
                         preprocessing: str, 
                         class_scenario: str, 
                         fold_id: int) -> bool:
    """
    Run a single experiment for specified variant
    
    Args:
        variant: 1 (baseline u-net) or 2 (attention u-net) or 3 (deeplabv3+ u-net) or 4 (trans u-net)
        preprocessing: 'standard' or 'zoomed'
        class_scenario: '3class' or '4class'
        fold_id: 0-4
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(f"RUNNING: Variant {variant} | {preprocessing} | {class_scenario} | Fold {fold_id}")
    print("="*80 + "\n")
    
    try:
        if variant == 1:
            # Baseline unet
            from p4_variant_all_net import ExperimentConfig, train_net
            
            config = ExperimentConfig(
                variant=variant,
                preprocessing=preprocessing,
                class_scenario=class_scenario,
                fold_id=fold_id,
                architecture_name='unet'
            )
            
            history, history_path = train_net(config)
            p4_unet_viz.main_viz(history_path)

            # Run Inference
            from p4_inference import InferenceConfig, run_inference, run_error_analysis

            config = InferenceConfig(
                variant=variant,
                preprocessing=preprocessing,
                class_scenario=class_scenario,
                fold_id=fold_id,
                model_name='best_dice_model.h5',
                architecture_name='unet'
            )

            results = run_inference(config)
            
            # ── Error Analysis ──────────────────────────────────────
            error_results = run_error_analysis(
                results=results,
                config=config,
                top_n_slices=30,      # visualise N hardest slices
                top_n_patients=10,    # patient summary plots
                fg_dice_weight=0.6,   # tunable ranking weights
                error_rate_weight=0.2,
                confidence_weight=0.2,
            )

        elif variant == 2:
            # Attention unet
            from p4_variant_all_net import ExperimentConfig, train_net
            
            config = ExperimentConfig(
                variant=variant,
                preprocessing=preprocessing,
                class_scenario=class_scenario,
                fold_id=fold_id,
                architecture_name='attnunet'
            )
            
            history, history_path = train_net(config)
            p4_unet_viz.main_viz(history_path)

            # Run Inference
            from p4_inference import InferenceConfig, run_inference, run_error_analysis

            config = InferenceConfig(
                variant=variant,
                preprocessing=preprocessing,
                class_scenario=class_scenario,
                fold_id=fold_id,
                model_name='best_dice_model.h5',
                architecture_name='attnunet'
            )

            results = run_inference(config)
            
            # ── Error Analysis ──────────────────────────────────────
            error_results = run_error_analysis(
                results=results,
                config=config,
                top_n_slices=30,      # visualise N hardest slices
                top_n_patients=10,    # patient summary plots
                fg_dice_weight=0.6,   # tunable ranking weights
                error_rate_weight=0.2,
                confidence_weight=0.2,
            )

        elif variant == 3:
            # DeepLabV3+ unet
            from p4_variant_all_net import ExperimentConfig, train_net
            
            config = ExperimentConfig(
                variant=variant,
                preprocessing=preprocessing,
                class_scenario=class_scenario,
                fold_id=fold_id,
                architecture_name='dlv3unet'
            )
            
            history, history_path = train_net(config)
            p4_unet_viz.main_viz(history_path)

            # Run Inference
            from p4_inference import InferenceConfig, run_inference, run_error_analysis

            config = InferenceConfig(
                variant=variant,
                preprocessing=preprocessing,
                class_scenario=class_scenario,
                fold_id=fold_id,
                model_name='best_dice_model.h5',
                architecture_name='dlv3unet'
            )

            results = run_inference(config)
            
            # ── Error Analysis ──────────────────────────────────────
            error_results = run_error_analysis(
                results=results,
                config=config,
                top_n_slices=30,      # visualise N hardest slices
                top_n_patients=10,    # patient summary plots
                fg_dice_weight=0.6,   # tunable ranking weights
                error_rate_weight=0.2,
                confidence_weight=0.2,
            )

        elif variant == 4:
            # Trans unet
            from p4_variant_all_net import ExperimentConfig, train_net
            
            config = ExperimentConfig(
                variant=variant,
                preprocessing=preprocessing,
                class_scenario=class_scenario,
                fold_id=fold_id,
                architecture_name='transunet'
            )
            
            history, history_path = train_net(config)
            p4_unet_viz.main_viz(history_path)

            # Run Inference
            from p4_inference import InferenceConfig, run_inference, run_error_analysis

            config = InferenceConfig(
                variant=variant,
                preprocessing=preprocessing,
                class_scenario=class_scenario,
                fold_id=fold_id,
                model_name='best_dice_model.h5',
                architecture_name='transunet'
            )

            results = run_inference(config)
            
            # ── Error Analysis ──────────────────────────────────────
            error_results = run_error_analysis(
                results=results,
                config=config,
                top_n_slices=30,      # visualise N hardest slices
                top_n_patients=10,    # patient summary plots
                fg_dice_weight=0.6,   # tunable ranking weights
                error_rate_weight=0.2,
                confidence_weight=0.2,
            )

        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        print(f"\n✅ Experiment completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_scenarios_for_variant_fold(variant: int, fold_id: int) -> dict:
    """
    Run all 4 scenarios for a given variant and fold
    
    Args:
        variant: 1 (baseline u-net) or 2 (attention u-net) or 3 (deeplabv3+ u-net) or 4 (trans u-net)
        fold_id: 0-4
        
    Returns:
        Dictionary with results for each scenario
    """
    print("\n" + "="*80)
    print(f"RUNNING ALL SCENARIOS FOR VARIANT {variant}, FOLD {fold_id}")
    print("="*80)
    print("\nTotal experiments: 4")
    print("  1. standard + 3class")
    print("  2. standard + 4class")
    print("  3. zoomed + 3class")
    print("  4. zoomed + 4class")
    print("\n" + "="*80 + "\n")
    
    experiments = [
        {'preprocessing': 'zoomed', 'class_scenario': '4class'},
        {'preprocessing': 'standard', 'class_scenario': '4class'},
        {'preprocessing': 'zoomed', 'class_scenario': '3class'},
        {'preprocessing': 'standard', 'class_scenario': '3class'},
    ]
    
    results = {}
    
    for idx, scenario in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"SCENARIO {idx}/4: {scenario['preprocessing']} + {scenario['class_scenario']}")
        print(f"{'#'*80}\n")
        
        # Run in subprocess for complete memory isolation
        import subprocess
        import sys
        
        cmd = [
            sys.executable,
            'p4_run_experiments_all.py',
            '--variant', str(variant),
            '--fold', str(fold_id),
            '--scenario', f"{scenario['preprocessing']}_{scenario['class_scenario']}"
        ]
        
        print(f"Running command: {' '.join(cmd)}\n")

        try:
            # Run experiment in separate process
            result = subprocess.run(cmd, check=True, capture_output=False)
            
            if result.returncode == 0:
                exp_name = f"v{variant}_{scenario['preprocessing']}_{scenario['class_scenario']}_fold{fold_id}"
                results[exp_name] = {'status': 'SUCCESS'}
                print(f"\n✅ {exp_name} completed successfully")
            else:
                raise Exception(f"Process returned code {result.returncode}")
                
        except subprocess.CalledProcessError as e:
            exp_name = f"v{variant}_{scenario['preprocessing']}_{scenario['class_scenario']}_fold{fold_id}"
            print(f"\n❌ Error in {scenario['preprocessing']} + {scenario['class_scenario']}")
            print(f"   Error: {str(e)}")
            results[exp_name] = {
                'status': 'FAILED',
                'error': str(e)
            }

            # Ask user if they want to continue
            response = input("\nContinue with remaining experiments? (y/n): ")
            if response.lower() != 'y':
                print("Stopping experiments...")
                break

        # Brief pause between experiments
        import time
        print("\n⏳ Waiting 5 seconds before next experiment...")
        time.sleep(5)
    
    # Summary
    print("\n" + "="*80)
    print(f"VARIANT {variant}, FOLD {fold_id} - SUMMARY")
    print("="*80)
    
    for exp_name, result in results.items():
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {exp_name}")
    
    print("\n" + "="*80 + "\n")
    
    return results


def run_all_folds_for_variant(variant: int) -> dict:
    """
    Run all scenarios for all folds for a given variant
    Run all 4 experiments for all 5 folds
    Total: 4 scenarios × 5 folds = 20 training runs

    Args:
        variant: 1 (baseline u-net) or 2 (attention u-net) or 3 (deeplabv3+ u-net) or 4 (trans u-net)

    Returns:
        Dictionary with results for all folds
    """
    print("\n" + "="*80)
    print(f"RUNNING ALL FOLDS FOR VARIANT {variant}")
    print("="*80)
    print("\nTotal experiments: 4 scenarios × 5 folds = 20 training runs")
    print("Estimated time: ~0.7 hour per experiment (with 60 epochs)")
    print("Total estimated time: 10-20 hours")
    print("\n" + "="*80 + "\n")
    
    response = input("This will take a long time. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return {}
    
    all_results = {}
    
    for fold_id in range(5):
        print(f"\n{'='*80}")
        print(f"STARTING FOLD {fold_id}")
        print(f"{'='*80}\n")
        
        fold_results = run_all_scenarios_for_variant_fold(variant, fold_id)
        all_results[f'fold_{fold_id}'] = fold_results
    
    # Final summary
    print("\n" + "="*80)
    print(f"VARIANT {variant} - ALL FOLDS COMPLETE")
    print("="*80)
    
    for fold_id in range(5):
        fold_key = f'fold_{fold_id}'
        if fold_key in all_results:
            print(f"\nFold {fold_id}:")
            for exp_name, result in all_results[fold_key].items():
                status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
                print(f"  {status_icon} {exp_name}")
    
    print("\n" + "="*80 + "\n")
    
    return all_results


def compare_variants(fold_id: int = 0):
    """
    Compare results between baseline and attention variants and newloss variants
    
    Args:
        fold_id: Fold to compare (0-4)
    """
    print("\n" + "="*80)
    print(f"COMPARING VARIANTS FOR FOLD {fold_id}")
    print("="*80)
    
    import json
    
    scenarios = [
        {'preprocessing': 'standard', 'class_scenario': '3class'},
        {'preprocessing': 'standard', 'class_scenario': '4class'},
        {'preprocessing': 'zoomed', 'class_scenario': '3class'},
        {'preprocessing': 'zoomed', 'class_scenario': '4class'},
    ]
    
    results_dir = Path(f"results_fold_{fold_id}")
    
    for scenario in scenarios:
        print(f"\n{scenario['preprocessing']} + {scenario['class_scenario']}:")
        print("-" * 60)
        
        # Baseline (variant 1)
        baseline_dir = results_dir / "models" / f"{scenario['preprocessing']}_{scenario['class_scenario']}" / f"fold_{fold_id}"
        baseline_history = baseline_dir / "history.json"
        
        # Attention (variant 2)
        attention_dir = results_dir / "models" / f"{scenario['preprocessing']}_{scenario['class_scenario']}" / f"fold_{fold_id}_variant2"
        attention_history = attention_dir / "history.json"
        
        # Attention (variant 3)
        newloss_dir = results_dir / "models" / f"{scenario['preprocessing']}_{scenario['class_scenario']}" / f"fold_{fold_id}_variant3"
        newloss_history = newloss_dir / "history.json"
        
        if baseline_history.exists() and attention_history.exists() and newloss_history.exists():
            with open(baseline_history, 'r') as f:
                baseline_data = json.load(f)
            
            with open(attention_history, 'r') as f:
                attention_data = json.load(f)
            
            with open(newloss_history, 'r') as f:
                newloss_data = json.load(f)
            
            # Compare final validation losses
            baseline_val = baseline_data['val_loss'][-1]
            attention_val = attention_data['val_loss'][-1]
            newloss_val = newloss_data['val_loss'][-1]
            
            improvement_1_2 = ((baseline_val - attention_val) / baseline_val) * 100
            improvement_1_3 = ((baseline_val - newloss_val) / baseline_val) * 100
            improvement_2_3 = ((attention_val - newloss_val) / attention_val) * 100
            
            print(f"  Baseline Val Loss:  {baseline_val:.4f}")
            print(f"  Attention Val Loss: {attention_val:.4f}")
            print(f"  NewLoss Val Loss: {newloss_val:.4f}")
            print(f"  Improvement by V2 on V1:        {improvement_1_2:+.2f}%")
            print(f"  Improvement by V3 on V1:        {improvement_1_3:+.2f}%")
            print(f"  Improvement by V3 on V2:        {improvement_2_3:+.2f}%")
            
        else:
            if not baseline_history.exists():
                print(f"  ⚠️  Baseline results not found")
            if not attention_history.exists():
                print(f"  ⚠️  Attention results not found")
            if not newloss_history.exists():
                print(f"  ⚠️  NewLoss results not found")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run P4 experiments for multiple variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single experiment
    python p4_run_experiments_all.py --variant 2 --fold 0 --scenario standard_3class
    
    # All scenarios for variant 2, fold 0
    python p4_run_experiments_all.py --variant 2 --fold 0
    
    # All folds for variant 3
    python p4_run_experiments_all.py --variant 2
    
    # Compare results
    python p4_run_experiments_all.py --compare --fold 0
        """
    )
    
    parser.add_argument(
        '--variant',
        type=int,
        choices=[1, 2, 3, 4],
        help='variant: 1 (baseline u-net) or 2 (attention u-net) or 3 (deeplabv3+ u-net) or 4 (trans u-net)'
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        choices=[0, 1, 2, 3, 4],
        help='Specific fold to train (0-4)'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['standard_3class', 'standard_4class', 'zoomed_3class', 'zoomed_4class'],
        help='Specific scenario to train'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare results between variants'
    )
    
    args = parser.parse_args()
    
    # Handle comparison mode (NOT READY YET!)
    if args.compare:
        fold_id = args.fold if args.fold is not None else 0
        compare_variants(fold_id)
        return
    
    # Validate arguments
    if args.variant is None:
        parser.error("--variant is required (unless using --compare)")
    
    # Single experiment
    if args.scenario is not None:
        preprocessing, class_scenario = args.scenario.split('_')
        fold_id = args.fold if args.fold is not None else 0
        
        print(f"\nRunning single experiment:")
        print(f"  Variant: {args.variant}")
        print(f"  Fold: {fold_id}")
        print(f"  Preprocessing: {preprocessing}")
        print(f"  Class scenario: {class_scenario}\n")
        
        success = run_single_experiment(
            variant=args.variant,
            preprocessing=preprocessing,
            class_scenario=class_scenario,
            fold_id=fold_id
        )
        
        if success:
            print("\n✅ Experiment complete!")
        else:
            print("\n❌ Experiment failed!")
            sys.exit(1)
    
    # All scenarios for specific fold
    elif args.fold is not None:
        run_all_scenarios_for_variant_fold(args.variant, args.fold)
    
    # All scenarios for all folds
    else:
        run_all_folds_for_variant(args.variant)


if __name__ == "__main__":
    main()
