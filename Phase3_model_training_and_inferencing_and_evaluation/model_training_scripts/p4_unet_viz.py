"""
P4 - All U-Net models with Adaptive Loss (WCE + UFL)

WMH and Ventricles Segmentation with U-Net Models - Journal Paper Implementation
Three-class segmentation: Background vs Ventricles vs Abnormal WMH
Professional results saving and visualization for publication

This relates to our article:
"Deep Learning-Based Neuroanatomical Profiling Reveals Detailed Brain Changes:
A Large-Scale Multiple Sclerosis Study"

Features:
- Visualization of Results

Authors:
"Mahdi Bashiri Bawil, Mousa Shamsi, Abolhassan Shakeri Bavil"

Developer:
"Mahdi Bashiri Bawil"
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_history(filepath):
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def detect_num_classes(history):
    """Detect number of classes from val_metrics."""
    if not history['val_metrics']:
        return 3
    first_metric = history['val_metrics'][0]
    # Count only class_X keys, not 'mean'
    num_classes = len([k for k in first_metric['dice'].keys() if k.startswith('class_')])
    return num_classes

def get_class_names(num_classes):
    """Get class names based on number of classes."""
    if num_classes == 3:
        return {
            'class_0': 'Background',
            'class_1': 'Ventricles',
            'class_2': 'Abnormal WMH'
        }
    elif num_classes == 4:
        return {
            'class_0': 'Background',
            'class_1': 'Ventricles',
            'class_2': 'Normal WMH',
            'class_3': 'Abnormal WMH'
        }
    else:
        return {f'class_{i}': f'Class {i}' for i in range(num_classes)}

def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj

def find_best_epoch(history, num_classes):
    """
    Find the best epoch based on prioritized criteria:
    1. Highest Dice for abnormal WMH (top priority)
    2. Highest Dice for ventricles (secondary)
    3. Lowest validation loss (tertiary)
    4. ONLY consider epochs where beta > 0.95 (CRITICAL REQUIREMENT)

    """
    if not history['val_metrics']:
        return None, {}
    
    epochs = range(1, len(history['val_metrics']) + 1)
    if 'beta_value' in history:
        beta_values = history['beta_value']
    else:
        beta_values = [1] * len(history.get('val_loss', []))
        history['beta_value'] = beta_values

    # Find epochs where beta > 0.95 (CRITICAL FILTER)
    valid_epoch_indices = [i for i, beta in enumerate(beta_values) if beta > 0.95]

    if not valid_epoch_indices:
        print("⚠️  WARNING: No epochs found with beta > 0.95!")
        print("    Using all epochs for analysis (not recommended).")
        valid_epoch_indices = list(range(len(beta_values)))
    
    first_valid_epoch = valid_epoch_indices[0] + 1 if valid_epoch_indices else 1

    # Determine the key for abnormal WMH
    abnormal_key = 'class_3' if num_classes == 4 else 'class_2'
    ventricles_key = 'class_1'
    
    # Extract metrics
    abnormal_dice = [m['dice'][abnormal_key] for m in history['val_metrics']]
    ventricles_dice = [m['dice'][ventricles_key] for m in history['val_metrics']]
    val_losses = history['val_loss']
    
    # Find best epoch for abnormal WMH dice (only among valid epochs)
    valid_abnormal_dice = [(i, abnormal_dice[i]) for i in valid_epoch_indices]
    best_abnormal_idx = max(valid_abnormal_dice, key=lambda x: x[1])[0]
    best_abnormal_epoch = best_abnormal_idx + 1
    best_abnormal_dice = abnormal_dice[best_abnormal_idx]
    
    # Find best epoch for ventricles dice (only among valid epochs)
    valid_ventricles_dice = [(i, ventricles_dice[i]) for i in valid_epoch_indices]
    best_ventricles_idx = max(valid_ventricles_dice, key=lambda x: x[1])[0]
    best_ventricles_epoch = best_ventricles_idx + 1
    best_ventricles_dice = ventricles_dice[best_ventricles_idx]
    
    # Find best epoch for validation loss (only among valid epochs)
    valid_val_losses = [(i, val_losses[i]) for i in valid_epoch_indices]
    best_val_loss_idx = min(valid_val_losses, key=lambda x: x[1])[0]
    best_val_loss_epoch = best_val_loss_idx + 1
    best_val_loss = val_losses[best_val_loss_idx]
    
    # Calculate composite score (weighted) - ONLY for valid epochs
    composite_scores = [float('-inf')] * len(abnormal_dice)
    
    for i in valid_epoch_indices:
        # Normalize and weight: 60% abnormal dice, 30% ventricles dice, 10% inv val_loss
        norm_abnormal = abnormal_dice[i]
        norm_ventricles = ventricles_dice[i]
        
        # Normalize validation loss among valid epochs only
        valid_val_loss_values = [val_losses[j] for j in valid_epoch_indices]
        max_val_loss = max(valid_val_loss_values) if valid_val_loss_values else 1
        norm_val_loss = 1 - (val_losses[i] / max_val_loss) if max_val_loss > 0 else 0
        
        composite = 0.6 * norm_abnormal + 0.3 * norm_ventricles + 0.1 * (1 - val_losses[i]) # norm_val_loss
        composite_scores[i] = composite
    
    best_overall_idx = int(np.argmax(composite_scores))  # Convert to int
    best_overall_epoch = best_overall_idx + 1
    
    # Get all metrics at best epoch
    best_epoch_metrics = history['val_metrics'][best_overall_idx]
    
    analysis = {
        'best_overall_epoch': int(best_overall_epoch),
        'best_overall_epoch_idx': int(best_overall_idx),
        'best_abnormal_epoch': int(best_abnormal_epoch),
        'best_abnormal_dice': float(best_abnormal_dice),
        'best_ventricles_epoch': int(best_ventricles_epoch),
        'best_ventricles_dice': float(best_ventricles_dice),
        'best_val_loss_epoch': int(best_val_loss_epoch),
        'best_val_loss': float(best_val_loss),
        'composite_score': float(composite_scores[best_overall_idx]),
        'abnormal_key': abnormal_key,
        'num_classes': int(num_classes),
        'first_valid_epoch': int(first_valid_epoch),
        'total_valid_epochs': int(len(valid_epoch_indices)),
        'beta_threshold': 0.95,
        'total_epochs': int(len(epochs)),
        # Add complete metrics at best epoch
        'best_epoch_metrics': {
            'dice': best_epoch_metrics['dice'],
            'precision': best_epoch_metrics['precision'],
            'recall': best_epoch_metrics['recall'],
            'val_loss': float(val_losses[best_overall_idx]),
            'train_loss': float(history['train_loss'][best_overall_idx]),
            'wce_loss': float(history['wce_loss'][best_overall_idx]),
            'ufd_loss': float(history['ufd_loss'][best_overall_idx]),
            'val_loss_wce': float(history['val_loss_wce'][best_overall_idx]) if 'val_loss_wce' in history else None,
            'val_loss_ufd': float(history['val_loss_ufd'][best_overall_idx]) if 'val_loss_ufd' in history else None,
            'beta_value': float(beta_values[best_overall_idx])
        }
    }

    # Convert all numpy types to native Python types
    analysis = convert_to_native_types(analysis)
    
    return best_overall_epoch, analysis

def save_analysis_json(analysis, output_path):
    """Save analysis results to a JSON file."""
    analysis = convert_to_native_types(analysis)
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Analysis saved to: {output_path}")

def save_enhanced_history(history, analysis, output_path):
    """Save enhanced history with best epoch analysis appended."""
    enhanced_history = history.copy()
    enhanced_history['best_epoch_analysis'] = convert_to_native_types(analysis)
    enhanced_history = convert_to_native_types(enhanced_history)
    
    with open(output_path, 'w') as f:
        json.dump(enhanced_history, f, indent=2)
    print(f"✓ Enhanced history saved to: {output_path}")

def create_training_summary(history, analysis, class_names):
    """Create a comprehensive training summary for easy parsing."""
    summary = {
        'training_config': {
            'total_epochs': analysis['total_epochs'],
            'num_classes': analysis['num_classes'],
            'class_names': class_names,
            'model_type': 'a U-Net'
        },
        'best_epoch_selection': {
            'overall_best_epoch': analysis['best_overall_epoch'],
            'composite_score': analysis['composite_score'],
            'selection_criteria': {
                'abnormal_wmh_weight': 0.6,
                'ventricles_weight': 0.3,
                'val_loss_weight': 0.1
            }
        },
        'priority_metrics': {
            'abnormal_wmh': {
                'best_epoch': analysis['best_abnormal_epoch'],
                'best_dice': analysis['best_abnormal_dice']
            },
            'ventricles': {
                'best_epoch': analysis['best_ventricles_epoch'],
                'best_dice': analysis['best_ventricles_dice']
            },
            'validation_loss': {
                'best_epoch': analysis['best_val_loss_epoch'],
                'best_loss': analysis['best_val_loss']
            }
        },
        'best_epoch_metrics': analysis['best_epoch_metrics'],
        'training_progression': {
            'final_epoch_metrics': {
                'dice': history['val_metrics'][-1]['dice'],
                'precision': history['val_metrics'][-1]['precision'],
                'recall': history['val_metrics'][-1]['recall'],
                'val_loss': history['val_loss'][-1],
                'train_loss': history['train_loss'][-1]
            },
            'convergence_info': {
                'epochs_trained': len(history['val_loss'])
            }
        }
    }
    
    # Add epoch-by-epoch metrics for important classes
    summary['epoch_progression'] = {
        'abnormal_wmh_dice': [m['dice'][analysis['abnormal_key']] for m in history['val_metrics']],
        'ventricles_dice': [m['dice']['class_1'] for m in history['val_metrics']],
        'mean_dice': [m['dice']['mean'] for m in history['val_metrics']],
        'val_loss': history['val_loss'],
        'train_loss': history['train_loss']
    }
    
    summary = convert_to_native_types(summary)

    return summary

def plot_training_history(history, save_path='training_history.png'):
    """Create comprehensive visualization of training history."""
    
    num_classes = detect_num_classes(history)
    class_names = get_class_names(num_classes)
    best_epoch, analysis = find_best_epoch(history, num_classes)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Detect whether new-style history (with val_loss_wce / val_loss_ufd) is present
    has_val_components = 'val_loss_wce' in history and 'val_loss_ufd' in history

    # Create figure — 3 rows × 3 cols when val components exist, else 2×3
    nrows = 3 if has_val_components else 2
    fig = plt.figure(figsize=(18, nrows * 5))
    gs = fig.add_gridspec(nrows, 3, hspace=0.35, wspace=0.3)
    
    # Color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    wce_color  = '#4CAF50'   # green  – WCE
    ufd_color  = '#9C27B0'   # purple – UFD
    beta_color = '#FF5722'   # deep-orange – beta
    
    # 1. Training and Validation Loss (combined / weighted)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], 'o-', linewidth=2, markersize=6, 
             color=colors[0], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 's-', linewidth=2, markersize=6, 
             color=colors[2], label='Val Loss')
    if best_epoch:
        ax1.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training & Validation Loss\n(Combined Adaptive Loss)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Dice Scores (excluding background)
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(1, num_classes):  # Skip class_0 (background)
        class_key = f'class_{i}'
        dice_scores = [m['dice'][class_key] for m in history['val_metrics']]
        ax2.plot(epochs, dice_scores, 'o-', linewidth=2, markersize=6, 
                label=class_names[class_key], color=colors[i % len(colors)])
    
    if best_epoch:
        ax2.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Dice Score', fontsize=11, fontweight='bold')
    ax2.set_title('Dice Scores by Class', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. Precision Scores (excluding background)
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(1, num_classes):
        class_key = f'class_{i}'
        precision_scores = [m['precision'][class_key] for m in history['val_metrics']]
        ax3.plot(epochs, precision_scores, 's-', linewidth=2, markersize=5, 
                label=class_names[class_key], color=colors[i % len(colors)])
    
    if best_epoch:
        ax3.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax3.set_title('Precision by Class', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Recall Scores (excluding background)
    ax4 = fig.add_subplot(gs[1, 0])
    for i in range(1, num_classes):
        class_key = f'class_{i}'
        recall_scores = [m['recall'][class_key] for m in history['val_metrics']]
        ax4.plot(epochs, recall_scores, '^-', linewidth=2, markersize=5, 
                label=class_names[class_key], color=colors[i % len(colors)])
    
    if best_epoch:
        ax4.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax4.set_title('Recall by Class', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Mean Metrics
    ax5 = fig.add_subplot(gs[1, 1])
    mean_dice = [m['dice']['mean'] for m in history['val_metrics']]
    mean_precision = [m['precision']['mean'] for m in history['val_metrics']]
    mean_recall = [m['recall']['mean'] for m in history['val_metrics']]
    
    ax5.plot(epochs, mean_dice, 'o-', linewidth=2, markersize=6, 
            color=colors[0], label='Mean Dice')
    ax5.plot(epochs, mean_precision, 's-', linewidth=2, markersize=5, 
            color=colors[1], label='Mean Precision')
    ax5.plot(epochs, mean_recall, '^-', linewidth=2, markersize=5, 
            color=colors[2], label='Mean Recall')
    
    if best_epoch:
        ax5.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax5.set_title('Mean Validation Metrics', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])

    # ── New Row 3 plots (only when val components are available) ──────────────
    if has_val_components:
        # 7. Training Loss Components (WCE vs UFD, train-side)
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(epochs, list(1*np.array(history['wce_loss'])), 'o-', linewidth=2, markersize=5,
                 color=wce_color, label='Train WCE Loss x10')
        ax7.plot(epochs, history['ufd_loss'], 's-', linewidth=2, markersize=5,
                 color=ufd_color, label='Train UFD Loss')
        ax7.plot(epochs, list(1*np.array(history['val_loss_wce'])), 'o--', linewidth=1.5, markersize=4,
                 color=wce_color, alpha=0.6, label='Val WCE Loss x10')
        ax7.plot(epochs, history['val_loss_ufd'], 's--', linewidth=1.5, markersize=4,
                 color=ufd_color, alpha=0.6, label='Val UFD Loss')
        if best_epoch:
            ax7.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2,
                        alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax7.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax7.set_title('Loss Components: WCE vs UFD\n(Train solid · Val dashed)', fontsize=13, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)

        # 8. Weighted contribution of each loss to the total loss
        ax8 = fig.add_subplot(gs[2, 1])
        beta_values = history.get('beta_value', [e / len(epochs) for e in epochs])
        betas = np.array(beta_values)
        ones  = np.ones_like(betas)

        # Weighted contributions
        train_wce_contrib = (ones - betas) * np.array(history['wce_loss'])
        train_ufd_contrib = betas            * np.array(history['ufd_loss'])
        val_wce_contrib   = (ones - betas) * np.array(history['val_loss_wce'])
        val_ufd_contrib   = betas            * np.array(history['val_loss_ufd'])

        ax8.stackplot(list(epochs),
                      train_wce_contrib, train_ufd_contrib,
                      labels=['(1−β)·WCE  [train] x10', 'β·UFD  [train]'],
                      colors=[wce_color, ufd_color], alpha=0.55)
        ax8.plot(epochs, history['train_loss'], 'k-', linewidth=1.5, label='Total Train Loss')

        # Overlay val contributions as lines for clarity
        ax8.plot(epochs, val_wce_contrib, '--', color=wce_color, linewidth=1.5,
                 alpha=0.8, label='(1−β)·WCE  [val] x10')
        ax8.plot(epochs, val_ufd_contrib, '--', color=ufd_color, linewidth=1.5,
                 alpha=0.8, label='β·UFD  [val]')
        if best_epoch:
            ax8.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax8.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Weighted Loss', fontsize=11, fontweight='bold')
        ax8.set_title('Weighted Loss Contributions\n(Adaptive β Schedule)', fontsize=13, fontweight='bold')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)

        # # 9. Beta schedule
        # ax9 = fig.add_subplot(gs[2, 2])
        # ax9.plot(list(epochs), betas, 'o-', linewidth=2, markersize=5,
        #          color=beta_color, label='β (epoch/total)')
        # ax9.fill_between(list(epochs), betas, alpha=0.15, color=beta_color)
        # ax9.axhline(y=0.95, color='gray', linestyle=':', linewidth=1.5,
        #             label='β = 0.95 threshold')
        # if best_epoch:
        #     ax9.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2,
        #                 alpha=0.7, label=f'Best Epoch ({best_epoch})')
        # ax9.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        # ax9.set_ylabel('β value', fontsize=11, fontweight='bold')
        # ax9.set_title('Beta Schedule\n(WCE → UFD transition)', fontsize=13, fontweight='bold')
        # ax9.set_ylim([0, 1.05])
        # ax9.legend(fontsize=9)
        # ax9.grid(True, alpha=0.3)

    # 6. Analysis Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    if analysis:
        abnormal_class = class_names[analysis['abnormal_key']]
        best_epoch_idx = analysis['best_overall_epoch'] - 1
        
        # Get dice scores for all classes at the best epoch
        best_epoch_metrics = history['val_metrics'][best_epoch_idx]['dice']
        
        # Build dice scores text (excluding background)
        dice_scores_text = ""
        for i in range(1, num_classes):
            class_key = f'class_{i}'
            dice_value = best_epoch_metrics[class_key]
            dice_scores_text += f"          {class_names[class_key]}: {dice_value:.4f}\n"
        
        summary_text = f"""
        TRAINING ANALYSIS SUMMARY
        {'=' * 40}
        
        Model: a U-Net
        Number of Classes: {analysis['num_classes']}
        Total Epochs: {len(epochs)}
        
        BEST OVERALL EPOCH: {analysis['best_overall_epoch']}
        (Composite Score: {analysis['composite_score']:.4f})
        
        Dice Scores at Best Epoch:
{dice_scores_text}
        {'─' * 40}
        Priority Metrics:
        {'─' * 40}
        
        Best {abnormal_class} Dice:
          Epoch {analysis['best_abnormal_epoch']}: {analysis['best_abnormal_dice']:.4f}
        
        Best Ventricles Dice:
          Epoch {analysis['best_ventricles_epoch']}: {analysis['best_ventricles_dice']:.4f}
        
        Best Validation Loss:
          Epoch {analysis['best_val_loss_epoch']}: {analysis['best_val_loss']:.4f}
        
        {'─' * 40}
        Loss at Best Epoch:
          Train WCE:  {analysis['best_epoch_metrics']['wce_loss']:.4f}
          Train UFD:  {analysis['best_epoch_metrics']['ufd_loss']:.4f}"""

        if analysis['best_epoch_metrics'].get('val_loss_wce') is not None:
            summary_text += f"""
          Val   WCE:  {analysis['best_epoch_metrics']['val_loss_wce']:.4f}
          Val   UFD:  {analysis['best_epoch_metrics']['val_loss_ufd']:.4f}"""

        summary_text += f"""
          β value:    {analysis['best_epoch_metrics']['beta_value']:.4f}
        
        {'─' * 40}
        Scoring Weights:
          {abnormal_class}: 60%
          Ventricles: 30%
          Val Loss: 10%
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('a U-Net Training History - Comprehensive Analysis\n'
                 '(Adaptive Loss: WCE + UFD with β schedule)', 
                 fontsize=16, fontweight='bold', y=0.998)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {save_path}")
    # plt.show()
    
    return analysis

def print_detailed_analysis(analysis):
    """Print detailed analysis to console."""
    if not analysis:
        print("No analysis available.")
        return
    
    print("\n" + "="*60)
    print("DETAILED TRAINING ANALYSIS - a U-NET")
    print("="*60)
    print(f"\n📊 Number of Classes: {analysis['num_classes']}")
    print(f"\n🏆 RECOMMENDED EPOCH: {analysis['best_overall_epoch']}")
    print(f"   Composite Score: {analysis['composite_score']:.4f}")
    print("\n" + "-"*60)
    print("Individual Best Performances:")
    print("-"*60)
    print(f"\n🎯 Abnormal WMH Dice (TOP PRIORITY):")
    print(f"   Best Epoch: {analysis['best_abnormal_epoch']}")
    print(f"   Best Score: {analysis['best_abnormal_dice']:.4f}")
    print(f"\n🫀 Ventricles Dice (SECONDARY):")
    print(f"   Best Epoch: {analysis['best_ventricles_epoch']}")
    print(f"   Best Score: {analysis['best_ventricles_dice']:.4f}")
    print(f"\n📉 Validation Loss (TERTIARY):")
    print(f"   Best Epoch: {analysis['best_val_loss_epoch']}")
    print(f"   Lowest Loss: {analysis['best_val_loss']:.4f}")
    print("\n" + "="*60)
    print("\nNote: Best overall epoch is calculated using weighted scoring:")
    print("  • Abnormal WMH Dice: 60%")
    print("  • Ventricles Dice: 30%")
    print("  • Validation Loss: 10%")
    print("="*60 + "\n")

def main_viz(filepath='history_sample.json', save_outputs=True):
    """Main execution function."""
    # Load history
    print(f"Loading training history from: {filepath}")
    history = load_history(filepath)
    
    print(f"✓ Loaded {len(history['train_loss'])} epochs of training data")
    
    # Get output directory
    out_dir = os.path.dirname(filepath)
    
    # Detect number of classes and get class names
    num_classes = detect_num_classes(history)
    class_names = get_class_names(num_classes)
    
    # Find best epoch and create analysis
    best_epoch, analysis = find_best_epoch(history, num_classes)
    
    # Create visualization
    plot_training_history(history, save_path=os.path.join(out_dir, 'a_unet_training_analysis.png'))
    
    # Print detailed analysis
    print_detailed_analysis(analysis)
    
    if save_outputs:
        print("\n" + "="*60)
        print("SAVING ANALYSIS OUTPUTS")
        print("="*60)
        
        # 1. Save standalone analysis JSON
        analysis_path = os.path.join(out_dir, 'best_epoch_analysis.json')
        save_analysis_json(analysis, analysis_path)
        
        # 2. Save enhanced history with analysis appended
        enhanced_history_path = os.path.join(out_dir, 'history_with_analysis.json')
        save_enhanced_history(history, analysis, enhanced_history_path)
        
        # 3. Save training summary
        summary = create_training_summary(history, analysis, class_names)
        summary_path = os.path.join(out_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Training summary saved to: {summary_path}")
        
        print("\n" + "="*60)
        print("ALL OUTPUTS SAVED SUCCESSFULLY")
        print("="*60)
        print("\nGenerated files:")
        print(f"  1. unet_training_analysis.png - Visualization")
        print(f"  2. best_epoch_analysis.json - Best epoch analysis")
        print(f"  3. history_with_analysis.json - Enhanced history")
        print(f"  4. training_summary.json - Comprehensive training summary")
        print("="*60 + "\n")
    
    return analysis, history

if __name__ == "__main__":

    # experiment_dir = '/mnt/e/MBashiri/ours_articles/Paper#2/Development/results_unet_baseline_fold_0/models'
    # scenario = 'standard_4class'
    # fold_num = 'fold_0'
    # filepath = os.path.join(experiment_dir, scenario, fold_num, 'history.json')
    
    # main_viz(filepath=filepath, save_outputs=True)
    
    for fold in range(5):

        # Skip folds:
        if fold in list(np.array([0, 2, 3, 4])):
            continue

        for variant in range(5):

            # # Skip variants:
            if variant not in list(np.array([1])):
                continue

            experiment_dir = f'/mnt/e/MBashiri/ours_articles/Paper#4/Development/results_fold_{fold}_var_{variant}_zscore2/models'
            scenario = 'standard_3class'
            fold_num = f'fold_{fold}'
            filepath = os.path.join(experiment_dir, scenario, fold_num, 'history.json')
            
            main_viz(filepath=filepath)

