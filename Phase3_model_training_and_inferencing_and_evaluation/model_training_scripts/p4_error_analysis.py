"""
P2 Article - Error Analysis & Hard Case Ranking Module
for Ventricles and WMH Segmentation

Integrates with p4_inference.py to identify problematic slices and patients,
rank them by difficulty, and produce rich diagnostic visualizations.

Developer: Mahdi Bashiri Bawil
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import binary_erosion, label as scipy_label
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Slice-level metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _dice_binary(gt_bin, pred_bin):
    """Dice for a single binary mask pair. Returns NaN if both are empty."""
    tp = np.sum(gt_bin & pred_bin)
    denom = np.sum(gt_bin) + np.sum(pred_bin)
    if denom == 0:
        return np.nan          # class truly absent — not a failure
    return float(2 * tp / (denom + 1e-7))


def _iou_binary(gt_bin, pred_bin):
    tp = np.sum(gt_bin & pred_bin)
    denom = np.sum(gt_bin | pred_bin)
    if denom == 0:
        return np.nan
    return float(tp / (denom + 1e-7))


def _precision_recall(gt_bin, pred_bin):
    tp = np.sum(gt_bin & pred_bin)
    fp = np.sum(~gt_bin & pred_bin)
    fn = np.sum(gt_bin & ~pred_bin)
    precision = float(tp / (tp + fp + 1e-7))
    recall    = float(tp / (tp + fn + 1e-7))
    return precision, recall


def _false_positive_volume(gt_bin, pred_bin):
    """Fraction of predicted pixels that are false positives."""
    fp = np.sum(~gt_bin & pred_bin)
    total_pred = np.sum(pred_bin)
    if total_pred == 0:
        return 0.0
    return float(fp / total_pred)


def _false_negative_volume(gt_bin, pred_bin):
    """Fraction of GT pixels that are missed."""
    fn = np.sum(gt_bin & ~pred_bin)
    total_gt = np.sum(gt_bin)
    if total_gt == 0:
        return 0.0
    return float(fn / total_gt)


def _gt_load(gt_hw, class_idx):
    """Return binary GT mask for a specific class from a (H,W) label map."""
    return gt_hw == class_idx


def _pred_load(pred_hw, class_idx):
    return pred_hw == class_idx


def compute_slice_metrics(gt_hw, pred_hw, num_classes, class_names,
                          mean_confidence=None):
    """
    Compute per-class and summary metrics for a single 2-D slice.

    Parameters
    ----------
    gt_hw : np.ndarray (H, W) — integer label map (ground truth)
    pred_hw : np.ndarray (H, W) — integer label map (prediction)
    num_classes : int
    class_names : list[str]
    mean_confidence : float | None — mean max-softmax probability for the slice

    Returns
    -------
    dict with per-class and aggregate metrics
    """
    results = {}
    dice_values   = []
    iou_values    = []

    for cls in range(num_classes):
        gt_bin   = _gt_load(gt_hw, cls)
        pred_bin = _pred_load(pred_hw, cls)

        dice = _dice_binary(gt_bin, pred_bin)
        iou  = _iou_binary(gt_bin, pred_bin)
        prec, rec = _precision_recall(gt_bin, pred_bin)
        fpr  = _false_positive_volume(gt_bin, pred_bin)
        fnr  = _false_negative_volume(gt_bin, pred_bin)

        gt_px   = int(np.sum(gt_bin))
        pred_px = int(np.sum(pred_bin))
        error_px = int(np.sum(gt_bin != pred_bin))

        results[class_names[cls]] = {
            'dice':           dice,
            'iou':            iou,
            'precision':      prec,
            'recall':         rec,
            'fp_rate':        fpr,
            'fn_rate':        fnr,
            'gt_pixels':      gt_px,
            'pred_pixels':    pred_px,
            'error_pixels':   error_px,
        }

        if not np.isnan(dice):
            dice_values.append(dice)
        if not np.isnan(iou):
            iou_values.append(iou)

    # Pixel-level error rate (ignoring class)
    total_px    = gt_hw.size
    wrong_px    = int(np.sum(gt_hw != pred_hw))
    error_rate  = wrong_px / total_px

    # Focus on foreground classes only (skip background=0) for composite score
    fg_dice = []
    for cls in range(1, num_classes):
        d = results[class_names[cls]]['dice']
        if not np.isnan(d):
            fg_dice.append(d)

    mean_fg_dice = float(np.mean(fg_dice)) if fg_dice else np.nan
    min_fg_dice  = float(np.min(fg_dice))  if fg_dice else np.nan

    results['_summary'] = {
        'error_rate':      error_rate,
        'wrong_pixels':    wrong_px,
        'total_pixels':    total_px,
        'mean_fg_dice':    mean_fg_dice,
        'min_fg_dice':     min_fg_dice,
        'mean_confidence': mean_confidence,
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Build slice-level and patient-level tables
# ─────────────────────────────────────────────────────────────────────────────

def build_error_tables(patient_results, num_classes, class_names):
    """
    Iterate over all patients / slices stored in patient_results
    (the dict returned by run_inference) and build:

      - slice_records : list of dicts, one per 2-D slice
      - patient_records : list of dicts, one per patient (aggregated)

    Parameters
    ----------
    patient_results : dict
        {patient_id: {'predictions', 'ground_truths', 'probabilities',
                      'flairs', 'slice_indices'}}
    num_classes : int
    class_names : list[str]

    Returns
    -------
    slice_df : pd.DataFrame
    patient_df : pd.DataFrame
    """
    slice_records   = []
    patient_records = []

    for patient_id, data in tqdm(patient_results.items(),
                                 desc="Building error tables"):
        order = np.argsort(data['slice_indices'])

        preds   = np.array(data['predictions'])[order]      # (S, H, W)
        gts     = np.array(data['ground_truths'])[order]    # (S, H, W, C) or (S, H, W)
        probs   = np.array(data['probabilities'])[order]    # (S, H, W)
        slices  = np.array(data['slice_indices'])[order]    # (S,)

        # Ground truth may be one-hot: collapse to label map
        if gts.ndim == 4:
            gts = np.argmax(gts, axis=-1)

        patient_fg_dice   = defaultdict(list)
        patient_error_rates = []

        for i, slice_num in enumerate(slices):
            gt_hw   = gts[i]
            pred_hw = preds[i]
            prob_hw = probs[i]

            mean_conf = float(np.mean(prob_hw))
            m = compute_slice_metrics(gt_hw, pred_hw, num_classes,
                                      class_names, mean_confidence=mean_conf)

            row = {
                'patient_id':   patient_id,
                'slice_num':    int(slice_num),
                'slice_id':     f"{patient_id}_slice_{int(slice_num):03d}",
                'error_rate':   m['_summary']['error_rate'],
                'wrong_pixels': m['_summary']['wrong_pixels'],
                'mean_fg_dice': m['_summary']['mean_fg_dice'],
                'min_fg_dice':  m['_summary']['min_fg_dice'],
                'mean_confidence': m['_summary']['mean_confidence'],
            }

            for cls in range(num_classes):
                cname = class_names[cls]
                cm    = m[cname]
                prefix = cname.lower().replace(' ', '_')
                row[f'{prefix}_dice']      = cm['dice']
                row[f'{prefix}_iou']       = cm['iou']
                row[f'{prefix}_precision'] = cm['precision']
                row[f'{prefix}_recall']    = cm['recall']
                row[f'{prefix}_fp_rate']   = cm['fp_rate']
                row[f'{prefix}_fn_rate']   = cm['fn_rate']
                row[f'{prefix}_gt_px']     = cm['gt_pixels']
                row[f'{prefix}_pred_px']   = cm['pred_pixels']
                row[f'{prefix}_err_px']    = cm['error_pixels']

                if cls > 0 and not np.isnan(cm['dice']):
                    patient_fg_dice[cname].append(cm['dice'])

            patient_error_rates.append(m['_summary']['error_rate'])
            slice_records.append(row)

        # ── Patient summary ──
        pat_row = {'patient_id': patient_id,
                   'n_slices':   len(slices),
                   'mean_error_rate': float(np.mean(patient_error_rates))}
        for cls in range(1, num_classes):
            cname  = class_names[cls]
            vals   = patient_fg_dice[cname]
            prefix = cname.lower().replace(' ', '_')
            pat_row[f'{prefix}_mean_dice'] = float(np.mean(vals)) if vals else np.nan
            pat_row[f'{prefix}_std_dice']  = float(np.std(vals))  if vals else np.nan
            pat_row[f'{prefix}_min_dice']  = float(np.min(vals))  if vals else np.nan

        # Composite: mean of per-class mean dices (foreground only)
        fg_means = [pat_row[f"{class_names[c].lower().replace(' ', '_')}_mean_dice"]
                    for c in range(1, num_classes)
                    if not np.isnan(pat_row.get(
                        f"{class_names[c].lower().replace(' ','_')}_mean_dice", np.nan))]
        pat_row['composite_dice'] = float(np.mean(fg_means)) if fg_means else np.nan

        patient_records.append(pat_row)

    slice_df   = pd.DataFrame(slice_records)
    patient_df = pd.DataFrame(patient_records)

    return slice_df, patient_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Composite difficulty score & ranking
# ─────────────────────────────────────────────────────────────────────────────

def rank_slices(slice_df, class_names, num_classes,
                fg_dice_weight=0.6, error_rate_weight=0.2,
                confidence_weight=0.2):
    """
    Add a `difficulty_score` column to slice_df (higher = harder).

    Score = fg_dice_weight * (1 - mean_fg_dice)
          + error_rate_weight * error_rate
          + confidence_weight * (1 - mean_confidence)

    NaN dice (class absent in GT) is neutral (0.5) so it doesn't
    inflate difficulty for slices where the class just doesn't exist.
    """
    df = slice_df.copy()

    # Fill NaN mean_fg_dice with 0.5 for scoring (class not present → neutral)
    fg_dice_filled = df['mean_fg_dice'].fillna(0.5)
    conf_filled    = df['mean_confidence'].fillna(0.5)

    df['difficulty_score'] = (
        fg_dice_weight    * (1 - fg_dice_filled) +
        error_rate_weight * df['error_rate'] +
        confidence_weight * (1 - conf_filled)
    )

    df = df.sort_values('difficulty_score', ascending=False).reset_index(drop=True)
    df['difficulty_rank'] = df.index + 1

    return df


def rank_patients(patient_df):
    """Sort patients from hardest to easiest (lowest composite dice first)."""
    df = patient_df.copy()
    df = df.sort_values('composite_dice', ascending=True).reset_index(drop=True)
    df['difficulty_rank'] = df.index + 1
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLORS_3 = ['black', '#2196F3', '#F44336']           # BG, Vent, WMH
CLASS_COLORS_4 = ['black', '#2196F3', '#4CAF50', '#F44336'] # BG, Vent, NormWMH, AbWMH

ERROR_CMAP = ListedColormap(['#1A1A1A',   # correct background
                              '#FF5722',  # FP (pred fg, gt bg)
                              '#03A9F4',  # FN (gt fg, pred bg)
                              '#FFEB3B']) # class confusion


def _get_class_cmap(num_classes):
    colors = CLASS_COLORS_3 if num_classes == 3 else CLASS_COLORS_4
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(range(num_classes + 1), num_classes)
    return cmap, norm


def _build_error_rgb(gt_hw, pred_hw, num_classes):
    """
    Build a pixel-wise error classification map:
      0 = correct
      1 = false positive (model predicts fg, GT is bg)
      2 = false negative (GT is fg, model predicts bg)
      3 = class confusion (both fg but wrong class)
    """
    gt_fg   = gt_hw > 0
    pred_fg = pred_hw > 0

    err = np.zeros_like(gt_hw, dtype=np.uint8)
    err[~gt_fg & pred_fg]              = 1   # FP
    err[gt_fg  & ~pred_fg]             = 2   # FN
    err[gt_fg  & pred_fg & (gt_hw != pred_hw)] = 3  # confusion
    return err


def _add_class_legend(ax, class_names, num_classes):
    colors = CLASS_COLORS_3 if num_classes == 3 else CLASS_COLORS_4
    patches = [mpatches.Patch(color=colors[i], label=class_names[i])
               for i in range(num_classes)]
    ax.legend(handles=patches, loc='lower right', fontsize=7,
              framealpha=0.8, markerscale=0.8)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Diagnostic slice visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_hard_slice(flair, gt_hw, pred_hw, prob_hw,
                         slice_metrics_row, class_names, num_classes,
                         save_path, rank=None):
    """
    Create a rich 3-row diagnostic panel for a single hard slice.

    Row 1 : FLAIR | GT mask | Predicted mask | Overlay (GT contour on FLAIR)
    Row 2 : Confidence map | Error type map | GT vs Pred contour overlay
    Row 3 : Per-class dice bar chart | FP/FN summary table
    """
    cmap_cls, norm_cls = _get_class_cmap(num_classes)
    err_map = _build_error_rgb(gt_hw, pred_hw, num_classes)

    patient_id  = slice_metrics_row.get('patient_id', '?')
    slice_num   = slice_metrics_row.get('slice_num', '?')
    diff_score  = slice_metrics_row.get('difficulty_score', float('nan'))
    diff_rank   = slice_metrics_row.get('difficulty_rank', rank)
    mean_conf   = slice_metrics_row.get('mean_confidence', float('nan'))
    mean_fg_d   = slice_metrics_row.get('mean_fg_dice', float('nan'))

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0D0D0D')
    title_str = (f"Patient: {patient_id}  |  Slice: {slice_num:03d}  |  "
                 f"Rank #{diff_rank}  |  Difficulty: {diff_score:.3f}  |  "
                 f"Mean FG Dice: {mean_fg_d:.3f}  |  Mean Conf: {mean_conf:.3f}")
    fig.suptitle(title_str, color='white', fontsize=12, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.35, wspace=0.25,
                           left=0.04, right=0.98,
                           top=0.93, bottom=0.04)

    def styled_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor('#0D0D0D')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        return ax

    # ── Row 0 ──────────────────────────────────────────────────────────────
    ax00 = styled_ax(gs[0, 0])
    ax00.imshow(flair, cmap='gray', vmin=flair.min(), vmax=flair.max())
    ax00.set_title('FLAIR', color='white', fontsize=10)
    ax00.axis('off')

    ax01 = styled_ax(gs[0, 1])
    ax01.imshow(gt_hw, cmap=cmap_cls, norm=norm_cls, interpolation='nearest')
    ax01.set_title('Ground Truth', color='white', fontsize=10)
    ax01.axis('off')
    _add_class_legend(ax01, class_names, num_classes)

    ax02 = styled_ax(gs[0, 2])
    ax02.imshow(pred_hw, cmap=cmap_cls, norm=norm_cls, interpolation='nearest')
    ax02.set_title('Prediction', color='white', fontsize=10)
    ax02.axis('off')
    _add_class_legend(ax02, class_names, num_classes)

    # GT contour overlay on FLAIR
    ax03 = styled_ax(gs[0, 3])
    ax03.imshow(flair, cmap='gray', vmin=flair.min(), vmax=flair.max())
    colors_cls = CLASS_COLORS_3 if num_classes == 3 else CLASS_COLORS_4
    for cls in range(1, num_classes):
        gt_bin   = (gt_hw == cls).astype(np.uint8)
        pred_bin = (pred_hw == cls).astype(np.uint8)
        if gt_bin.any():
            ax03.contour(gt_bin,   levels=[0.5], colors=[colors_cls[cls]],
                         linewidths=1.5, linestyles='solid')
        if pred_bin.any():
            ax03.contour(pred_bin, levels=[0.5], colors=[colors_cls[cls]],
                         linewidths=1.2, linestyles='dashed')
    gt_patch   = mpatches.Patch(color='white',  linestyle='solid',  label='GT (solid)')
    pred_patch = mpatches.Patch(color='white',  linestyle='dashed', label='Pred (dashed)')
    ax03.legend(handles=[gt_patch, pred_patch], loc='lower right',
                fontsize=7, framealpha=0.7)
    ax03.set_title('GT vs Pred Contours', color='white', fontsize=10)
    ax03.axis('off')

    # ── Row 1 ──────────────────────────────────────────────────────────────
    ax10 = styled_ax(gs[1, 0])
    im_conf = ax10.imshow(prob_hw, cmap='plasma', vmin=0, vmax=1)
    plt.colorbar(im_conf, ax=ax10, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')
    ax10.set_title('Confidence Map', color='white', fontsize=10)
    ax10.axis('off')

    # Low-confidence overlay on FLAIR
    ax11 = styled_ax(gs[1, 1])
    ax11.imshow(flair, cmap='gray')
    low_conf_mask = prob_hw < 0.5
    overlay = np.zeros((*flair.shape, 4))
    overlay[low_conf_mask] = [1, 0.3, 0, 0.55]   # orange-red for uncertain regions
    ax11.imshow(overlay)
    ax11.set_title('Low-Confidence Regions (<0.5)', color='white', fontsize=10)
    ax11.axis('off')

    ax12 = styled_ax(gs[1, 2])
    err_colors = ['#1A1A1A', '#FF5722', '#03A9F4', '#FFEB3B']
    err_cmap   = ListedColormap(err_colors)
    err_norm   = BoundaryNorm([0, 1, 2, 3, 4], 4)
    ax12.imshow(err_map, cmap=err_cmap, norm=err_norm, interpolation='nearest')
    patches_err = [
        mpatches.Patch(color='#1A1A1A', label='Correct'),
        mpatches.Patch(color='#FF5722', label='False Positive'),
        mpatches.Patch(color='#03A9F4', label='False Negative'),
        mpatches.Patch(color='#FFEB3B', label='Class Confusion'),
    ]
    ax12.legend(handles=patches_err, loc='lower right', fontsize=6.5, framealpha=0.8)
    ax12.set_title('Error Type Map', color='white', fontsize=10)
    ax12.axis('off')

    # FLAIR + error overlay
    ax13 = styled_ax(gs[1, 3])
    flair_rgb = np.stack([flair] * 3, axis=-1)
    # Normalise 0-1
    flair_rgb = (flair_rgb - flair_rgb.min()) / (flair_rgb.max() - flair_rgb.min() + 1e-7)
    err_overlay = flair_rgb.copy()
    err_overlay[err_map == 1] = [1.0, 0.34, 0.13]  # FP
    err_overlay[err_map == 2] = [0.01, 0.66, 0.96] # FN
    err_overlay[err_map == 3] = [1.0, 0.92, 0.23]  # confusion
    ax13.imshow(err_overlay)
    ax13.set_title('FLAIR + Error Overlay', color='white', fontsize=10)
    ax13.axis('off')

    # ── Row 2: metrics ─────────────────────────────────────────────────────
    ax20 = styled_ax(gs[2, 0:2])
    ax20.set_facecolor('#111')

    bar_labels  = []
    bar_dice    = []
    bar_colors  = []
    for cls in range(1, num_classes):
        cname  = class_names[cls]
        prefix = cname.lower().replace(' ', '_')
        d = slice_metrics_row.get(f'{prefix}_dice', np.nan)
        bar_labels.append(cname)
        bar_dice.append(d if not np.isnan(d) else 0)
        bar_colors.append(colors_cls[cls])

    x = np.arange(len(bar_labels))
    bars = ax20.bar(x, bar_dice, color=bar_colors, edgecolor='white',
                    linewidth=0.8, width=0.5)
    ax20.axhline(0.5, color='red',    linestyle='--', linewidth=1, label='Threshold 0.5')
    ax20.axhline(0.8, color='yellow', linestyle='--', linewidth=1, label='Good 0.8')
    ax20.set_xticks(x)
    ax20.set_xticklabels(bar_labels, color='white', fontsize=9)
    ax20.set_ylim(0, 1.05)
    ax20.set_ylabel('Dice Score', color='white', fontsize=9)
    ax20.set_title('Per-Class Dice', color='white', fontsize=10)
    ax20.tick_params(axis='y', colors='white')
    ax20.legend(fontsize=7, labelcolor='white', framealpha=0.3)
    for bar, val in zip(bars, bar_dice):
        ax20.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f'{val:.3f}', ha='center', color='white', fontsize=9)

    # Table: per-class FP/FN/precision/recall
    ax21 = styled_ax(gs[2, 2:4])
    ax21.axis('off')

    col_labels = ['Class', 'Dice', 'Prec', 'Recall', 'FP rate', 'FN rate',
                  'GT px', 'Pred px']
    table_data = []
    for cls in range(1, num_classes):
        cname  = class_names[cls]
        prefix = cname.lower().replace(' ', '_')
        def _g(k):
            v = slice_metrics_row.get(f'{prefix}_{k}', np.nan)
            return f'{v:.3f}' if not np.isnan(v) else 'N/A'
        table_data.append([
            cname,
            _g('dice'), _g('precision'), _g('recall'),
            _g('fp_rate'), _g('fn_rate'),
            str(int(slice_metrics_row.get(f'{prefix}_gt_px', 0))),
            str(int(slice_metrics_row.get(f'{prefix}_pred_px', 0))),
        ])

    tbl = ax21.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#444')
        if r == 0:
            cell.set_facecolor('#2C2C2C')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#1A1A1A')
            cell.set_text_props(color='white')
    ax21.set_title('Per-Class Metrics Summary', color='white', fontsize=10, pad=8)

    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Patient-level summary visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_patient_summary(patient_id, patient_data, slice_df_patient,
                               class_names, num_classes, save_path):
    """
    One-page summary for a single patient showing:
      - Dice scores across all slices (line plot per class)
      - Confidence vs. error rate scatter
      - Per-slice FP / FN bar chart
      - Overall dice distribution box plots
    """
    order    = np.argsort(patient_data['slice_indices'])
    slices   = np.array(patient_data['slice_indices'])[order]
    n_slices = len(slices)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.patch.set_facecolor('#0D0D0D')
    fig.suptitle(f'Patient Summary  |  ID: {patient_id}  |  {n_slices} slices',
                 color='white', fontsize=13, fontweight='bold')

    colors_cls = CLASS_COLORS_3 if num_classes == 3 else CLASS_COLORS_4

    df = slice_df_patient.sort_values('slice_num').reset_index(drop=True)

    # ── Plot 1: Per-slice Dice per class ──────────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor('#111')
    for cls in range(1, num_classes):
        cname  = class_names[cls]
        prefix = cname.lower().replace(' ', '_')
        col    = f'{prefix}_dice'
        if col in df.columns:
            valid = df[col].notna()
            ax.plot(df.loc[valid, 'slice_num'], df.loc[valid, col],
                    color=colors_cls[cls], linewidth=1.5,
                    marker='o', markersize=3, label=cname)
    ax.axhline(0.5, color='red',    linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(0.8, color='yellow', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Slice Number', color='white')
    ax.set_ylabel('Dice Score', color='white')
    ax.set_title('Per-Slice Dice by Class', color='white', fontsize=10)
    ax.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_ylim(0, 1.05)

    # ── Plot 2: Confidence vs Error rate scatter ───────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor('#111')
    sc = ax.scatter(df['mean_confidence'], df['error_rate'],
                    c=df['mean_fg_dice'].fillna(0.5),
                    cmap='RdYlGn', vmin=0, vmax=1,
                    s=50, edgecolors='white', linewidths=0.3, alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Mean FG Dice', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    ax.set_xlabel('Mean Confidence', color='white')
    ax.set_ylabel('Pixel Error Rate', color='white')
    ax.set_title('Confidence vs Error Rate\n(colour = Mean FG Dice)',
                 color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    # Annotate worst 3 slices
    worst3 = df.nlargest(3, 'difficulty_score') if 'difficulty_score' in df.columns \
             else df.nlargest(3, 'error_rate')
    for _, row in worst3.iterrows():
        ax.annotate(f"sl{int(row['slice_num']):03d}",
                    (row['mean_confidence'], row['error_rate']),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color='white')

    # ── Plot 3: FP / FN pixel rates per slice ─────────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor('#111')
    x = df['slice_num'].values
    # Use WMH class (last foreground class) as primary interest
    cls_main   = num_classes - 1
    prefix_m   = class_names[cls_main].lower().replace(' ', '_')
    fp_col     = f'{prefix_m}_fp_rate'
    fn_col     = f'{prefix_m}_fn_rate'

    if fp_col in df.columns and fn_col in df.columns:
        width = 0.4
        ax.bar(x - width/2, df[fp_col].fillna(0), width=width,
               color='#FF5722', alpha=0.8, label='FP Rate')
        ax.bar(x + width/2, df[fn_col].fillna(0), width=width,
               color='#03A9F4', alpha=0.8, label='FN Rate')
    ax.set_xlabel('Slice Number', color='white')
    ax.set_ylabel('Rate', color='white')
    ax.set_title(f'FP / FN Rate per Slice  [{class_names[cls_main]}]',
                 color='white', fontsize=10)
    ax.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    # ── Plot 4: Dice distribution box plots ───────────────────────────────
    ax = axes[1, 1]
    ax.set_facecolor('#111')
    box_data   = []
    box_labels = []
    box_colors = []
    for cls in range(1, num_classes):
        cname  = class_names[cls]
        prefix = cname.lower().replace(' ', '_')
        col    = f'{prefix}_dice'
        vals   = df[col].dropna().values if col in df.columns else np.array([])
        box_data.append(vals)
        box_labels.append(cname)
        box_colors.append(colors_cls[cls])

    bp = ax.boxplot(box_data, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'fliers']:
        for item in bp[element]:
            item.set_color('white')

    ax.set_xticklabels(box_labels, color='white')
    ax.set_ylabel('Dice Score', color='white')
    ax.set_title('Dice Score Distribution per Class', color='white', fontsize=10)
    ax.axhline(0.5, color='red',    linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(0.8, color='yellow', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Dataset-level overview visualizations
# ─────────────────────────────────────────────────────────────────────────────

def visualize_dataset_overview(slice_df, patient_df, class_names,
                                num_classes, save_dir):
    """
    Global overview plots saved to save_dir/overview/:
      1. Dice distribution across all slices (violin per class)
      2. Patient ranking bar chart (composite dice)
      3. Error rate histogram
      4. Confidence vs dice scatter (all slices)
      5. Difficulty score distribution
    """
    overview_dir = Path(save_dir) / 'overview'
    overview_dir.mkdir(parents=True, exist_ok=True)

    colors_cls = CLASS_COLORS_3 if num_classes == 3 else CLASS_COLORS_4

    # ── 1. Dice violin ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0D0D0D')
    ax.set_facecolor('#111')

    violin_data   = []
    violin_labels = []
    for cls in range(1, num_classes):
        cname  = class_names[cls]
        prefix = cname.lower().replace(' ', '_')
        col    = f'{prefix}_dice'
        vals   = slice_df[col].dropna().values if col in slice_df.columns else np.array([])
        violin_data.append(vals)
        violin_labels.append(cname)

    parts = ax.violinplot(violin_data, showmedians=True, showextrema=True)
    for i, (pc, color) in enumerate(zip(parts['bodies'],
                                        [colors_cls[c] for c in range(1, num_classes)])):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts['cmedians'].set_colors('white')
    parts['cmaxes'].set_colors('#aaa')
    parts['cmins'].set_colors('#aaa')
    parts['cbars'].set_colors('#aaa')

    ax.set_xticks(range(1, len(violin_labels) + 1))
    ax.set_xticklabels(violin_labels, color='white')
    ax.axhline(0.5, color='red',    linestyle='--', linewidth=0.9, label='0.5 threshold')
    ax.axhline(0.8, color='yellow', linestyle='--', linewidth=0.9, label='0.8 target')
    ax.set_ylabel('Dice Score', color='white')
    ax.set_title('Dice Distribution — All Slices', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(overview_dir / 'dice_violin_all_slices.png', dpi=130,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── 2. Patient ranking bar chart ──────────────────────────────────────
    pat_sorted = patient_df.sort_values('composite_dice').reset_index(drop=True)
    n_patients = len(pat_sorted)

    fig, ax = plt.subplots(figsize=(max(12, n_patients * 0.6), 5))
    fig.patch.set_facecolor('#0D0D0D')
    ax.set_facecolor('#111')

    bar_colors = ['#F44336' if v < 0.5 else '#FFC107' if v < 0.7 else '#4CAF50'
                  for v in pat_sorted['composite_dice'].fillna(0)]
    ax.bar(range(n_patients), pat_sorted['composite_dice'].fillna(0),
           color=bar_colors, edgecolor='#333', linewidth=0.5)
    ax.set_xticks(range(n_patients))
    ax.set_xticklabels(pat_sorted['patient_id'], rotation=75,
                       ha='right', color='white', fontsize=7)
    ax.axhline(0.5, color='red',    linestyle='--', linewidth=0.9)
    ax.axhline(0.7, color='orange', linestyle='--', linewidth=0.9)
    ax.axhline(0.8, color='yellow', linestyle='--', linewidth=0.9)
    ax.set_ylabel('Composite Dice (mean FG classes)', color='white')
    ax.set_title('Patient Ranking — Composite Dice (worst → best)',
                 color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_ylim(0, 1.05)

    red_p    = mpatches.Patch(color='#F44336', label='< 0.5 (critical)')
    orange_p = mpatches.Patch(color='#FFC107', label='0.5–0.7 (poor)')
    green_p  = mpatches.Patch(color='#4CAF50', label='≥ 0.7 (acceptable)')
    ax.legend(handles=[red_p, orange_p, green_p],
              fontsize=8, labelcolor='white', framealpha=0.3)

    plt.tight_layout()
    plt.savefig(overview_dir / 'patient_ranking.png', dpi=130,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── 3. Error rate histogram ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0D0D0D')
    ax.set_facecolor('#111')
    ax.hist(slice_df['error_rate'].dropna(), bins=40, color='#9C27B0',
            edgecolor='white', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('Pixel Error Rate per Slice', color='white')
    ax.set_ylabel('Count', color='white')
    ax.set_title('Pixel Error Rate Distribution — All Slices', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    plt.tight_layout()
    plt.savefig(overview_dir / 'error_rate_histogram.png', dpi=130,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── 4. Confidence vs mean FG Dice scatter ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#0D0D0D')
    ax.set_facecolor('#111')
    sc = ax.scatter(slice_df['mean_confidence'], slice_df['mean_fg_dice'].fillna(0),
                    c=slice_df['error_rate'], cmap='RdYlGn_r',
                    vmin=0, vmax=0.3, s=10, alpha=0.6)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Pixel Error Rate', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    ax.set_xlabel('Mean Softmax Confidence', color='white')
    ax.set_ylabel('Mean FG Dice', color='white')
    ax.set_title('Confidence vs FG Dice — All Slices', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    plt.tight_layout()
    plt.savefig(overview_dir / 'confidence_vs_dice_scatter.png', dpi=130,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── 5. Difficulty score distribution ──────────────────────────────────
    if 'difficulty_score' in slice_df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#0D0D0D')
        ax.set_facecolor('#111')
        ax.hist(slice_df['difficulty_score'].dropna(), bins=40,
                color='#FF9800', edgecolor='white', linewidth=0.3, alpha=0.85)
        ax.set_xlabel('Difficulty Score', color='white')
        ax.set_ylabel('Count', color='white')
        ax.set_title('Difficulty Score Distribution — All Slices', color='white', fontsize=12)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        plt.tight_layout()
        plt.savefig(overview_dir / 'difficulty_score_histogram.png', dpi=130,
                    bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"  ✅ Overview plots saved to: {overview_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Main entry point: run_error_analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_error_analysis(results, config,
                       top_n_slices=30,
                       top_n_patients=10,
                       fg_dice_weight=0.6,
                       error_rate_weight=0.2,
                       confidence_weight=0.2):
    """
    Full pipeline: build tables → rank → save CSVs → generate visualizations.

    Call after run_inference():
        results = run_inference(config)
        run_error_analysis(results, config)

    Parameters
    ----------
    results : dict  — returned by run_inference()
    config  : InferenceConfig
    top_n_slices : int  — how many hardest slices to visualize individually
    top_n_patients : int — how many hardest patients to get summary plots
    fg_dice_weight, error_rate_weight, confidence_weight : floats for ranking
    """
    patient_results = results['patients_results']
    class_names     = config.class_names
    num_classes     = config.num_classes

    # Output sub-directories
    error_dir = config.inference_dir / 'error_analysis'
    hard_slices_dir  = error_dir / 'hard_slices'
    patient_summaries_dir = error_dir / 'patient_summaries'
    tables_dir = error_dir / 'tables'

    for d in [hard_slices_dir, patient_summaries_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS — Building slice & patient tables")
    print("=" * 70)

    # ── Step 1: build tables ──────────────────────────────────────────────
    slice_df, patient_df = build_error_tables(patient_results, num_classes, class_names)

    # ── Step 2: rank ──────────────────────────────────────────────────────
    slice_df   = rank_slices(slice_df, class_names, num_classes,
                             fg_dice_weight, error_rate_weight, confidence_weight)
    patient_df = rank_patients(patient_df)

    # ── Step 3: save CSVs ─────────────────────────────────────────────────
    slice_csv   = tables_dir / 'slice_difficulty_ranking.csv'
    patient_csv = tables_dir / 'patient_difficulty_ranking.csv'
    slice_df.to_csv(slice_csv,   index=False)
    patient_df.to_csv(patient_csv, index=False)
    print(f"  ✅ Slice table  → {slice_csv}")
    print(f"  ✅ Patient table → {patient_csv}")

    # ── Step 4: dataset overview plots ────────────────────────────────────
    print("\nGenerating dataset overview plots...")
    visualize_dataset_overview(slice_df, patient_df, class_names,
                               num_classes, error_dir)

    # ── Step 5: hard slice visualizations ────────────────────────────────
    print(f"\nVisualizing top-{top_n_slices} hardest slices...")
    hard_slices = slice_df.head(top_n_slices)

    for _, row in tqdm(hard_slices.iterrows(),
                       total=len(hard_slices), desc="Hard slice panels"):
        patient_id = row['patient_id']
        slice_num  = int(row['slice_num'])

        data  = patient_results[patient_id]
        order = np.argsort(data['slice_indices'])
        slices_sorted = np.array(data['slice_indices'])[order]

        # Find position of this slice
        pos = np.where(slices_sorted == slice_num)[0]
        if len(pos) == 0:
            continue
        pos = pos[0]

        gts   = np.array(data['ground_truths'])[order]
        preds = np.array(data['predictions'])[order]
        probs = np.array(data['probabilities'])[order]
        flairs = np.array(data['flairs'])[order]

        gt_hw    = gts[pos]
        pred_hw  = preds[pos]
        prob_hw  = probs[pos]
        flair_hw = flairs[pos]

        # Collapse one-hot GT if needed
        if gt_hw.ndim == 3:
            gt_hw = np.argmax(gt_hw, axis=-1)

        rank = int(row['difficulty_rank'])
        fname = (f"rank{rank:04d}_"
                 f"{patient_id}_slice{slice_num:03d}"
                 f"_dice{row['mean_fg_dice']:.3f}.png")
        save_path = hard_slices_dir / fname

        visualize_hard_slice(
            flair=flair_hw,
            gt_hw=gt_hw,
            pred_hw=pred_hw,
            prob_hw=prob_hw,
            slice_metrics_row=row.to_dict(),
            class_names=class_names,
            num_classes=num_classes,
            save_path=save_path,
            rank=rank
        )

    print(f"  ✅ Hard slice panels → {hard_slices_dir}")

    # ── Step 6: patient summary visualizations ────────────────────────────
    print(f"\nGenerating top-{top_n_patients} hardest patient summaries...")
    hard_patients = patient_df.head(top_n_patients)

    for _, pat_row in tqdm(hard_patients.iterrows(),
                            total=len(hard_patients), desc="Patient summaries"):
        patient_id = pat_row['patient_id']
        if patient_id not in patient_results:
            continue

        data = patient_results[patient_id]
        slice_df_patient = slice_df[slice_df['patient_id'] == patient_id].copy()

        rank = int(pat_row['difficulty_rank'])
        comp = pat_row.get('composite_dice', float('nan'))
        fname = (f"rank{rank:03d}_{patient_id}"
                 f"_composite{comp:.3f}.png")
        save_path = patient_summaries_dir / fname

        visualize_patient_summary(
            patient_id=patient_id,
            patient_data=data,
            slice_df_patient=slice_df_patient,
            class_names=class_names,
            num_classes=num_classes,
            save_path=save_path
        )

    print(f"  ✅ Patient summaries → {patient_summaries_dir}")

    # ── Step 7: print console summary ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nTotal slices analysed : {len(slice_df)}")
    print(f"Total patients         : {len(patient_df)}")

    print(f"\nTop-10 Hardest Slices:")
    top10_cols = ['difficulty_rank', 'slice_id', 'mean_fg_dice',
                  'error_rate', 'mean_confidence', 'difficulty_score']
    top10_cols = [c for c in top10_cols if c in slice_df.columns]
    print(slice_df[top10_cols].head(10).to_string(index=False))

    print(f"\nTop-10 Hardest Patients:")
    fg_dice_cols = [f"{class_names[c].lower().replace(' ', '_')}_mean_dice"
                    for c in range(1, num_classes)]
    pat_cols = ['difficulty_rank', 'patient_id', 'n_slices', 'composite_dice'] + \
               [c for c in fg_dice_cols if c in patient_df.columns]
    print(patient_df[pat_cols].head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"All error analysis outputs → {error_dir}")
    print("=" * 70 + "\n")

    return {
        'slice_df':   slice_df,
        'patient_df': patient_df,
        'error_dir':  error_dir
    }
