import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Tuple, Dict



def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = (pred_mask > 127).astype(np.float32) if pred_mask.max() > 1 else pred_mask.astype(np.float32)
    gt = (gt_mask > 127).astype(np.float32) if gt_mask.max() > 1 else gt_mask.astype(np.float32)
    
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    total = tp + tn + fp + fn
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU and Dice
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0
    
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1-Score': f1,
        'IoU': iou,
        'Dice': dice
    }



COLORS = {
    'TP': [0, 255, 0],      # Zielony - TP
    'FP': [255, 0, 0],      # Czerwony - FP
    'FN': [0, 0, 255],      # Niebieski - FN
    'TN': [50, 50, 50],     # Ciemnoszary - TN
}


def create_comparison_mask(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    pred = (pred_mask > 127).astype(np.uint8) if pred_mask.max() > 1 else pred_mask.astype(np.uint8)
    gt = (gt_mask > 127).astype(np.uint8) if gt_mask.max() > 1 else gt_mask.astype(np.uint8)
    
    h, w = pred.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    # TP - predykcja=1, gt=1
    tp_mask = (pred == 1) & (gt == 1)
    result[tp_mask] = COLORS['TP']
    
    # FP - predykcja=1, gt=0
    fp_mask = (pred == 1) & (gt == 0)
    result[fp_mask] = COLORS['FP']
    
    # FN - predykcja=0, gt=1
    fn_mask = (pred == 0) & (gt == 1)
    result[fn_mask] = COLORS['FN']
    
    # TN - predykcja=0, gt=0
    tn_mask = (pred == 0) & (gt == 0)
    result[tn_mask] = COLORS['TN']
    
    return result


def create_overlay_image(original: np.ndarray, pred_mask: np.ndarray, 
                         gt_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    
    comparison = create_comparison_mask(pred_mask, gt_mask)
    
    orig_float = original.astype(np.float32)
    comp_float = comparison.astype(np.float32)
    
    mask_non_tn = ~((pred_mask <= 127) & (gt_mask <= 127))
    if pred_mask.max() <= 1:
        mask_non_tn = ~((pred_mask == 0) & (gt_mask == 0))
    
    result = orig_float.copy()
    result[mask_non_tn] = (1 - alpha) * orig_float[mask_non_tn] + alpha * comp_float[mask_non_tn]
    
    return result.astype(np.uint8)


def create_evaluation_figure(original: np.ndarray, 
                             pred_mask: np.ndarray, 
                             gt_mask: np.ndarray,
                             metrics: Dict[str, float]) -> plt.Figure:
   
    fig = plt.figure(figsize=(16, 10))
    
    # Rzad 1: Obrazy
    # Oryginal
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(original)
    ax1.set_title('Obraz oryginalny', fontsize=11)
    ax1.axis('off')
    
    # Maska GT
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gt_mask, cmap='gray')
    ax2.set_title('Ground Truth', fontsize=11)
    ax2.axis('off')
    
    # Maska predykcji
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(pred_mask, cmap='gray')
    ax3.set_title('Predykcja modelu', fontsize=11)
    ax3.axis('off')
    
    #Rzad 2
    # Mapa porownawcza
    ax4 = fig.add_subplot(2, 3, 4)
    comparison = create_comparison_mask(pred_mask, gt_mask)
    ax4.imshow(comparison)
    ax4.set_title('Mapa porownawcza', fontsize=11)
    ax4.axis('off')
    
    # Legenda
    legend_patches = [
        Patch(color=np.array(COLORS['TP'])/255, label='True Positive (TP)'),
        Patch(color=np.array(COLORS['FP'])/255, label='False Positive (FP)'),
        Patch(color=np.array(COLORS['FN'])/255, label='False Negative (FN)'),
        Patch(color=np.array(COLORS['TN'])/255, label='True Negative (TN)'),
    ]
    ax4.legend(handles=legend_patches, loc='upper right', fontsize=8)
    
    # Nakladka na oryginal
    ax5 = fig.add_subplot(2, 3, 5)
    overlay = create_overlay_image(original, pred_mask, gt_mask, alpha=0.6)
    ax5.imshow(overlay)
    ax5.set_title('Nakladka na obraz', fontsize=11)
    ax5.axis('off')
    
    # Metryki tekstowe
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    metrics_text = f"""
METRYKI EWALUACJI

CONFUSION MATRIX:
  True Positive (TP):  {metrics['TP']:,} px
  True Negative (TN):  {metrics['TN']:,} px
  False Positive (FP): {metrics['FP']:,} px
  False Negative (FN): {metrics['FN']:,} px


METRYKI:
  Accuracy:    {metrics['Accuracy']:.4f}  ({metrics['Accuracy']*100:.2f}%)
  Precision:   {metrics['Precision']:.4f}  ({metrics['Precision']*100:.2f}%)
  Recall:      {metrics['Recall']:.4f}  ({metrics['Recall']*100:.2f}%)
  Specificity: {metrics['Specificity']:.4f}  ({metrics['Specificity']*100:.2f}%)
  
  F1-Score:    {metrics['F1-Score']:.4f}
  IoU:         {metrics['IoU']:.4f}
  Dice:        {metrics['Dice']:.4f}

"""
    
    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    return fig



def evaluate_mask(pred_mask: np.ndarray, 
                  gt_mask: np.ndarray, 
                  original_image: np.ndarray = None) -> Dict:
    
    if pred_mask.shape[:2] != gt_mask.shape[:2]:
        raise ValueError(
            f"Wymiary masek sie nie zgadzaja: "
            f"pred={pred_mask.shape[:2]}, gt={gt_mask.shape[:2]}"
        )
    
    metrics = compute_metrics(pred_mask, gt_mask)
    
    comparison_mask = create_comparison_mask(pred_mask, gt_mask)
    
    result = {
        'metrics': metrics,
        'comparison_mask': comparison_mask,
        'figure': None
    }
    
    if original_image is not None:
        result['figure'] = create_evaluation_figure(
            original_image, pred_mask, gt_mask, metrics
        )
    
    return result
