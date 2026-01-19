import numpy as np
import cv2
from skimage import measure, morphology
from skimage.measure import regionprops, regionprops_table
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


SIZE_CLASSES = {
    1: (1000, float('inf')),   # > 1000 
    2: (500, 1000),            # 500-1000 
    3: (250, 500),             # 250-500 
    4: (120, 250),             # 120-250 
    5: (60, 120),              # 60-120 
    6: (30, 60),               # 30-60 
    7: (15, 30),               # 15-30 
    8: (0, 15),                # < 15 
}

SIZE_LABELS = {
    1: '>1000 µm',
    2: '500-1000 µm',
    3: '250-500 µm',
    4: '120-250 µm',
    5: '60-120 µm',
    6: '30-60 µm',
    7: '15-30 µm',
    8: '<15 µm',
}

GRAPHITE_FORMS = {
    'I': 'Płatkowy (lamellar)',
    'II': 'Rozetkowy (rosette)',
    'III': 'Wermikularny (vermicular/compacted)',
    'IV': 'Kłaczkowy (undercooled/flake)',
    'V': 'Sferoidalny nieregularny (irregular nodular)',
    'VI': 'Sferoidalny regularny (regular nodular)',
}

FORM_LABELS = {
    'I': 'Płatkowy',
    'II': 'Rozetkowy',
    'III': 'Wermikularny',
    'IV': 'Kłaczkowy',
    'V': 'Sferoid. niereg.',
    'VI': 'Sferoid. reg.',
}

SIZE_COLORS = {
    1: '#FF0001',  # czerwony
    2: '#FF6600',  # pomarańczowy
    3: '#FFCC00',  # żółty
    4: '#99FF00',  # jasny zielony
    5: '#00FF00',  # zielony
    6: '#00FFCC',  # jasny niebieski
    7: '#0066FF',  # niebieski
    8: '#9900FF',  # fioletowy
}

FORM_COLORS = {
    'I': '#E41A1C',    # czerwony
    'II': '#377EB8',   # niebieski
    'III': '#4DAF4A',  # zielony
    'IV': '#984EA3',   # fioletowy
    'V': '#FF7F00',    # pomarańczowy
    'VI': '#FFFF33',   # żółty
}



@dataclass
class ParticleFeatures:
    label: int                  # ID cząstki
    area_px: float              # Pole w pikselach
    area_um2: float             # Pole w um^2
    perimeter_px: float         # Obwód w pikselach
    perimeter_um: float         # Obwód w um
    d_max_px: float             # Maksymalna średnica Fereta (px)
    d_max_um: float             # Maksymalna średnica Fereta (um)
    d_min_px: float             # Minimalna średnica Fereta (px)
    d_min_um: float             # Minimalna średnica Fereta (um)
    aspect_ratio: float         # d_min / d_max
    roundness: float            # A / (PI * (d_max/2)^2)
    solidity: float             # Area / ConvexHullArea
    convexity: float            # ConvexHullPerimeter / Perimeter
    centroid: Tuple[int, int]   # Środek cząstki (row, col)
    size_class: int             # Klasa wielkości (1-8)
    form: str                   # Forma grafitu (I-VI)


def extract_particles(mask: np.ndarray, min_area_px: int = 10) -> Tuple[np.ndarray, List[dict]]:
   
    # Normalizacja maski do binarnej
    if mask.max() > 1:
        binary = (mask > 127).astype(np.uint8)
    else:
        binary = (mask > 0.5).astype(np.uint8)
    
    labeled_mask = measure.label(binary, connectivity=2)
    
    regions = regionprops(labeled_mask)
    
    valid_labels = [r.label for r in regions if r.area >= min_area_px]
    
    filtered_mask = np.zeros_like(labeled_mask)
    for i, label in enumerate(valid_labels, start=1):
        filtered_mask[labeled_mask == label] = i
    
    regions = regionprops(filtered_mask)
    
    return filtered_mask, regions



def compute_feret_diameters(region) -> Tuple[float, float]:
  
    coords = region.coords
    if len(coords) < 3:
        return 0, 0
    
    try:
        hull = cv2.convexHull(coords[:, ::-1].astype(np.float32))
        if hull is None or len(hull) < 3:
            return region.major_axis_length, region.minor_axis_length
        
        # średnica Fereta
        rect = cv2.minAreaRect(hull)
        width, height = rect[1]
        d_max = max(width, height)
        d_min = min(width, height)
        
        if d_max == 0:
            d_max = region.major_axis_length
            d_min = region.minor_axis_length
            
    except Exception:
        d_max = region.major_axis_length
        d_min = region.minor_axis_length
    
    return d_max, d_min


def compute_convexity(region) -> float:
   
    if region.perimeter == 0:
        return 0
    
    # convex hull
    convex_perimeter = region.convex_image.sum() 
    
    try:
        contours, _ = cv2.findContours(
            region.convex_image.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            convex_perimeter = cv2.arcLength(contours[0], True)
    except Exception:
        pass
    
    convexity = convex_perimeter / region.perimeter if region.perimeter > 0 else 0
    
    return min(convexity, 1.0)


def compute_features(regions: List, scale_um_per_px: float, magnification_factor: float = 5.0) -> List[ParticleFeatures]:
    
    effective_scale = scale_um_per_px * magnification_factor
    
    features_list = []
    
    for region in regions:
        d_max_px, d_min_px = compute_feret_diameters(region)
        
        # Konwersja do um
        area_um2 = region.area * (effective_scale ** 2)
        perimeter_um = region.perimeter * effective_scale
        d_max_um = d_max_px * effective_scale
        d_min_um = d_min_px * effective_scale
        
        # Aspect ratio (d_min / d_max, zakres 0-1)
        aspect_ratio = d_min_px / d_max_px if d_max_px > 0 else 0
        
        # Roundness = A / (PI * (d_max/2)^2)
        ideal_area = np.pi * (d_max_px / 2) ** 2
        roundness = region.area / ideal_area if ideal_area > 0 else 0
        
        # Solidity = Area / ConvexHullArea
        solidity = region.solidity
        
        # Convexity
        convexity = compute_convexity(region)
        
        # Klasyfikacja
        size_class = classify_size(d_max_um)
        form = classify_shape(roundness, aspect_ratio, convexity, solidity)
        
        features = ParticleFeatures(
            label=region.label,
            area_px=region.area,
            area_um2=area_um2,
            perimeter_px=region.perimeter,
            perimeter_um=perimeter_um,
            d_max_px=d_max_px,
            d_max_um=d_max_um,
            d_min_px=d_min_px,
            d_min_um=d_min_um,
            aspect_ratio=aspect_ratio,
            roundness=roundness,
            solidity=solidity,
            convexity=convexity,
            centroid=(int(region.centroid[0]), int(region.centroid[1])),
            size_class=size_class,
            form=form
        )
        features_list.append(features)
    
    return features_list



def classify_size(d_max_um: float) -> int:
    for class_num, (min_val, max_val) in SIZE_CLASSES.items():
        if min_val <= d_max_um < max_val:
            return class_num
    return 8 



def classify_shape(roundness: float, aspect_ratio: float, 
                   convexity: float, solidity: float) -> str:
    
    # Forma VI - Sferoidalny regularny 
    if roundness > 0.75 and solidity > 0.9:
        return 'VI'
    
    # Forma V - Sferoidalny nieregularny
    if roundness > 0.6 and solidity > 0.85:
        return 'V'
    
    # Forma I - Płatkowy
    if aspect_ratio < 0.25 and roundness < 0.3:
        return 'I'
    
    # Forma IV - Kłaczkowy
    if aspect_ratio > 0.7 and roundness < 0.4 and solidity < 0.7:
        return 'IV'
    
    # Forma III - Wermikularny
    if 0.25 <= aspect_ratio <= 0.6 and convexity > 0.85:
        return 'III'
    
    # Forma II - Rozetkowy
    if solidity < 0.6 and roundness < 0.5:
        return 'II'
    
    # Domyślnie - forma III
    return 'III'



def summarize_results(features_list: List[ParticleFeatures]) -> Dict:
   
    if not features_list:
        return {
            'total_particles': 0,
            'total_area_um2': 0,
            'form_distribution': {},
            'size_distribution': {},
            'dominant_form': None,
            'dominant_size_class': None,
            'dataframe': pd.DataFrame()
        }
    
    data = []
    for f in features_list:
        data.append({
            'ID': f.label,
            'Area (µm²)': round(f.area_um2, 2),
            'Perimeter (µm)': round(f.perimeter_um, 2),
            'd_max (µm)': round(f.d_max_um, 2),
            'd_min (µm)': round(f.d_min_um, 2),
            'Aspect Ratio': round(f.aspect_ratio, 3),
            'Roundness': round(f.roundness, 3),
            'Solidity': round(f.solidity, 3),
            'Convexity': round(f.convexity, 3),
            'Size Class': f.size_class,
            'Size Range': SIZE_LABELS.get(f.size_class, '-'),
            'Form': f.form,
            'Form Name': FORM_LABELS.get(f.form, '-')
        })
    
    df = pd.DataFrame(data)
    
    form_areas = {}
    for f in features_list:
        form_areas[f.form] = form_areas.get(f.form, 0) + f.area_um2
    
    total_area = sum(form_areas.values())
    form_distribution = {
        form: (area / total_area * 100) if total_area > 0 else 0 
        for form, area in form_areas.items()
    }
    
    size_counts = {}
    for f in features_list:
        size_counts[f.size_class] = size_counts.get(f.size_class, 0) + 1
    
    total_particles = len(features_list)
    size_distribution = {
        size: (count / total_particles * 100) 
        for size, count in size_counts.items()
    }
    
    dominant_form = max(form_distribution, key=form_distribution.get) if form_distribution else None
    
    dominant_size_class = max(size_counts, key=size_counts.get) if size_counts else None
    
    return {
        'total_particles': total_particles,
        'total_area_um2': total_area,
        'form_distribution': form_distribution,
        'size_distribution': size_distribution,
        'dominant_form': dominant_form,
        'dominant_size_class': dominant_size_class,
        'dataframe': df
    }



def create_colored_mask(labeled_mask: np.ndarray, 
                        features_list: List[ParticleFeatures],
                        color_by: str = 'size') -> np.ndarray:
    h, w = labeled_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    label_to_features = {f.label: f for f in features_list}
    
    for label, features in label_to_features.items():
        mask = labeled_mask == label
        
        if color_by == 'size':
            hex_color = SIZE_COLORS.get(features.size_class, '#FFFFFF')
        else:
            hex_color = FORM_COLORS.get(features.form, '#FFFFFF')
        
        # Konwersja hex do RGB
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        colored[mask] = rgb
    
    return colored


def create_analysis_figure(original_image: np.ndarray,
                          colored_mask_size: np.ndarray,
                          colored_mask_form: np.ndarray,
                          summary: Dict) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    

    
    #Rząd 1
    ax1 = fig.add_subplot(2, 3, 1)
    if original_image is not None:
        ax1.imshow(original_image)
    ax1.set_title('Obraz oryginalny', fontsize=11)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(colored_mask_size)
    ax2.set_title('Klasyfikacja wg wielkości', fontsize=11)
    ax2.axis('off')
    
    # Legenda wielkości
    size_patches = [Patch(color=SIZE_COLORS[i], label=SIZE_LABELS[i]) for i in range(1, 9)]
    ax2.legend(handles=size_patches, loc='upper right', fontsize=6, ncol=2)
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(colored_mask_form)
    ax3.set_title('Klasyfikacja wg formy', fontsize=11)
    ax3.axis('off')
    
    # Legenda form
    form_patches = [Patch(color=FORM_COLORS[f], label=FORM_LABELS[f]) for f in ['I', 'II', 'III', 'IV', 'V', 'VI']]
    ax3.legend(handles=form_patches, loc='upper right', fontsize=6)
    
    # Rząd 2
    # Wykres kołowy
    ax4 = fig.add_subplot(2, 3, 4)
    form_dist = summary.get('form_distribution', {})
    if form_dist:
        forms = list(form_dist.keys())
        values = list(form_dist.values())
        colors = [FORM_COLORS.get(f, '#CCCCCC') for f in forms]
        form_names = [FORM_LABELS.get(f, f) for f in forms]
        wedges, texts, autotexts = ax4.pie(
            values, labels=form_names, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 8}
        )
        ax4.set_title('Udział form grafitu (%)', fontsize=11)
    else:
        ax4.text(0.5, 0.5, 'Brak danych', ha='center', va='center')
        ax4.set_title('Udział form grafitu', fontsize=11)
    
    # Wykres słupkowy wielkości
    ax5 = fig.add_subplot(2, 3, 5)
    size_dist = summary.get('size_distribution', {})
    if size_dist:
        classes = sorted(size_dist.keys())
        values = [size_dist[c] for c in classes]
        colors = [SIZE_COLORS[c] for c in classes]
        labels = [SIZE_LABELS[c] for c in classes]
        bars = ax5.bar(labels, values, color=colors, edgecolor='black')
        ax5.set_xlabel('Przedział wielkości')
        ax5.set_ylabel('Udział (%)')
        ax5.set_title('Rozkład wielkości cząstek', fontsize=11)
        ax5.set_ylim(0, max(values) * 1.2 if values else 100)
        ax5.tick_params(axis='x', rotation=45, labelsize=7)
    else:
        ax5.text(0.5, 0.5, 'Brak danych', ha='center', va='center')
    
    # Tekstowe podsumowanie
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    dominant_form = summary.get('dominant_form', '-')
    dominant_size = summary.get('dominant_size_class', '-')
    total_particles = summary.get('total_particles', 0)
    total_area = summary.get('total_area_um2', 0)
    
    form_pct = form_dist.get(dominant_form, 0) if dominant_form else 0
    form_name = FORM_LABELS.get(dominant_form, '-') if dominant_form else '-'
    
    size_label = SIZE_LABELS.get(dominant_size, '-') if dominant_size else '-'
    
    summary_text = f"""
PODSUMOWANIE ANALIZY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Liczba cząstek grafitu: {total_particles}
Całkowite pole grafitu: {total_area:.1f} µm²

DOMINUJĄCA FORMA: {form_name} ({form_pct:.1f}%)

DOMINUJĄCY PRZEDZIAŁ WIELKOŚCI:
{size_label}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analiza wg PN-EN ISO 945-1
"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig


def analyze_graphite(mask: np.ndarray, 
                     scale_um_per_px: float,
                     original_image: np.ndarray = None,
                     min_area_px: int = 10,
                     magnification: int = 500) -> Dict:
    
    magnification_factor = magnification / 100
    
    labeled_mask, regions = extract_particles(mask, min_area_px)
    
    if len(regions) == 0:
        return {
            'summary': summarize_results([]),
            'features': [],
            'labeled_mask': labeled_mask,
            'colored_mask_size': np.zeros((*mask.shape, 3), dtype=np.uint8),
            'colored_mask_form': np.zeros((*mask.shape, 3), dtype=np.uint8),
            'figure': None
        }
    
    features_list = compute_features(regions, scale_um_per_px, magnification_factor)
    
    
    summary = summarize_results(features_list)
    
    colored_mask_size = create_colored_mask(labeled_mask, features_list, 'size')
    colored_mask_form = create_colored_mask(labeled_mask, features_list, 'form')
    

    figure = create_analysis_figure(
        original_image, 
        colored_mask_size, 
        colored_mask_form, 
        summary
    )
    
    return {
        'summary': summary,
        'features': features_list,
        'labeled_mask': labeled_mask,
        'colored_mask_size': colored_mask_size,
        'colored_mask_form': colored_mask_form,
        'figure': figure
    }