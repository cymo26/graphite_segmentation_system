"""
Skrypt do wykonywania tilingu obrazow i masek z obsluga overlap/halo.
Dzieli obrazy i maski na mniejsze kafelki (tiles) dla sieci U-Net i DeepLabV3.

Uzycie:
    python tiling.py
    python tiling.py --tile_size 256 --overlap 16 dla U-Net
    python tiling.py --tile_size 512 --overlap 32 dla DeepLabV3

"""

import argparse
from pathlib import Path
from typing import Tuple, List
import shutil
import cv2
import numpy as np
from tqdm import tqdm


# KONFIGURACJA

SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = SCRIPT_DIR / ".." / ".." / "data" / "processed" / "split"

# parametry tilingu
DEFAULT_TILE_SIZE = 256  # rozmiar dla U-Net
DEFAULT_OVERLAP = 32     # overlap/halo w pikselach

# rozmiar oryginalnych obrazow
ORIGINAL_WIDTH = 2560
ORIGINAL_HEIGHT = 1920


# FUNKCJE TILINGU

def calculate_tile_positions(
    image_size: Tuple[int, int],
    tile_size: int,
    overlap: int
) -> List[Tuple[int, int]]:
    """
    Oblicza pozycje (x, y) lewego gornego rogu kazdego kafelka.
    Kafelki sa rozmieszczone tak, aby pokryc caly obraz z zadanym overlap.
    """
    width, height = image_size
    stride = tile_size - overlap
    
    positions = []
    
    y = 0
    while y + tile_size <= height:
        x = 0
        while x + tile_size <= width:
            positions.append((x, y))
            x += stride
        # ostatni kafelek w rzedzie (jesli nie pokrywa prawej krawedzi)
        if positions and positions[-1][0] + tile_size < width:
            positions.append((width - tile_size, y))
        y += stride
    
    # ostatni rzad (jesli nie pokrywa dolnej krawedzi)
    if positions and positions[-1][1] + tile_size < height:
        y = height - tile_size
        x = 0
        while x + tile_size <= width:
            positions.append((x, y))
            x += stride
        if positions[-1][0] + tile_size < width:
            positions.append((width - tile_size, y))
    
    # usun duplikaty zachowujac kolejnosc
    seen = set()
    unique_positions = []
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            unique_positions.append(pos)
    
    return unique_positions


def extract_tile(
    image: np.ndarray,
    position: Tuple[int, int],
    tile_size: int
) -> np.ndarray:
    """Wycina kafelek z obrazu."""
    x, y = position
    return image[y:y + tile_size, x:x + tile_size].copy()


def process_single_image(
    image_path: Path,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    is_mask: bool = False
) -> int:
    """
    Przetwarza pojedynczy obraz - dzieli go na kafelki.
    Zwraca liczbe utworzonych kafelkow.
    """
    if is_mask:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"  Nie mozna wczytac: {image_path}")
        return 0
    
    height, width = image.shape[:2]
    positions = calculate_tile_positions((width, height), tile_size, overlap)
    
    # folder dla kafelkow tego obrazu
    image_name = image_path.stem
    tiles_folder = output_dir / image_name
    tiles_folder.mkdir(parents=True, exist_ok=True)
    
    # informacje o tilingu (przydatne przy rekonstrukcji)
    info_file = tiles_folder / "tiling_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"original_width={width}\n")
        f.write(f"original_height={height}\n")
        f.write(f"tile_size={tile_size}\n")
        f.write(f"overlap={overlap}\n")
        f.write(f"num_tiles={len(positions)}\n")
    
    # wytnij i zapisz kafelki
    for idx, (x, y) in enumerate(positions):
        tile = extract_tile(image, (x, y), tile_size)
        tile_name = f"tile_{idx:03d}_x{x}_y{y}.png"
        tile_path = tiles_folder / tile_name
        cv2.imwrite(str(tile_path), tile)
    
    return len(positions)


def process_split(
    split_name: str,
    split_dir: Path,
    tile_size: int,
    overlap: int,
    clean: bool = False
) -> dict:
    """Przetwarza caly zbior (train/val/test)."""
    stats = {'images': 0, 'masks': 0, 'image_tiles': 0, 'mask_tiles': 0}
    
    # obrazy i maski sa w podfolderze raw/
    images_dir = split_dir / split_name / "images" / "raw"
    masks_dir = split_dir / split_name / "masks" / "raw"
    
    # kafelki trafiaja do images/tiled i masks/tiled
    tiled_images_dir = split_dir / split_name / "images" / "tiled"
    tiled_masks_dir = split_dir / split_name / "masks" / "tiled"
    
    # wyczysc foldery tiled jesli --clean
    if clean:
        if tiled_images_dir.exists():
            shutil.rmtree(tiled_images_dir)
        if tiled_masks_dir.exists():
            shutil.rmtree(tiled_masks_dir)
    
    # przetworz obrazy
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"{split_name}/images", unit="img"):
            num_tiles = process_single_image(
                img_path, tiled_images_dir, tile_size, overlap, is_mask=False
            )
            stats['images'] += 1
            stats['image_tiles'] += num_tiles
    
    # przetworz maski
    if masks_dir.exists():
        mask_files = list(masks_dir.glob("*.jpg")) + list(masks_dir.glob("*.png"))
        
        for mask_path in tqdm(mask_files, desc=f"{split_name}/masks ", unit="mask"):
            num_tiles = process_single_image(
                mask_path, tiled_masks_dir, tile_size, overlap, is_mask=True
            )
            stats['masks'] += 1
            stats['mask_tiles'] += num_tiles
    
    return stats


# FUNKCJA REKONSTRUKCJI (do uzycia przy predykcji)

def reconstruct_from_tiles(
    tiles_folder: Path,
    tile_size: int,
    overlap: int,
    original_size: Tuple[int, int],
    blend: bool = True
) -> np.ndarray:
    """
    Rekonstruuje obraz z kafelkow.
    Uzywa usredniania w obszarach nakladajacych sie (blending).
    """
    width, height = original_size
    
    reconstructed = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    
    tile_files = sorted(tiles_folder.glob("tile_*.png"))
    
    for tile_path in tile_files:
        name_parts = tile_path.stem.split('_')
        x = int(name_parts[2][1:])
        y = int(name_parts[3][1:])
        
        tile = cv2.imread(str(tile_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        if blend:
            weight = create_blend_weight(tile_size, overlap)
            reconstructed[y:y + tile_size, x:x + tile_size] += tile * weight
            counts[y:y + tile_size, x:x + tile_size] += weight
        else:
            reconstructed[y:y + tile_size, x:x + tile_size] += tile
            counts[y:y + tile_size, x:x + tile_size] += 1
    
    reconstructed = np.divide(reconstructed, counts, where=counts > 0)
    
    return reconstructed.astype(np.uint8)


def create_blend_weight(tile_size: int, overlap: int) -> np.ndarray:
    """Tworzy maske wagowa dla blendingu."""
    weight_1d = np.ones(tile_size)
    
    if overlap > 0:
        weight_1d[:overlap] = np.linspace(0, 1, overlap)
        weight_1d[-overlap:] = np.linspace(1, 0, overlap)
    
    weight_2d = np.outer(weight_1d, weight_1d)
    
    return weight_2d


# GLOWNA FUNKCJA

def main():
    parser = argparse.ArgumentParser(
        description="Tiling obrazow i masek dla U-Net/DeepLabV3"
    )
    parser.add_argument(
        "--tile_size", type=int, default=DEFAULT_TILE_SIZE,
        help=f"Rozmiar kafelka (domyslnie: {DEFAULT_TILE_SIZE})"
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_OVERLAP,
        help=f"Overlap/halo w pikselach (domyslnie: {DEFAULT_OVERLAP})"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Zbiory do przetworzenia (domyslnie: train val test)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Usun istniejace kafelki przed nowym tilingiem"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TILING OBRAZOW I MASEK")
    print("=" * 60)
    
    split_dir = SPLIT_DIR.resolve()
    
    print(f"\nKatalog split: {split_dir}")
    print(f"\nParametry tilingu:")
    print(f"   - Rozmiar kafelka: {args.tile_size}x{args.tile_size}")
    print(f"   - Overlap (halo):  {args.overlap} px")
    print(f"   - Stride:          {args.tile_size - args.overlap} px")
    
    positions = calculate_tile_positions(
        (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
        args.tile_size,
        args.overlap
    )
    print(f"\nDla obrazu {ORIGINAL_WIDTH}x{ORIGINAL_HEIGHT}:")
    print(f"   - Liczba kafelkow: {len(positions)}")
    
    total_stats = {'images': 0, 'masks': 0, 'image_tiles': 0, 'mask_tiles': 0}
    
    print("\nPrzetwarzanie zbiorow...\n")
    
    for split_name in args.splits:
        print(f"\n{split_name.upper()}")
        stats = process_split(split_name, split_dir, args.tile_size, args.overlap, args.clean)
        
        for key in total_stats:
            total_stats[key] += stats[key]
        
        print(f"   Obrazy: {stats['images']} -> {stats['image_tiles']} kafelkow")
        print(f"   Maski:  {stats['masks']} -> {stats['mask_tiles']} kafelkow")
    
    print("\n" + "=" * 60)
    print("TILING ZAKONCZONY")
    print("=" * 60)
    print(f"\nPodsumowanie:")
    print(f"   - Przetworzono obrazow: {total_stats['images']}")
    print(f"   - Przetworzono masek:   {total_stats['masks']}")
    print(f"   - Utworzono kafelkow obrazow: {total_stats['image_tiles']}")
    print(f"   - Utworzono kafelkow masek:   {total_stats['mask_tiles']}")
    print(f"\nStruktura wyjsciowa:")
    print(f"   split/train/images/tiled/<nazwa_obrazu>/tile_XXX.png")
    print(f"   split/train/masks/tiled/<nazwa_maski>/tile_XXX.png")
    
    print(f"\nWskazowki:")
    print(f"   - Dla U-Net:      --tile_size 256 --overlap 32")
    print(f"   - Dla DeepLabV3:  --tile_size 512 --overlap 64")
    print(f"   - Flaga --clean usuwa istniejace kafelki przed nowym tilingiem.")


if __name__ == "__main__":
    main()
