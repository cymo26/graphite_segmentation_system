import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# KONFIGURACJA ŚCIEŻEK I PARAMETRÓW

# ścieżki plików
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / ".." / ".." / "data" / "raw"
MASKS_DIR = SCRIPT_DIR / ".." / ".." / "data" / "processed" / "masks"
OUTPUT_DIR = SCRIPT_DIR / ".." / ".." / "data" / "processed" / "split"

# podział danych
TRAIN_RATIO = 0.70  # 70% trening
VAL_RATIO = 0.15    # 15% walidacja
TEST_RATIO = 0.15   # 15% testy

# Seed
RANDOM_STATE = 42

# Rozszerzenia
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# FUNKCJE POMOCNICZE

def find_all_images(raw_dir: Path) -> list:
    
    images = []
    for ext in IMAGE_EXTENSIONS:
        # szuka plików z danym rozszerzeniem
        images.extend(raw_dir.rglob(f"*{ext}"))
        images.extend(raw_dir.rglob(f"*{ext.upper()}"))
    
    # pomijamy pliki  :Zone.Identifier
    images = [img for img in images if ':Zone.Identifier' not in str(img)]
    
    return sorted(images)


def get_mask_path(image_path: Path, masks_dir: Path, raw_dir: Path) -> Path:
    
    # ścieżka obrazu 
    relative_path = image_path.relative_to(raw_dir)
    mask_name = image_path.stem + "_mask" + image_path.suffix
    
    # ścieżka do maski
    mask_path = masks_dir / relative_path.parent / mask_name
    
    return mask_path


def validate_pairs(images: list, masks_dir: Path, raw_dir: Path) -> list:

    valid_pairs = []
    missing_masks = []
    
    for image_path in images:
        mask_path = get_mask_path(image_path, masks_dir, raw_dir)
        
        if mask_path.exists():
            valid_pairs.append((image_path, mask_path))
        else:
            missing_masks.append(image_path)
    
    # brakujące maski
    if missing_masks:
        print(f"\n  Znaleziono {len(missing_masks)} obrazów bez odpowiadających masek:")
        for img in missing_masks[:5]:  # max 5
            print(f"   - {img.name}")
        if len(missing_masks) > 5:
            print(f"   ... i {len(missing_masks) - 5} więcej")
        print(f"   Te obrazy zostaną pominięte w podziale.\n")
    
    return valid_pairs


def create_output_structure(output_dir: Path) -> dict:
    paths = {}
    
    for split in ['train', 'val', 'test']:
        for data_type in ['images', 'masks']:
            path = output_dir / split / data_type
            path.mkdir(parents=True, exist_ok=True)
            paths[f"{split}_{data_type}"] = path
    
    return paths


def copy_files(pairs: list, dest_images: Path, dest_masks: Path, desc: str):
    for image_path, mask_path in tqdm(pairs, desc=desc, unit="plików"):
        # kopiujemy obraz
        shutil.copy2(image_path, dest_images / image_path.name)
        # kopiujemy maskę
        shutil.copy2(mask_path, dest_masks / mask_path.name)

def main():
    print("PODZIAŁ ZBIORU DANYCH NA TRAIN/VAL/TEST")
    
    # Rozwiązujemy ścieżki absolutne
    raw_dir = RAW_DIR.resolve()
    masks_dir = MASKS_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    
    print(f"\n Katalog z obrazami: {raw_dir}")
    print(f" Katalog z maskami:  {masks_dir}")
    print(f" Katalog wyjściowy:  {output_dir}")
    
    # Znajdź wszystkie obrazy
    print("\n Szukanie obrazów...")
    all_images = find_all_images(raw_dir)
    print(f"   Znaleziono {len(all_images)} obrazów")
    
    # Walidacja par obraz-maska
    print("\n Walidacja par obraz-maska...")
    valid_pairs = validate_pairs(all_images, masks_dir, raw_dir)
    print(f"   Znaleziono {len(valid_pairs)} poprawnych par")
    
    if len(valid_pairs) == 0:
        print("\n Brak danych do podziału! Sprawdź ścieżki i nazwy plików.")
        return
    
    # Podział danych
    print(f"\n Podział danych (seed={RANDOM_STATE}):")
    print(f"   - Train: {TRAIN_RATIO*100:.0f}%")
    print(f"   - Val:   {VAL_RATIO*100:.0f}%")
    print(f"   - Test:  {TEST_RATIO*100:.0f}%")
    
    # Pierwszy podział: train vs (val + test)
    train_pairs, temp_pairs = train_test_split(
        valid_pairs,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE
    )
    
    # Drugi podział: val vs test
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        train_size=VAL_RATIO / (VAL_RATIO + TEST_RATIO),  # 0.15 / 0.30 = 0.5
        random_state=RANDOM_STATE
    )
    
    print(f"\n Wyniki podziału:")
    print(f"   - Train: {len(train_pairs)} obrazów ({len(train_pairs)/len(valid_pairs)*100:.1f}%)")
    print(f"   - Val:   {len(val_pairs)} obrazów ({len(val_pairs)/len(valid_pairs)*100:.1f}%)")
    print(f"   - Test:  {len(test_pairs)} obrazów ({len(test_pairs)/len(valid_pairs)*100:.1f}%)")
    
    # Tworzenie struktury katalogów
    print("\n Tworzenie struktury katalogów...")
    paths = create_output_structure(output_dir)
    
    # Kopiowanie plików
    print("\n Kopiowanie plików...")
    
    copy_files(train_pairs, paths['train_images'], paths['train_masks'], "Train")
    copy_files(val_pairs, paths['val_images'], paths['val_masks'], "Val  ")
    copy_files(test_pairs, paths['test_images'], paths['test_masks'], "Test ")
    
    # Podsumowanie

    print(" PODZIAŁ ZAKOŃCZONY")
    print(f"\nStruktura katalogów:")
    print(f"   {output_dir}/")
    print(f"   ├── train/")
    print(f"   │   ├── images/ ({len(train_pairs)} plików)")
    print(f"   │   └── masks/  ({len(train_pairs)} plików)")
    print(f"   ├── val/")
    print(f"   │   ├── images/ ({len(val_pairs)} plików)")
    print(f"   │   └── masks/  ({len(val_pairs)} plików)")
    print(f"   └── test/")
    print(f"       ├── images/ ({len(test_pairs)} plików)")
    print(f"       └── masks/  ({len(test_pairs)} plików)")


if __name__ == "__main__":
    main()
