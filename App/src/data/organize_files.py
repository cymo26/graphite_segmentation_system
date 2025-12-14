import shutil
from pathlib import Path
from tqdm import tqdm


# KONFIGURACJA ŚCIEŻEK

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / ".." / ".." / "data" / "raw"
MASKS_DIR = SCRIPT_DIR / ".." / ".." / "data" / "processed" / "masks"
SCHEMAS_DIR = SCRIPT_DIR / ".." / ".." / "data" / "processed" / "schemas"

# Numery folderów do przetworzenia
FOLDER_NUMBERS = range(1, 12)  # 1 do 11


def ensure_directory(path: Path) -> None:
   
    path.mkdir(parents=True, exist_ok=True)


def move_file(src: Path, dst: Path) -> bool:
    
    if src.exists():
        shutil.move(str(src), str(dst))
        return True
    return False


def organize_folder(folder_num: int, raw_dir: Path, masks_dir: Path, schemas_dir: Path) -> dict:
    
    stats = {'masks': 0, 'schemas': 0, 'labels': 0}
    
    # Ścieżka źródłowa: raw/N/N/
    src_folder = raw_dir / str(folder_num) / str(folder_num)
    
    # Ścieżki docelowe
    mask_dst = masks_dir / str(folder_num) / str(folder_num)
    schema_dst = schemas_dir / str(folder_num) / str(folder_num)
    
    # Sprawdź czy folder źródłowy istnieje
    if not src_folder.exists():
        return stats
    
    # Utwórz katalogi docelowe
    ensure_directory(mask_dst)
    ensure_directory(schema_dst)
    
    # Przeglądaj wszystkie pliki w folderze
    for file_path in src_folder.iterdir():
        if not file_path.is_file():
            continue
            
        filename = file_path.name
        
        # Pomiń pliki Zone.Identifier
        if ':Zone.Identifier' in filename:
            continue
        
        # Maski: *_mask.jpg
        if filename.endswith('_mask.jpg'):
            if move_file(file_path, mask_dst / filename):
                stats['masks'] += 1
        
        # Schema: schema.json
        elif filename == 'schema.json':
            if move_file(file_path, schema_dst / filename):
                stats['schemas'] += 1
        
        # Labels: *__labels.json
        elif filename.endswith('__labels.json'):
            if move_file(file_path, schema_dst / filename):
                stats['labels'] += 1
    
    return stats


# GŁÓWNA FUNKCJA

def main():
   
    print("MASKI I SCHEMATY")
    
    # Rozwiązujemy ścieżki absolutne
    raw_dir = RAW_DIR.resolve()
    masks_dir = MASKS_DIR.resolve()
    schemas_dir = SCHEMAS_DIR.resolve()
    
    print(f"\n Źródło (raw):           {raw_dir}")
    print(f" Docelowy (masks):       {masks_dir}")
    print(f" Docelowy (schemas):     {schemas_dir}")
    
    # Statystyki globalne
    total_stats = {'masks': 0, 'schemas': 0, 'labels': 0}
    
    print("\n Przenoszenie plików...\n")
    
    # Przetwarzaj każdy folder
    for folder_num in tqdm(FOLDER_NUMBERS, desc="Foldery", unit="folder"):
        stats = organize_folder(folder_num, raw_dir, masks_dir, schemas_dir)
        
        # Aktualizuj statystyki globalne
        for key in total_stats:
            total_stats[key] += stats[key]
        
        # Pokaż szczegóły dla niepustych folderów
        if any(stats.values()):
            tqdm.write(f"   Folder {folder_num:2d}: "
                      f"maski={stats['masks']:3d}, "
                      f"schema={stats['schemas']}, "
                      f"labels={stats['labels']:3d}")
    
    # Podsumowanie
    print(" ORGANIZACJA ZAKOŃCZONA!")
    print(f"\n Podsumowanie:")
    print(f"   - Przeniesiono masek:        {total_stats['masks']}")
    print(f"   - Przeniesiono schematów:    {total_stats['schemas']}")
    print(f"   - Przeniesiono labels:       {total_stats['labels']}")
    print(f"\n Struktura docelowa:")
    print(f"   data/processed/")
    print(f"   ├── masks/")
    print(f"   │   ├── 1/1/  ... 11/11/")
    print(f"   └── schemas/")
    print(f"       ├── 1/1/  ... 11/11/")


if __name__ == "__main__":
    main()
