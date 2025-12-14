# Segmentacja grafitu na obrazach mikrostruktur żeliwa

Projekt dotyczy opracowania modelu segmentacyjnego do oceny grafitu na obrazach mikrostruktur żeliwa z wykorzystaniem uczenia maszynowego.

## Struktura projektu
- **data/raw/** — surowe dane (obrazy, maski, pliki json)
- **data/processed/** — dane przetworzone
- **notebooks/** — analizy eksploracyjne, prototypy
- **src/data/** — przygotowanie i ładowanie danych
- **src/models/** — architektury sieci
- **src/training/** — skrypty treningowe
- **src/evaluation/** — ewaluacja, metryki
- **src/utils/** — funkcje pomocnicze
- **reports/** — wyniki eksperymentów
- **tests/** — testy jednostkowe

## Wymagania
Zobacz `requirements.txt`.

## Cel
Zbadanie możliwości automatycznej oceny grafitu na obrazach mikrostruktur żeliwa przy użyciu konwolucyjnych sieci neuronowych.

