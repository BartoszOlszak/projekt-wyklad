import cv2
import numpy as np
import os
import glob

def process_images(zdjecia, wynik):
    # Tworzy folder wynikowy, jeśli jeszcze nie istnieje
    if not os.path.exists(wynik):
        os.makedirs(wynik)

    # Pobiera ścieżki do wszystkich plików w folderze wejściowym
    sciezki = glob.glob(os.path.join(zdjecia, '*'))

    if not sciezki:
        return

    for sciezka in sciezki:
        # Wczytanie zdjęcia z dysku
        zdjecie = cv2.imread(sciezka)
        if zdjecie is None:
            continue

        # Konwersja na model HSV (łatwiejsze wycinanie konkretnych kolorów)
        hsv = cv2.cvtColor(zdjecie, cv2.COLOR_BGR2HSV)

        # Definicja zakresów koloru czerwonego (czerwień jest na obu końcach skali H)
        czerwony11 = np.array([0, 40, 40])
        czerwony12 = np.array([10, 255, 255])
        czerwony21 = np.array([170, 40, 40])
        czerwony22 = np.array([180, 255, 255])

        # Tworzenie maski binarnej (białe tam gdzie czerwone, reszta czarna)
        maska = cv2.inRange(hsv, czerwony11, czerwony12) + cv2.inRange(hsv, czerwony21, czerwony22)

        kernel = np.ones((5, 5), np.uint8)
        maska = cv2.dilate(maska, kernel, iterations=1)
        maska = cv2.morphologyEx(maska, cv2.MORPH_CLOSE, kernel)

        # Wykrywanie konturów (płaska lista wszystkich znalezionych kształtów)
        ramki, _ = cv2.findContours(maska, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Przygotowanie nazwy bazowej pliku do zapisu wyników
        nazwa = os.path.splitext(os.path.basename(sciezka))[0]
        licznik = 0
        zdj_h, zdj_w = zdjecie.shape[:2]

        for ramka in ramki:
            # Pobranie wymiarów prostokąta otaczającego kontur
            x, y, w, h = cv2.boundingRect(ramka)
            pole = cv2.contourArea(ramka)

            # FILTRY ODRZUCAJĄCE BŁĘDNE WYNIKI:
            # Zbyt małe pole powierzchni
            if pole < 300:
                continue

            # Zbyt duże obiekty
            if w > zdj_w * 0.6 or h > zdj_h * 0.6:
                continue

            # Obiekty o nienaturalnie małych wymiarach
            if w < 40 or h < 20:
                continue

            # Filtr proporcji (odrzuca pionowe kreski i zbyt długie pasy)
            proporcje = float(w) / h
            if proporcje < 1.1 or proporcje > 6:
                continue

            # Obliczanie współrzędnych z marginesem 5px i blokadą wyjścia poza obraz
            px = max(0, x - 5)
            py = max(0, y - 5)
            pw = min(zdj_w - px, w + 10)
            ph = min(zdj_h - py, h + 10)

            # Wycięcie fragmentu i zapisanie jako nowy plik JPG
            wyciety = zdjecie[py:py + ph, px:px + pw]
            nazwa_wyniku = f"{nazwa}_crop_{licznik}.jpg"
            sciezka_wyniku = os.path.join(wynik, nazwa_wyniku)

            cv2.imwrite(sciezka_wyniku, wyciety)
            licznik += 1

# Ścieżki do folderów
zdjecia = 'zdjecia'
wynik = 'wynik'

process_images(zdjecia, wynik)