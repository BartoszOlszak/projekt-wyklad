import cv2
import numpy as np
import os
import glob
import pandas as pd
from google import genai
from PIL import Image
import time

client = genai.Client(api_key="AIzaSyASYhHx5tLMPEnDFKMUju8IGrjY5sxnGFE")  # tutaj dajemy klucz api do google AI studio


def wytnij_zdjecia(zdjecia, wynik):
    # Tworzy folder wynikowy, jeśli jeszcze nie istnieje
    if not os.path.exists(wynik):
        os.makedirs(wynik)

    # Pobiera ścieżki do wszystkich plików w folderze wejściowym
    sciezki = glob.glob(os.path.join(zdjecia, '*'))

    if not sciezki:
        return

    # Lista na ostateczne, odczytane teksty do zapisania w Excelu
    dane_excel_1 = []
    dane_excel_2 = []

    # Tymczasowy "koszyk" na wycięte zdjęcia, z którego zrobimy siatkę
    paczka_obrazkow = []
    paczka_metadane = []
    globalny_licznik = 1

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
        # Rozszerzony zakres łapiący odcienie różu/fioletu
        czerwony21 = np.array([150, 40, 40])
        czerwony22 = np.array([180, 255, 255])

        # Tworzenie maski binarnej (białe tam gdzie czerwone/fioletowe, reszta czarna)
        maska = cv2.inRange(hsv, czerwony11, czerwony12) + cv2.inRange(hsv, czerwony21, czerwony22)

        # "Czyszczenie" maski - pogrubia kształty i zatyka małe dziury wewnątrz liter
        kernel = np.ones((5, 5), np.uint8)
        maska = cv2.dilate(maska, kernel, iterations=1)
        maska = cv2.morphologyEx(maska, cv2.MORPH_CLOSE, kernel)

        # Wykrywanie konturów (płaska lista wszystkich znalezionych kształtów)
        ramki, _ = cv2.findContours(maska, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Przygotowanie nazwy bazowej pliku do zapisu wyników
        nazwa = os.path.splitext(os.path.basename(sciezka))[0]
        zdj_h, zdj_w = zdjecie.shape[:2]

        for ramka in ramki:
            # Pobranie wymiarów prostokąta otaczającego kontur
            x, y, w, h = cv2.boundingRect(ramka)
            pole = cv2.contourArea(ramka)

            # FILTRY ODRZUCAJĄCE BŁĘDNE WYNIKI (np. plamki lub wielkie bloki):
            # Zbyt małe pole powierzchni
            if pole < 500:
                continue

            # Zbyt duże obiekty (zajmujące więcej niż 20% szerokości/wysokości całego zdjęcia)
            if w > zdj_w * 0.2 or h > zdj_h * 0.2:
                continue

            # Obiekty o nienaturalnie małych wymiarach bezwzględnych
            if w < 80 or h < 40:
                continue

            # Filtr proporcji (odrzuca pionowe kreski i zbyt długie poziome pasy)
            proporcje = float(w) / h
            if proporcje < 1.1 or proporcje > 6:
                continue

            # Obliczanie współrzędnych do wycięcia - dodajemy po 5px marginesu, żeby nie ucinać liter
            px = max(0, x - 5)
            py = max(0, y - 5)
            pw = min(zdj_w - px, w + 10)
            ph = min(zdj_h - py, h + 10)

            # Wycięcie fragmentu z oryginalnego zdjęcia i zapisanie jako nowy plik JPG
            wyciety = zdjecie[py:py + ph, px:px + pw]

            # Konwersja wycinka do formatu obsługiwanego przez Google AI (PIL Image)
            rgb_crop = cv2.cvtColor(wyciety, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_crop)
            # Wrzuć wycięte słowo do koszyka oczekującego na wysłanie
            paczka_obrazkow.append(pil_img)
            paczka_metadane.append(str(globalny_licznik))

            # Fizyczny zapis wycinka na dysku w folderze 'wynik'
            nazwa_wyniku = f"{globalny_licznik}.jpg"
            sciezka_wyniku = os.path.join(wynik, nazwa_wyniku)
            cv2.imwrite(sciezka_wyniku, wyciety)

            globalny_licznik += 1

            # --- GŁÓWNY SYSTEM SIATKI ---
            # Kiedy koszyk uzbiera 100 obrazków, zaczynamy proces sklejania
            if len(paczka_obrazkow) >= 100:
                # Szukamy największej szerokości i wysokości w paczce, żeby wyrównać komórki siatki
                max_w = max(img.width for img in paczka_obrazkow)
                max_h = max(img.height for img in paczka_obrazkow)

                kolumny = 10
                wiersze = (len(paczka_obrazkow) + kolumny - 1) // kolumny

                siatka = Image.new('RGB', (kolumny * max_w, wiersze * max_h), color='white')

                # Wklejamy poszczególne słowa na odpowiednie miejsca siatki
                for idx, img in enumerate(paczka_obrazkow):
                    pos_x = (idx % kolumny) * max_w
                    pos_y = (idx // kolumny) * max_h
                    siatka.paste(img, (pos_x, pos_y))

                try:
                    # Wysyłamy do AI tylko całą siatkę zamiast małych
                    prompt = f"Na obrazku jest siatka wyciętych, odręcznych słów. Komórek jest dokładnie {len(paczka_obrazkow)}. Odczytaj wszystkie słowa po kolei, czytając wierszami od lewej do prawej. Zwróć dokładnie {len(paczka_obrazkow)} linijek, po jednej dla każdego słowa. Jeśli w jakiejś komórce nie rozpoznajesz słowa lub jest pusta, wpisz znak '-'. ZACZNIJ ODPOWIEDŹ BEZPOŚREDNIO OD PIERWSZEGO SŁOWA. Nie używaj żadnych zdań wstępnych, nie pisz 'Oto lista', nie dodawaj nic od siebie."
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[prompt, siatka]
                    )

                    # Odbieramy długi tekst, dzielimy go na linijki
                    tekst = response.text.strip().split('\n')
                    tekst = [t.strip() for t in tekst if t.strip()]

                    if tekst and (":" in tekst[0] or len(tekst[0].split()) > 3):
                        tekst = tekst[1:]

                    while len(tekst) < len(paczka_metadane):
                        tekst.append('-')
                    tekst = tekst[:len(paczka_metadane)]

                    # Filtr śmieciowy - odrzuca znaki interpunkcyjne potraktowane jako słowa
                    for m, t in zip(paczka_metadane, tekst):
                        oczyszczony = t.strip()

                        if oczyszczony:
                            oczyszczony = oczyszczony.split()[0]

                        if not oczyszczony or oczyszczony in ['--', '_', '.', ',', '|'] or oczyszczony.lower() in ['2m',
                                                                                                                   'zm']:
                            oczyszczony = '-'

                        dane_excel_2.append([m, oczyszczony])

                        if oczyszczony != '-':
                            dane_excel_1.append([m, oczyszczony])
                            print(f"Rozpoznano z siatki (Plik: {m}): {oczyszczony}")
                        else:
                            print(f"Nie rozpoznano słowa (Plik: {m})")

                except Exception as e:
                    print(f"Błąd API: {e}")

                # Czyścimy koszyk, żeby zbierać kolejną partię słów
                paczka_obrazkow = []
                paczka_metadane = []

                time.sleep(35)

    # --- CZYSZCZENIE RESZTEK ---
    # Ten blok wykonuje się po zakończeniu wszystkich zdjęć
    # Jeśli w koszyku zostało np. 45 zdjęć (nie dobiło do 60), to wysyłamy je teraz
    if paczka_obrazkow:
        max_w = max(img.width for img in paczka_obrazkow)
        max_h = max(img.height for img in paczka_obrazkow)

        kolumny = 10
        wiersze = (len(paczka_obrazkow) + kolumny - 1) // kolumny

        siatka = Image.new('RGB', (kolumny * max_w, wiersze * max_h), color='white')

        for idx, img in enumerate(paczka_obrazkow):
            pos_x = (idx % kolumny) * max_w
            pos_y = (idx // kolumny) * max_h
            siatka.paste(img, (pos_x, pos_y))

        try:
            prompt = f"Na obrazku jest siatka wyciętych, odręcznych słów. Komórek jest dokładnie {len(paczka_obrazkow)}. Odczytaj wszystkie słowa po kolei, czytając wierszami od lewej do prawej. Zwróć dokładnie {len(paczka_obrazkow)} linijek, po jednej dla każdego słowa. Jeśli w jakiejś komórce nie rozpoznajesz słowa lub jest pusta, wpisz znak '-'. ZACZNIJ ODPOWIEDŹ BEZPOŚREDNIO OD PIERWSZEGO SŁOWA. Nie używaj żadnych zdań wstępnych, nie pisz 'Oto lista', nie dodawaj nic od siebie."
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=[prompt, siatka]
            )
            tekst = response.text.strip().split('\n')
            tekst = [t.strip() for t in tekst if t.strip()]

            if tekst and (":" in tekst[0] or len(tekst[0].split()) > 3):
                tekst = tekst[1:]

            while len(tekst) < len(paczka_metadane):
                tekst.append('-')
            tekst = tekst[:len(paczka_metadane)]

            for m, t in zip(paczka_metadane, tekst):
                oczyszczony = t.strip()

                if oczyszczony:
                    oczyszczony = oczyszczony.split()[0]

                if not oczyszczony or oczyszczony in ['--', '_', '.', ',', '|'] or oczyszczony.lower() in ['2m', 'zm']:
                    oczyszczony = '-'

                dane_excel_2.append([m, oczyszczony])

                if oczyszczony != '-':
                    dane_excel_1.append([m, oczyszczony])
                    print(f"Rozpoznano z siatki (Plik: {m}): {oczyszczony}")
                else:
                    print(f"Nie rozpoznano słowa (Plik: {m})")
        except Exception as e:
            print(f"Błąd API: {e}")

    # Zapis całego odczytanego tekstu do pliku Excel
    if dane_excel_1:
        df1 = pd.DataFrame(dane_excel_1, columns=['Plik', 'Rozpoznany Tekst'])
        df1.to_excel('wynikexcel_rozpoznane.xlsx', index=False)
        print("Utworzono plik wynikexcel_rozpoznane.xlsx")

    if dane_excel_2:
        df2 = pd.DataFrame(dane_excel_2, columns=['Plik', 'Rozpoznany Tekst'])
        df2.to_excel('wynikexcel_wszystkie.xlsx', index=False)
        print("Utworzono plik wynikexcel_wszystkie.xlsx")


# Ścieżki do folderów wejścia i wyjścia
zdjecia = 'zdjecia'
wynik = 'wynik'

wytnij_zdjecia(zdjecia, wynik)