import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# KONFIGURACJA
POMIARY_DIR = r"C:\Users\user\PycharmProjects\SISE\zad2_norm\pomiary"
OUTPUT_DIR  = "./wyniki"
WINDOW_SIZE = 5         # Sieć patrzy na 5 ostatnich próbek (przyspieszy działanie bez utraty trajektorii)
EPOCHS      = 150
BATCH_SIZE  = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. WCZYTYWANIE DANYCH I DETEKCJA KOLUMN
def load_xlsx_files(data_dir, pattern):
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, engine='openpyxl')
            dfs.append(df)
        except Exception as e:
            print(f"Błąd pliku {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None

def detect_columns(df):
    cols = [c.lower().strip() for c in df.columns]
    orig = df.columns.tolist()
    cx = cy = rx = ry = None
    for i, c in enumerate(cols):
        if 'coordinate' in c and 'x' in c: cx = orig[i]
        elif 'coordinate' in c and 'y' in c: cy = orig[i]
        elif 'reference' in c and 'x' in c: rx = orig[i]
        elif 'reference' in c and 'y' in c: ry = orig[i]
    if cx is None: # Fallback, jeśli nazwy są dziwne
        cx, cy, rx, ry = orig[-4], orig[-3], orig[-2], orig[-1]
    return cx, cy, rx, ry

def get_common_features(df_stat, df_dyn, cx, cy, rx, ry):
    """Bierzemy tylko te kolumny UWB, które występują OBU zbiorach"""
    META = {'unnamed: 0.1', 'unnamed: 0', 'version', 'alive', 'tagid', 'success',
            'timestamp', 'data__anchordata', 'errorcode', 'data__coordinates__z',
            cx.lower(), cy.lower(), rx.lower(), ry.lower()}

    stat_num = set(df_stat.select_dtypes(include=[np.number]).columns)
    dyn_num  = set(df_dyn.select_dtypes(include=[np.number]).columns)

    common = list(stat_num.intersection(dyn_num))
    features = [c for c in common if c.lower() not in META]
    return sorted(features)

# 2. MECHANIZM NA OCENĘ 5 (Pre-processing: Filtr Medianowy)
def remove_outliers(series, window=5, threshold=500):

    rolling_med = series.rolling(window=window, center=True).median().bfill().ffill()
    diff = np.abs(series - rolling_med)
    return np.where(diff > threshold, rolling_med, series)

# 3. TWORZENIE OKIEN CZASOWYCH (Time Steps dla sieci)
def create_dataset(df, features, cx_col, cy_col, rx_col, ry_col, window_size):
    X, Y, raw_cx, raw_cy, ref_x, ref_y = [], [], [], [], [], []

    feat_data = df[features].values
    cx_data = df[cx_col].values
    cy_data = df[cy_col].values
    rx_data = df[rx_col].values
    ry_data = df[ry_col].values

    for i in range(len(df) - window_size):
        window_features = feat_data[i : i + window_size].flatten()

        window_cx = cx_data[i : i + window_size]
        window_cy = cy_data[i : i + window_size]

        X.append(np.concatenate([window_features, window_cx, window_cy]))

        current_cx = cx_data[i + window_size - 1]
        current_cy = cy_data[i + window_size - 1]
        target_rx = rx_data[i + window_size - 1]
        target_ry = ry_data[i + window_size - 1]

        Y.append([target_rx - current_cx, target_ry - current_cy])

        raw_cx.append(current_cx)
        raw_cy.append(current_cy)
        ref_x.append(target_rx)
        ref_y.append(target_ry)

    return np.array(X), np.array(Y), np.array(raw_cx), np.array(raw_cy), np.array(ref_x), np.array(ref_y)

# 4. GŁÓWNY POTOK (Dla wybranej Sali)
def process_sala(sala):
    print(f"\n{'='*50}\nRozpoczynam analizę sali: {sala}\n{'='*50}")
    sala_dir = os.path.join(POMIARY_DIR, sala)
    out_dir = os.path.join(OUTPUT_DIR, sala)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Ładowanie
    df_stat = load_xlsx_files(sala_dir, f"{sala.lower()}_stat_*.xlsx")
    dyn_pats = [f"{sala.lower()}_[0-9]p.xlsx", f"{sala.lower()}_[0-9]z.xlsx", f"{sala.lower()}_random_*.xlsx"]
    dyn_files = [f for pat in dyn_pats for f in sorted(glob.glob(os.path.join(sala_dir, pat)))]
    df_dyn = pd.concat([pd.read_excel(f, engine='openpyxl') for f in dyn_files], ignore_index=True) if dyn_files else None

    if df_stat is None or df_dyn is None:
        print(f"Brak pełnych danych dla {sala}. Pomijam.")
        return

    cx, cy, rx, ry = detect_columns(df_stat)
    df_stat[[rx, ry]] = df_stat[[rx, ry]].ffill().bfill()
    df_dyn[[rx, ry]] = df_dyn[[rx, ry]].ffill().bfill()
    df_stat = df_stat.dropna(subset=[cx, cy, rx, ry]).reset_index(drop=True)
    df_dyn = df_dyn.dropna(subset=[cx, cy, rx, ry]).reset_index(drop=True)

    # 2. Mechanizm na ocenę 5 (Czyszczenie grubych błędów UWB)
    print("-> Aplikuję filtr eliminacji grubych błędów (Mechanizm na 5-kę)...")
    df_stat[cx] = remove_outliers(df_stat[cx])
    df_stat[cy] = remove_outliers(df_stat[cy])
    df_dyn[cx]  = remove_outliers(df_dyn[cx])
    df_dyn[cy]  = remove_outliers(df_dyn[cy])

    # 3. Wybór wspólnych cech
    features = get_common_features(df_stat, df_dyn, cx, cy, rx, ry)
    print(f"-> Znaleziono {len(features)} wspólnych cech dla stat/dyn.")

    # 4. Przygotowanie okien (Time steps)
    X_train, Y_train, _, _, _, _ = create_dataset(df_stat, features, cx, cy, rx, ry, WINDOW_SIZE)
    X_test, Y_test, test_cx, test_cy, test_rx, test_ry = create_dataset(df_dyn, features, cx, cy, rx, ry, WINDOW_SIZE)

    # Skalowanie
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)

    # 5. Tworzenie Sieci Neuronowej (Keras)
    print("-> Budowanie i uczenie sieci neuronowej...")
    model = Sequential([
        Input(shape=(X_train_sc.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(2, activation='linear') # Wyjście: Delta X, Delta Y
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    model.fit(X_train_sc, Y_train, validation_split=0.2, epochs=EPOCHS,
              batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=1)

    # 6. Predykcja i obliczanie błędów euklidesowych
    print("-> Testowanie na danych dynamicznych...")
    predicted_deltas = model.predict(X_test_sc)

    pred_x = test_cx + predicted_deltas[:, 0]
    pred_y = test_cy + predicted_deltas[:, 1]

    err_surowy = np.sqrt((test_cx - test_rx)**2 + (test_cy - test_ry)**2)
    err_siec = np.sqrt((pred_x - test_rx)**2 + (pred_y - test_ry)**2)

    print(f"Mediana błędu surowego: {np.median(err_surowy):.0f} mm")
    print(f"Mediana błędu po sieci: {np.median(err_siec):.0f} mm")

    # 7. GENEROWANIE RAPORTÓW (Wykresy, Excel, Wagi)
    print("-> Generowanie wykresów i plików do raportu...")

    # A) CDF Plot (Dystrybuanta)
    def plot_cdf_single(errors, label, color):
        s = np.sort(errors)
        p = np.arange(1, len(s) + 1) / len(s)
        plt.plot(s, p, label=label, color=color, lw=2)

    plt.figure(figsize=(9, 5))
    plot_cdf_single(err_surowy, "Błąd surowy UWB", "red")
    plot_cdf_single(err_siec, "Błąd po Korekcji (NN)", "blue")
    plt.title(f"Dystrybuanta błędu euklidesowego - {sala}")
    plt.xlabel("Błąd [mm]")
    plt.ylabel("Prawdopodobieństwo")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "1_CDF_wykres.png"), dpi=150)
    plt.close()

    # B) Krzywa Uczenia (Training Curve)
    plt.figure(figsize=(9, 4))
    plt.plot(model.history.history['loss'], label='Train Loss (MSE)', color='steelblue')
    plt.plot(model.history.history['val_loss'], label='Val Loss (MSE)', color='orange')
    plt.title(f"Krzywa uczenia sieci - {sala}")
    plt.xlabel("Epoka")
    plt.ylabel("Błąd MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "2_Krzywa_uczenia.png"), dpi=150)
    plt.close()

    # C) Trajektoria ruchu (Z góry)
    plt.figure(figsize=(11, 6))
    plt.plot(test_rx, test_ry, 'k-', lw=2, label='Referencja (Idealna trasa)')
    plt.plot(test_cx, test_cy, 'b.', ms=3, alpha=0.4, label='UWB surowy')
    plt.plot(pred_x, pred_y, 'r.', ms=3, alpha=0.5, label='Po korekcji NN')
    plt.title(f"Trajektoria ruchu - {sala}")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "3_Trajektoria.png"), dpi=150)
    plt.close()

    # D) Błąd w czasie (Error over time)
    plt.figure(figsize=(12, 4))
    t = np.arange(len(err_surowy))
    plt.plot(t, err_surowy, alpha=0.5, color='blue', label='Surowy Błąd UWB')
    plt.plot(t, err_siec, alpha=0.8, color='red', label='Błąd po Sieci Neuronowej')
    plt.title(f"Błąd w kolejnych próbkach czasu - {sala}")
    plt.xlabel("Próbka")
    plt.ylabel("Błąd [mm]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "4_Blad_w_czasie.png"), dpi=150)
    plt.close()

    # E) Wynikowy Excel
    df_wyniki = pd.DataFrame({
        'Dystrybuanta_Bledu_Euklidesowego_NN': np.sort(err_siec),
        'Blad_Surowy_Euklidesowy': np.sort(err_surowy)
    })
    df_wyniki.to_excel(os.path.join(out_dir, f"5_Wyniki_{sala}.xlsx"), index=False)

    # F) Eksport wag dla raportu
    with open(os.path.join(out_dir, f"6_Wagi_sieci_{sala}.txt"), "w") as f:
        f.write(f"Architektura dla {sala}:\n")
        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if weights:
                f.write(f"\n--- WARSTWA {i + 1} ---\n")
                f.write(f"Shape: {weights[0].shape}\n")
                np.savetxt(f, weights[0][:5, :5], fmt="%.3f")
                f.write("... (ucieto dla czytelnosci)\n")

    print(f"-> Zakończono {sala}. Zapisano 6 plików w katalogu: {out_dir}")

if __name__ == "__main__":
    for sala in ['F8', 'F10']:
        if os.path.exists(os.path.join(POMIARY_DIR, sala)):
            process_sala(sala)
        else:
            print(f"Brak folderu {sala} w {POMIARY_DIR}.")