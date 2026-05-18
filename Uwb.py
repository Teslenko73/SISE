"""
Korekcja pomiarów UWB przy użyciu sieci neuronowej
Badania nad poprawą lokalizacji UWB przy pomocy sieci neuronowych

ZMIANY względem poprzedniej wersji:
  - Osobne sieci dla sal F8 i F10 (eliminacja błędu site-dependent)
  - Dane dynamiczne (okrążenia 1p/1z/2p/2z/3p/3z + random) użyte do TESTOWANIA
  - Statystyki i dystrybuanty generowane osobno dla każdej sali
  - Wyniki zbiorczy xlsx zawiera dystrybuantę z DANYCH DYNAMICZNYCH (testowych)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURACJA
# ============================================================
POMIARY_DIR = "./pomiary"         # <-- zmień na ścieżkę do katalogu pomiary
OUTPUT_DIR  = "./wyniki"
SEQUENCE_LEN = 5                  # liczba próbek z poprzednich chwil czasowych
EPOCHS = 100
BATCH_SIZE = 32
HIDDEN_LAYERS = [128, 64, 32]     # neurony w ukrytych warstwach
LEARNING_RATE = 0.001
OUTLIER_THRESHOLD = 3.0           # próg do eliminacji błędnych pomiarów (sigma)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. WCZYTYWANIE DANYCH
# ============================================================

def load_xlsx_files(data_dir, pattern="*.xlsx"):
    """Wczytuje wszystkie pliki xlsx pasujące do wzorca z podanego katalogu."""
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        return []
    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, engine='openpyxl')
            df['_source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Błąd wczytywania {f}: {e}")
    return dfs


def load_sala_data(pomiary_dir, sala):
    """
    Wczytuje dane statyczne i dynamiczne dla jednej sali (F8 lub F10).

    Zwraca:
        df_static  – połączone pliki stat_* (do UCZENIA)
        df_dynamic – połączone pliki dynamiczne: Np, Nz, random (do TESTOWANIA)
    """
    sala_dir = os.path.join(pomiary_dir, sala)
    prefix = sala.lower()  # f8 lub f10

    if not os.path.isdir(sala_dir):
        raise FileNotFoundError(f"Katalog {sala_dir} nie istnieje.")

    # --- Statyczne (trening) ---
    static_dfs = load_xlsx_files(sala_dir, pattern=f"{prefix}_stat_*.xlsx")
    if not static_dfs:
        raise FileNotFoundError(f"Brak plików statycznych w {sala_dir}")
    df_static = pd.concat(static_dfs, ignore_index=True)
    print(f"  [{sala}] Statyczne: {len(static_dfs)} plików, {len(df_static)} wierszy")

    # --- Dynamiczne (test): okrążenia + random ---
    dyn_patterns = [
        f"{prefix}_[0-9]p.xlsx",
        f"{prefix}_[0-9]z.xlsx",
        f"{prefix}_random_*.xlsx",
    ]
    dyn_dfs = []
    for pat in dyn_patterns:
        found = load_xlsx_files(sala_dir, pattern=pat)
        dyn_dfs.extend(found)

    if dyn_dfs:
        df_dynamic = pd.concat(dyn_dfs, ignore_index=True)
        print(f"  [{sala}] Dynamiczne: {len(dyn_dfs)} plików, {len(df_dynamic)} wierszy")
    else:
        print(f"  [{sala}] UWAGA: Brak plików dynamicznych — test na danych statycznych.")
        df_dynamic = None

    return df_static, df_dynamic


def detect_column_names(df):
    """Automatycznie wykrywa nazwy kolumn dla wsp. pomiarowych i referencyjnych."""
    cols = [c.lower().strip() for c in df.columns]
    original = df.columns.tolist()

    coord_x = coord_y = ref_x = ref_y = None
    for i, c in enumerate(cols):
        if 'coordinate' in c and 'x' in c:
            coord_x = original[i]
        elif 'coordinate' in c and 'y' in c:
            coord_y = original[i]
        elif 'reference' in c and 'x' in c:
            ref_x = original[i]
        elif 'reference' in c and 'y' in c:
            ref_y = original[i]

    if coord_x is None:
        coord_x = original[-4]
        coord_y = original[-3]
        ref_x   = original[-2]
        ref_y   = original[-1]

    print(f"  Wykryte kolumny:")
    print(f"    Pomiar X: '{coord_x}', Pomiar Y: '{coord_y}'")
    print(f"    Referencja X: '{ref_x}', Referencja Y: '{ref_y}'")
    return coord_x, coord_y, ref_x, ref_y


def prepare_df(df, feature_cols, coord_x, coord_y, ref_x, ref_y):
    """
    Uzupełnia NaN w kolumnach referencyjnych (ffill/bfill),
    usuwa wiersze z NaN w kolumnach cech i pomiarowych.
    """
    for col in [ref_x, ref_y]:
        df[col] = df[col].ffill().bfill()

    df = df.dropna(subset=feature_cols + [coord_x, coord_y])
    df = df.dropna(subset=[ref_x, ref_y])
    df = df.reset_index(drop=True)
    return df


# ============================================================
# 2. MECHANIZM ELIMINACJI BŁĘDNYCH POMIARÓW
# ============================================================

class OutlierDetector:
    """
    Eliminuje próbki obarczone ewidentnym, dużym błędem.
    Metoda: Z-score na podstawie błędu euklidesowego pomiaru względem
    wygładzonej (mediana krocząca) trajektorii.
    """
    def __init__(self, threshold=3.0, window=10):
        self.threshold = threshold
        self.window = window

    def fit(self, x_meas, y_meas):
        self.median_x = pd.Series(x_meas).rolling(self.window, center=True, min_periods=1).median().values
        self.median_y = pd.Series(y_meas).rolling(self.window, center=True, min_periods=1).median().values
        dist = np.sqrt((x_meas - self.median_x)**2 + (y_meas - self.median_y)**2)
        self.mean_dist = np.mean(dist)
        self.std_dist  = np.std(dist)
        return self

    def clean(self, x_meas, y_meas):
        """Zastępuje outliery medianową trajektorią."""
        median_x = pd.Series(x_meas).rolling(self.window, center=True, min_periods=1).median().values
        median_y = pd.Series(y_meas).rolling(self.window, center=True, min_periods=1).median().values
        dist = np.sqrt((x_meas - median_x)**2 + (y_meas - median_y)**2)
        z_score = (dist - self.mean_dist) / (self.std_dist + 1e-9)
        mask = z_score < self.threshold
        n_removed = np.sum(~mask)
        print(f"    Usunięto {n_removed} outlierów ({100*n_removed/len(mask):.1f}%)")
        x_clean = x_meas.copy()
        y_clean = y_meas.copy()
        x_clean[~mask] = median_x[~mask]
        y_clean[~mask] = median_y[~mask]
        return x_clean, y_clean, mask


# ============================================================
# 3. BUDOWANIE MACIERZY CECH (time-window)
# ============================================================

def build_feature_matrix(df, feature_cols, coord_x, coord_y, ref_x, ref_y,
                          seq_len=5, outlier_detector=None):
    X_raw = df[feature_cols].values.astype(float)
    cx    = df[coord_x].values.astype(float)
    cy    = df[coord_y].values.astype(float)
    rx    = df[ref_x].values.astype(float)
    ry    = df[ref_y].values.astype(float)

    if outlier_detector is not None:
        cx, cy, _ = outlier_detector.clean(cx, cy)

    N = len(X_raw)
    features, targets_x, targets_y, raw_errors = [], [], [], []

    for i in range(seq_len - 1, N):
        window_feats = X_raw[i - seq_len + 1 : i + 1].flatten()
        window_cx = cx[i - seq_len + 1 : i + 1]
        window_cy = cy[i - seq_len + 1 : i + 1]
        feat = np.concatenate([window_feats, window_cx, window_cy])
        features.append(feat)
        targets_x.append(rx[i] - cx[i])
        targets_y.append(ry[i] - cy[i])
        raw_errors.append(np.sqrt((cx[i] - rx[i])**2 + (cy[i] - ry[i])**2))

    return (np.array(features),
            np.array(targets_x),
            np.array(targets_y),
            np.array(raw_errors),
            cx[seq_len-1:], cy[seq_len-1:],
            rx[seq_len-1:], ry[seq_len-1:])


# ============================================================
# 4. SIEĆ NEURONOWA (implementacja NumPy — bez zewnętrznych frameworków ML)
# ============================================================

def relu(x):       return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def linear(x):     return x
def linear_deriv(x): return np.ones_like(x)


class DenseLayer:
    def __init__(self, n_in, n_out, activation='relu', seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        self.activation_name = activation
        self.act   = relu if activation == 'relu' else linear
        self.act_d = relu_deriv if activation == 'relu' else linear_deriv
        self.z = self.a = self.dW = self.db = self.input = None

    def forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b
        self.a = self.act(self.z)
        return self.a

    def backward(self, delta):
        d = delta * self.act_d(self.z)
        self.dW = self.input.T @ d
        self.db = d.sum(axis=0, keepdims=True)
        return d @ self.W.T


class NeuralNetwork:
    """
    Wielowarstwowa sieć neuronowa z propagacją wsteczną.

    Architektura:
      - Warstwa wejściowa: n_features neuronów
      - Warstwy ukryte: hidden_layers neuronów, aktywacja ReLU
      - Warstwa wyjściowa: 2 neurony (delta_x, delta_y), aktywacja liniowa

    Algorytm uczenia: SGD z momentum (mini-batch)
    """
    def __init__(self, n_features, hidden_layers, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.layers = []
        sizes = [n_features] + hidden_layers + [2]
        activations = ['relu'] * len(hidden_layers) + ['linear']
        for i in range(len(sizes) - 1):
            self.layers.append(DenseLayer(sizes[i], sizes[i+1], activations[i], seed=RANDOM_SEED+i))
        self.vW = [np.zeros_like(l.W) for l in self.layers]
        self.vb = [np.zeros_like(l.b) for l in self.layers]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_pred, y_true):
        delta = 2 * (y_pred - y_true) / len(y_true)
        for i, layer in reversed(list(enumerate(self.layers))):
            delta = layer.backward(delta)
        for i, layer in enumerate(self.layers):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * layer.dW
            self.vb[i] = self.momentum * self.vb[i] - self.lr * layer.db
            layer.W += self.vW[i]
            layer.b += self.vb[i]

    def fit(self, X_train, Y_train, X_val, Y_val, epochs=100, batch_size=32):
        n = len(X_train)
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuf, Y_shuf = X_train[idx], Y_train[idx]
            epoch_loss = 0
            for start in range(0, n, batch_size):
                Xb, Yb = X_shuf[start:start+batch_size], Y_shuf[start:start+batch_size]
                pred = self.forward(Xb)
                epoch_loss += np.mean((pred - Yb)**2) * len(Xb)
                self.backward(pred, Yb)
            train_loss = epoch_loss / n
            val_pred = self.forward(X_val)
            val_loss = np.mean((val_pred - Y_val)**2)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if (epoch + 1) % 10 == 0:
                print(f"    Epoka {epoch+1:3d}/{epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        return train_losses, val_losses

    def predict(self, X):
        return self.forward(X)

    def get_weights_info(self):
        info = []
        for i, layer in enumerate(self.layers):
            info.append({
                'warstwa': i + 1,
                'n_in': layer.W.shape[0],
                'n_out': layer.W.shape[1],
                'aktywacja': layer.activation_name,
                'W_mean': np.mean(layer.W),
                'W_std': np.std(layer.W),
                'W_min': np.min(layer.W),
                'W_max': np.max(layer.W),
            })
        return info


# ============================================================
# 5. OBLICZANIE DYSTRYBUANTY BŁĘDU (CDF)
# ============================================================

def compute_cdf(errors):
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    return sorted_errors, cdf


# ============================================================
# 6. WYKRESY
# ============================================================

def plot_training_curve(train_losses, val_losses, save_path, title=""):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='Train MSE')
    ax.plot(val_losses, label='Walidacja MSE')
    ax.set_xlabel('Epoka')
    ax.set_ylabel('MSE')
    ax.set_title(f'Krzywa uczenia {title}')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_cdf_comparison(raw_errors, nn_errors, nn_outlier_errors, save_path, title=""):
    fig, ax = plt.subplots(figsize=(9, 5))
    e1, c1 = compute_cdf(raw_errors)
    ax.plot(e1, c1, 'b-',  linewidth=2, label='Dane surowe (bez filtracji)')
    e2, c2 = compute_cdf(nn_errors)
    ax.plot(e2, c2, 'g--', linewidth=2, label='Po filtracji NN (bez eliminacji outlierów)')
    if nn_outlier_errors is not None:
        e3, c3 = compute_cdf(nn_outlier_errors)
        ax.plot(e3, c3, 'r-.', linewidth=2, label='Po filtracji NN (z eliminacją outlierów)')
    ax.set_xlabel('Błąd euklidesowy [mm]')
    ax.set_ylabel('Dystrybuanta F(e)')
    ax.set_title(f'Porównanie dystrybuant błędu lokalizacji UWB {title}')
    ax.legend(); ax.grid(True, alpha=0.4)
    ax.set_xlim(left=0); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_trajectory(cx, cy, rx, ry, pred_x, pred_y, save_path, title="Trajektoria"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rx, ry, 'k-',  linewidth=2, label='Referencja')
    ax.plot(cx, cy, 'b.',  markersize=3, alpha=0.5, label='Pomiar UWB')
    ax.plot(pred_x, pred_y, 'r.', markersize=3, alpha=0.5, label='Po korekcji NN')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


# ============================================================
# 7. POTOK DLA JEDNEJ SALI
# ============================================================

def get_feature_cols(df, coord_x, coord_y, ref_x, ref_y):
    META_COLS = {
        'unnamed: 0.1', 'unnamed: 0', 'version', 'alive', 'tagid',
        'success', 'timestamp', 'data__anchordata', 'errorcode',
        'data__coordinates__z', '_source_file',
        ref_x.lower(), ref_y.lower(),
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c.lower() not in META_COLS]
    return feature_cols


def run_sala_pipeline(sala, df_static, df_dynamic, use_outlier_detection=True):
    """
    Trenuje dwie sieci (z/bez outlier elimination) na danych STATYCZNYCH danej sali.
    Testuje na danych DYNAMICZNYCH (jeśli dostępne) lub 30% statycznych.

    Zwraca nn_err_out z danych dynamicznych (do globalnej dystrybuanty).
    """
    print(f"\n{'='*60}")
    print(f"  Sala: {sala}")
    print(f"{'='*60}")

    sala_out = os.path.join(OUTPUT_DIR, sala)
    os.makedirs(sala_out, exist_ok=True)

    # --- Wykryj kolumny ---
    coord_x, coord_y, ref_x, ref_y = detect_column_names(df_static)
    feature_cols = get_feature_cols(df_static, coord_x, coord_y, ref_x, ref_y)

    # --- Przygotuj dane statyczne ---
    use_cols = list(dict.fromkeys(feature_cols + [coord_x, coord_y, ref_x, ref_y]))
    df_stat = df_static[use_cols].copy()
    df_stat = prepare_df(df_stat, feature_cols, coord_x, coord_y, ref_x, ref_y)
    print(f"  Próbki statyczne (trening): {len(df_stat)}")
    if len(df_stat) == 0:
        print("  BŁĄD: Brak danych statycznych po czyszczeniu. Pomijam salę.")
        return None, None, None

    # --- Outlier detector (dopasowany na danych statycznych) ---
    od = OutlierDetector(threshold=OUTLIER_THRESHOLD, window=15)
    od.fit(df_stat[coord_x].values.astype(float),
           df_stat[coord_y].values.astype(float))

    # --- Buduj macierze cech (statyczne) ---
    print(f"  Budowanie cech treningowych (seq_len={SEQUENCE_LEN})...")
    (X_raw, dX_raw, dY_raw, raw_err_stat,
     cx_raw, cy_raw, rx_arr, ry_arr) = build_feature_matrix(
        df_stat, feature_cols, coord_x, coord_y, ref_x, ref_y,
        seq_len=SEQUENCE_LEN, outlier_detector=None)

    (X_out, dX_out, dY_out, _,
     cx_out, cy_out, _, _) = build_feature_matrix(
        df_stat, feature_cols, coord_x, coord_y, ref_x, ref_y,
        seq_len=SEQUENCE_LEN, outlier_detector=od)

    Y_raw = np.stack([dX_raw, dY_raw], axis=1)
    Y_out = np.stack([dX_out, dY_out], axis=1)

    # --- Normalizacja (fit na danych statycznych) ---
    scaler_X = StandardScaler()
    X_raw_sc = scaler_X.fit_transform(X_raw)
    X_out_sc = scaler_X.transform(X_out)

    scaler_Y     = StandardScaler()
    scaler_Y_out = StandardScaler()
    Y_raw_sc = scaler_Y.fit_transform(Y_raw)
    Y_out_sc = scaler_Y_out.fit_transform(Y_out)

    # --- Podział statycznych na train/val (70/30) ---
    split = int(0.7 * len(X_raw_sc))
    X_train,   X_val   = X_raw_sc[:split], X_raw_sc[split:]
    Y_train,   Y_val   = Y_raw_sc[:split], Y_raw_sc[split:]
    X_train_o, X_val_o = X_out_sc[:split], X_out_sc[split:]
    Y_train_o, Y_val_o = Y_out_sc[:split], Y_out_sc[split:]

    n_features = X_train.shape[1]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Cechy: {n_features}")

    # ====== SIEĆ 1: bez eliminacji outlierów ======
    print(f"\n  [Sieć 1 – {sala}] Trening (bez outlier elimination)...")
    nn = NeuralNetwork(n_features, HIDDEN_LAYERS, lr=LEARNING_RATE)
    tl, vl = nn.fit(X_train, Y_train, X_val, Y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_training_curve(tl, vl,
                        os.path.join(sala_out, "training_curve_no_outlier.png"),
                        title=f"{sala} – bez outlier")

    # ====== SIEĆ 2: z eliminacją outlierów ======
    print(f"\n  [Sieć 2 – {sala}] Trening (z outlier elimination)...")
    nn_out = NeuralNetwork(n_features, HIDDEN_LAYERS, lr=LEARNING_RATE)
    tl2, vl2 = nn_out.fit(X_train_o, Y_train_o, X_val_o, Y_val_o, epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_training_curve(tl2, vl2,
                        os.path.join(sala_out, "training_curve_with_outlier.png"),
                        title=f"{sala} – z outlier")

    # ====== DANE TESTOWE: dynamiczne (jeśli dostępne) lub 30% statycznych ======
    if df_dynamic is not None and len(df_dynamic) > 0:
        print(f"\n  Przygotowanie danych DYNAMICZNYCH do testowania...")
        df_dyn = df_dynamic[
            [c for c in use_cols if c in df_dynamic.columns]
        ].copy()
        df_dyn = prepare_df(df_dyn, feature_cols, coord_x, coord_y, ref_x, ref_y)
        print(f"  Próbki dynamiczne (test): {len(df_dyn)}")

        if len(df_dyn) >= SEQUENCE_LEN:
            (X_test_dyn, dX_test, dY_test, raw_errors_test,
             cx_test, cy_test, rx_test, ry_test) = build_feature_matrix(
                df_dyn, feature_cols, coord_x, coord_y, ref_x, ref_y,
                seq_len=SEQUENCE_LEN, outlier_detector=None)

            (X_test_dyn_o, _, _, _,
             cx_test_o, cy_test_o, _, _) = build_feature_matrix(
                df_dyn, feature_cols, coord_x, coord_y, ref_x, ref_y,
                seq_len=SEQUENCE_LEN, outlier_detector=od)

            X_test_sc   = scaler_X.transform(X_test_dyn)
            X_test_sc_o = scaler_X.transform(X_test_dyn_o)
            test_source = "dynamiczne"
        else:
            print("  UWAGA: Za mało danych dynamicznych — test na zbiorze walidacyjnym.")
            X_test_sc   = X_val
            X_test_sc_o = X_val_o
            raw_errors_test = raw_err_stat[split:]
            cx_test  = cx_raw[split:]; cy_test  = cy_raw[split:]
            rx_test  = rx_arr[split:]; ry_test  = ry_arr[split:]
            cx_test_o = cx_out[split:]; cy_test_o = cy_out[split:]
            test_source = "statyczne-walidacja"
    else:
        # Fallback: 30% statycznych
        X_test_sc   = X_val
        X_test_sc_o = X_val_o
        raw_errors_test = raw_err_stat[split:]
        cx_test  = cx_raw[split:]; cy_test  = cy_raw[split:]
        rx_test  = rx_arr[split:]; ry_test  = ry_arr[split:]
        cx_test_o = cx_out[split:]; cy_test_o = cy_out[split:]
        test_source = "statyczne-walidacja"

    print(f"  Źródło danych testowych: {test_source}")

    # --- Predykcja na zbiorze testowym ---
    pred_delta_sc   = nn.predict(X_test_sc)
    pred_delta      = scaler_Y.inverse_transform(pred_delta_sc)
    pred_x  = cx_test + pred_delta[:, 0]
    pred_y  = cy_test + pred_delta[:, 1]
    nn_errors = np.sqrt((pred_x - rx_test)**2 + (pred_y - ry_test)**2)

    pred_delta_o_sc = nn_out.predict(X_test_sc_o)
    pred_delta_o    = scaler_Y_out.inverse_transform(pred_delta_o_sc)
    pred_x_o = cx_test_o + pred_delta_o[:, 0]
    pred_y_o = cy_test_o + pred_delta_o[:, 1]
    nn_errors_out = np.sqrt((pred_x_o - rx_test)**2 + (pred_y_o - ry_test)**2)
    df_pred = pd.DataFrame({
        'pred_x': pred_x_o,
        'pred_y': pred_y_o,
        'ref_x': rx_test,
        'ref_y': ry_test,
        'blad_mm': nn_errors_out,
    })
    pred_path = os.path.join(sala_out, f"poprawione_xy_{sala}.xlsx")
    df_pred.to_excel(pred_path, index=False, engine='openpyxl')
    print(f"  Zapisano: {pred_path}")

    # ====== Wykresy ======
    print(f"\n  Generowanie wykresów dla sali {sala}...")
    plot_cdf_comparison(
        raw_errors_test, nn_errors, nn_errors_out,
        os.path.join(sala_out, "cdf_comparison.png"),
        title=f"– {sala} ({test_source})"
    )
    plot_trajectory(
        cx_test, cy_test, rx_test, ry_test, pred_x, pred_y,
        os.path.join(sala_out, "trajectory_no_outlier.png"),
        f"Trajektoria {sala} – NN bez outlier ({test_source})"
    )
    plot_trajectory(
        cx_test_o, cy_test_o, rx_test, ry_test, pred_x_o, pred_y_o,
        os.path.join(sala_out, "trajectory_with_outlier.png"),
        f"Trajektoria {sala} – NN z outlier ({test_source})"
    )

    # ====== Statystyki ======
    print(f"\n  Statystyki błędów [{sala}] (test: {test_source}):")
    print(f"    Błąd surowy   – mediana: {np.median(raw_errors_test):.1f} mm,"
          f" 75%: {np.percentile(raw_errors_test, 75):.1f} mm")
    print(f"    NN (bez out.) – mediana: {np.median(nn_errors):.1f} mm,"
          f" 75%: {np.percentile(nn_errors, 75):.1f} mm")
    print(f"    NN (z out.)   – mediana: {np.median(nn_errors_out):.1f} mm,"
          f" 75%: {np.percentile(nn_errors_out, 75):.1f} mm")

    # ====== Architektura ======
    print(f"\n  Architektura sieci [{sala}]:")
    print(f"    Warstwy: wejście={n_features}, ukryte={HIDDEN_LAYERS}, wyjście=2")
    print(f"    Aktywacje: ReLU (ukryte), Liniowa (wyjście)")
    print(f"    Okno czasowe: {SEQUENCE_LEN}")
    print(f"    SGD z momentum={nn.momentum}, lr={LEARNING_RATE}")
    for info in nn_out.get_weights_info():
        print(f"    Warstwa {info['warstwa']} ({info['n_in']}→{info['n_out']}, {info['aktywacja']}): "
              f"W∈[{info['W_min']:.3f}, {info['W_max']:.3f}], "
              f"μ={info['W_mean']:.3f}, σ={info['W_std']:.3f}")

    return raw_errors_test, nn_errors, nn_errors_out


# ============================================================
# 8. GŁÓWNY POTOK
# ============================================================

def run_pipeline(pomiary_dir):
    print("=" * 60)
    print("  Korekcja pomiarów UWB - Sieć Neuronowa")
    print("  Osobne sieci dla sal F8 i F10")
    print("  Trening: dane statyczne | Test: dane dynamiczne")
    print("=" * 60)

    all_raw, all_nn, all_nn_out = [], [], []
    results = {}

    for sala in ['F8', 'F10']:
        print(f"\n[Wczytywanie {sala}]...")
        try:
            df_static, df_dynamic = load_sala_data(pomiary_dir, sala)
        except FileNotFoundError as e:
            print(f"  POMINIĘTO {sala}: {e}")
            continue

        raw_err, nn_err, nn_err_out = run_sala_pipeline(sala, df_static, df_dynamic)
        if raw_err is None:
            continue

        all_raw.extend(raw_err.tolist())
        all_nn.extend(nn_err.tolist())
        all_nn_out.extend(nn_err_out.tolist())
        results[sala] = (raw_err, nn_err, nn_err_out)

    if not all_nn_out:
        print("\nBłąd: brak wyników do zapisu.")
        return

    # ====== Zbiorczy wykres CDF (obie sale razem) ======
    print("\n[Zbiorczy wykres CDF — F8 + F10]...")
    plot_cdf_comparison(
        np.array(all_raw), np.array(all_nn), np.array(all_nn_out),
        os.path.join(OUTPUT_DIR, "cdf_comparison_all.png"),
        title="– F8 + F10 łącznie"
    )

    # ====== Zapis dystrybuanty (z outlier elimination, dane dynamiczne) ======
    print("\n[Zapis dystrybuanty do xlsx]...")
    # ====== Zapis dystrybuant osobno dla każdej sali ======
    print("\n[Zapis dystrybuant do xlsx]...")
    for sala, (raw_err, nn_err, nn_err_out) in results.items():
        _, cdf_vals = compute_cdf(nn_err_out)
        df_cdf = pd.DataFrame({'dystrybuanta_bledu_NN': cdf_vals})
        cdf_path = os.path.join(OUTPUT_DIR, sala, f"dystrybuanta_NN_{sala}.xlsx")
        df_cdf.to_excel(cdf_path, index=False, engine='openpyxl')
        print(f"  Zapisano: {cdf_path}")
    # ====== Zbiorczy raport statystyczny ======
    print("\n[Zbiorowe statystyki błędów — F8 + F10]:")
    all_raw_a   = np.array(all_raw)
    all_nn_a    = np.array(all_nn)
    all_nn_out_a = np.array(all_nn_out)
    print(f"  Błąd surowy   – mediana: {np.median(all_raw_a):.1f} mm,"
          f" 75%: {np.percentile(all_raw_a, 75):.1f} mm")
    print(f"  NN (bez out.) – mediana: {np.median(all_nn_a):.1f} mm,"
          f" 75%: {np.percentile(all_nn_a, 75):.1f} mm")
    print(f"  NN (z out.)   – mediana: {np.median(all_nn_out_a):.1f} mm,"
          f" 75%: {np.percentile(all_nn_out_a, 75):.1f} mm")

    print(f"\nZakończono. Wyniki w katalogu: {OUTPUT_DIR}")
    return results


# ============================================================
# 9. URUCHOMIENIE
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        POMIARY_DIR = sys.argv[1]
    run_pipeline(POMIARY_DIR)