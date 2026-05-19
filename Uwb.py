"""
Korekcja pomiarów UWB przy użyciu sieci neuronowej
Badania nad poprawą lokalizacji UWB przy pomocy sieci neuronowych

ZMIANY:
  - Osobne sieci dla sal F8 i F10 (eliminacja błędu site-dependent)
  - Dane dynamiczne użyte do TESTOWANIA, statyczne do UCZENIA
  - POWRÓT: Sieć przewiduje delty (rx - cx, ry - cy) zamiast bezpośrednich współrzędnych
  - Naprawa skalowania Y: MinMaxScaler(-1,1) zamiast StandardScaler
    (eliminuje gigantyczne błędy predykcji na danych dynamicznych przy zachowaniu uczenia delt)
  - Osobne pliki dystrybuanty dla F8 i F10
  - Sekcja diagnostyczna danych dynamicznych zachowana do weryfikacji jakości referencji
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURACJA
# ============================================================
POMIARY_DIR      = "./pomiary"
OUTPUT_DIR       = "./wyniki"
EPOCHS        = 300
HIDDEN_LAYERS = [256, 128, 64, 32]
LEARNING_RATE = 0.0005
SEQUENCE_LEN  = 10
BATCH_SIZE       = 32
OUTLIER_THRESHOLD = 3.0
RANDOM_SEED      = 42

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. WCZYTYWANIE DANYCH
# ============================================================

def load_xlsx_files(data_dir, pattern="*.xlsx"):
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
    sala_dir = os.path.join(pomiary_dir, sala)
    prefix   = sala.lower()

    if not os.path.isdir(sala_dir):
        raise FileNotFoundError(f"Katalog {sala_dir} nie istnieje.")

    static_dfs = load_xlsx_files(sala_dir, pattern=f"{prefix}_stat_*.xlsx")
    if not static_dfs:
        raise FileNotFoundError(f"Brak plików statycznych w {sala_dir}")
    df_static = pd.concat(static_dfs, ignore_index=True)
    print(f"  [{sala}] Statyczne: {len(static_dfs)} plików, {len(df_static)} wierszy")

    dyn_dfs = []
    for pat in [f"{prefix}_[0-9]p.xlsx", f"{prefix}_[0-9]z.xlsx", f"{prefix}_random_*.xlsx"]:
        dyn_dfs.extend(load_xlsx_files(sala_dir, pattern=pat))

    if dyn_dfs:
        df_dynamic = pd.concat(dyn_dfs, ignore_index=True)
        print(f"  [{sala}] Dynamiczne: {len(dyn_dfs)} plików, {len(df_dynamic)} wierszy")
    else:
        print(f"  [{sala}] UWAGA: Brak plików dynamicznych — test na danych statycznych.")
        df_dynamic = None

    return df_static, df_dynamic


def detect_column_names(df):
    cols     = [c.lower().strip() for c in df.columns]
    original = df.columns.tolist()
    coord_x = coord_y = ref_x = ref_y = None
    for i, c in enumerate(cols):
        if 'coordinate' in c and 'x' in c:   coord_x = original[i]
        elif 'coordinate' in c and 'y' in c:  coord_y = original[i]
        elif 'reference' in c and 'x' in c:   ref_x   = original[i]
        elif 'reference' in c and 'y' in c:   ref_y   = original[i]
    if coord_x is None:
        coord_x, coord_y, ref_x, ref_y = original[-4], original[-3], original[-2], original[-1]
    print(f"  Wykryte kolumny:")
    print(f"    Pomiar X: '{coord_x}', Pomiar Y: '{coord_y}'")
    print(f"    Referencja X: '{ref_x}', Referencja Y: '{ref_y}'")
    return coord_x, coord_y, ref_x, ref_y


def get_feature_cols(df, coord_x, coord_y, ref_x, ref_y):
    META_COLS = {
        'unnamed: 0.1', 'unnamed: 0', 'version', 'alive', 'tagid',
        'success', 'timestamp', 'data__anchordata', 'errorcode',
        'data__coordinates__z', '_source_file',
        ref_x.lower(), ref_y.lower(),
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c.lower() not in META_COLS]


def prepare_df(df, feature_cols, coord_x, coord_y, ref_x, ref_y):
    for col in [ref_x, ref_y]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    valid_feat = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=valid_feat + [coord_x, coord_y])
    df = df.dropna(subset=[ref_x, ref_y])
    return df.reset_index(drop=True)


# ============================================================
# 2. MECHANIZM ELIMINACJI BŁĘDNYCH POMIARÓW
# ============================================================

class OutlierDetector:
    def __init__(self, threshold=3.0, window=15):
        self.threshold = threshold
        self.window    = window

    def fit(self, x_meas, y_meas):
        self.median_x  = pd.Series(x_meas).rolling(self.window, center=True, min_periods=1).median().values
        self.median_y  = pd.Series(y_meas).rolling(self.window, center=True, min_periods=1).median().values
        dist           = np.sqrt((x_meas - self.median_x)**2 + (y_meas - self.median_y)**2)
        self.mean_dist = np.mean(dist)
        self.std_dist  = np.std(dist)
        return self

    def clean(self, x_meas, y_meas):
        median_x = pd.Series(x_meas).rolling(self.window, center=True, min_periods=1).median().values
        median_y = pd.Series(y_meas).rolling(self.window, center=True, min_periods=1).median().values
        dist     = np.sqrt((x_meas - median_x)**2 + (y_meas - median_y)**2)
        z_score  = (dist - self.mean_dist) / (self.std_dist + 1e-9)
        mask     = z_score < self.threshold
        print(f"    Usunięto {np.sum(~mask)} outlierów ({100*np.sum(~mask)/len(mask):.1f}%)")
        x_clean, y_clean = x_meas.copy(), y_meas.copy()
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

    features, targets_x, targets_y, raw_errors = [], [], [], []
    for i in range(seq_len - 1, len(X_raw)):
        window_feats = X_raw[i - seq_len + 1 : i + 1].flatten()
        window_cx    = cx[i - seq_len + 1 : i + 1]
        window_cy    = cy[i - seq_len + 1 : i + 1]
        features.append(np.concatenate([window_feats, window_cx, window_cy]))

        # POWRÓT: Ponownie przewidujemy deltę (błąd) pomiaru względem prawdy
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
# 4. SIEĆ NEURONOWA (NumPy, bez zewnętrznych frameworków ML)
# ============================================================

def relu(x):         return np.maximum(0, x)
def relu_deriv(x):   return (x > 0).astype(float)
def linear(x):       return x
def linear_deriv(x): return np.ones_like(x)


class DenseLayer:
    def __init__(self, n_in, n_out, activation='relu', seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        self.activation_name = activation
        self.act   = relu   if activation == 'relu' else linear
        self.act_d = relu_deriv if activation == 'relu' else linear_deriv
        self.z = self.a = self.dW = self.db = self.input = None

    def forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b
        self.a = self.act(self.z)
        return self.a

    def backward(self, delta):
        d       = delta * self.act_d(self.z)
        self.dW = self.input.T @ d
        self.db = d.sum(axis=0, keepdims=True)
        return d @ self.W.T


class NeuralNetwork:
    def __init__(self, n_features, hidden_layers, lr=0.001, momentum=0.9):
        self.lr       = lr
        self.momentum = momentum
        sizes       = [n_features] + hidden_layers + [2]
        activations = ['relu'] * len(hidden_layers) + ['linear']
        self.layers = [DenseLayer(sizes[i], sizes[i+1], activations[i], seed=RANDOM_SEED+i)
                       for i in range(len(sizes) - 1)]
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
            layer.W   += self.vW[i]
            layer.b   += self.vb[i]

    def fit(self, X_train, Y_train, X_val, Y_val, epochs=100, batch_size=32):
        n = len(X_train)
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            idx    = np.random.permutation(n)
            X_shuf = X_train[idx]
            Y_shuf = Y_train[idx]
            epoch_loss = 0
            for start in range(0, n, batch_size):
                Xb, Yb = X_shuf[start:start+batch_size], Y_shuf[start:start+batch_size]
                pred   = self.forward(Xb)
                epoch_loss += np.mean((pred - Yb)**2) * len(Xb)
                self.backward(pred, Yb)
            train_loss = epoch_loss / n
            val_loss   = np.mean((self.forward(X_val) - Y_val)**2)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if (epoch + 1) % 10 == 0:
                print(f"    Epoka {epoch+1:3d}/{epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        return train_losses, val_losses

    def predict(self, X):
        return self.forward(X)

    def get_weights_info(self):
        return [{
            'warstwa':   i + 1,
            'n_in':      l.W.shape[0],
            'n_out':     l.W.shape[1],
            'aktywacja': l.activation_name,
            'W_mean':    np.mean(l.W),
            'W_std':     np.std(l.W),
            'W_min':     np.min(l.W),
            'W_max':     np.max(l.W),
        } for i, l in enumerate(self.layers)]


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
    ax.plot(val_losses,   label='Walidacja MSE')
    ax.set_xlabel('Epoka'); ax.set_ylabel('MSE')
    ax.set_title(f'Krzywa uczenia {title}')
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
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
    ax.set_xlabel('Błąd euklidesowy [mm]'); ax.set_ylabel('Dystrybuanta F(e)')
    ax.set_title(f'Porównanie dystrybuant błędu lokalizacji UWB {title}')
    ax.legend(); ax.grid(True, alpha=0.4)
    ax.set_xlim(left=0); ax.set_ylim([0, 1])
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Zapisano: {save_path}")


def plot_trajectory(cx, cy, rx, ry, pred_x, pred_y, save_path, title="Trajektoria"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rx, ry, 'k-',  linewidth=2,             label='Referencja')
    ax.plot(cx, cy, 'b.',  markersize=3, alpha=0.5, label='Pomiar UWB')
    ax.plot(pred_x, pred_y, 'r.', markersize=3, alpha=0.5, label='Po korekcji NN')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Zapisano: {save_path}")


# ============================================================
# 7. POTOK DLA JEDNEJ SALI
# ============================================================

def run_sala_pipeline(sala, df_static, df_dynamic):
    print(f"\n{'='*60}")
    print(f"  Sala: {sala}")
    print(f"{'='*60}")
    sala_out = os.path.join(OUTPUT_DIR, sala)
    os.makedirs(sala_out, exist_ok=True)

    coord_x, coord_y, ref_x, ref_y = detect_column_names(df_static)
    feature_cols = get_feature_cols(df_static, coord_x, coord_y, ref_x, ref_y)

    use_cols = list(dict.fromkeys(feature_cols + [coord_x, coord_y, ref_x, ref_y]))
    df_stat  = prepare_df(df_static[use_cols].copy(), feature_cols, coord_x, coord_y, ref_x, ref_y)

    print(f"  Próbki statyczne (trening): {len(df_stat)}")

    if len(df_stat) == 0:
        print("  BŁĄD: Brak danych po czyszczeniu. Pomijam salę.")
        return None, None, None

    print(f"  Zakresy: cx={df_stat[coord_x].min():.0f}–{df_stat[coord_x].max():.0f} mm, "
          f"ref_x={df_stat[ref_x].min():.0f}–{df_stat[ref_x].max():.0f} mm")

    # --- Outlier detector ---
    od = OutlierDetector(threshold=OUTLIER_THRESHOLD, window=15)
    od.fit(df_stat[coord_x].values.astype(float),
           df_stat[coord_y].values.astype(float))

    # --- Buduj macierze cech ---
    print(f"  Budowanie cech (seq_len={SEQUENCE_LEN})...")
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

    # --- Normalizacja ---
    # MinMaxScaler zapobiega sztucznemu przeskalowaniu wartości delta na danych dynamicznych.
    scaler_X     = StandardScaler()
    scaler_Y     = MinMaxScaler(feature_range=(-1, 1))
    scaler_Y_out = MinMaxScaler(feature_range=(-1, 1))

    X_raw_sc = scaler_X.fit_transform(X_raw)
    X_out_sc = scaler_X.transform(X_out)
    Y_raw_sc = scaler_Y.fit_transform(Y_raw)
    Y_out_sc = scaler_Y_out.fit_transform(Y_out)

    # --- Podział train/val (70/30) ---
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
    plot_training_curve(tl, vl, os.path.join(sala_out, "training_curve_no_outlier.png"),
                        title=f"{sala} – bez outlier")

    # ====== SIEĆ 2: z eliminacją outlierów ======
    print(f"\n  [Sieć 2 – {sala}] Trening (z outlier elimination)...")
    nn_out = NeuralNetwork(n_features, HIDDEN_LAYERS, lr=LEARNING_RATE)
    tl2, vl2 = nn_out.fit(X_train_o, Y_train_o, X_val_o, Y_val_o, epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_training_curve(tl2, vl2, os.path.join(sala_out, "training_curve_with_outlier.png"),
                        title=f"{sala} – z outlier")

    # ====== DANE TESTOWE ======
    if df_dynamic is not None and len(df_dynamic) > 0:
        avail_cols    = [c for c in use_cols if c in df_dynamic.columns]
        feat_cols_dyn = [c for c in feature_cols if c in df_dynamic.columns]
        df_dyn = prepare_df(df_dynamic[avail_cols].copy(),
                            feat_cols_dyn, coord_x, coord_y, ref_x, ref_y)
        print(f"  Próbki dynamiczne (test): {len(df_dyn)}")

        # --- DIAGNOSTYKA DANYCH DYNAMICZNYCH ---
        print(f"\n  DIAGNOSTYKA danych dynamicznych:")
        print(df_dyn[[coord_x, coord_y, ref_x, ref_y]].describe())
        print(f"\n  Pierwsze 10 wierszy ref:")
        print(df_dyn[[ref_x, ref_y]].head(10))
        print("-" * 40)

        if len(df_dyn) >= SEQUENCE_LEN and len(feat_cols_dyn) > 0:
            (X_test_dyn, _, _, raw_errors_test,
             cx_test, cy_test, rx_test, ry_test) = build_feature_matrix(
                df_dyn, feat_cols_dyn, coord_x, coord_y, ref_x, ref_y,
                seq_len=SEQUENCE_LEN, outlier_detector=None)

            (X_test_dyn_o, _, _, _,
             cx_test_o, cy_test_o, _, _) = build_feature_matrix(
                df_dyn, feat_cols_dyn, coord_x, coord_y, ref_x, ref_y,
                seq_len=SEQUENCE_LEN, outlier_detector=od)

            def align_features(X, n_target):
                if X.shape[1] < n_target:
                    X = np.hstack([X, np.zeros((X.shape[0], n_target - X.shape[1]))])
                elif X.shape[1] > n_target:
                    X = X[:, :n_target]
                return X

            X_test_dyn   = align_features(X_test_dyn,   X_raw.shape[1])
            X_test_dyn_o = align_features(X_test_dyn_o, X_raw.shape[1])
            X_test_sc    = scaler_X.transform(X_test_dyn)
            X_test_sc_o  = scaler_X.transform(X_test_dyn_o)
            test_source  = "dynamiczne"
        else:
            print("  UWAGA: Za mało danych dynamicznych — test na walidacyjnych.")
            X_test_sc, X_test_sc_o = X_val, X_val_o
            raw_errors_test = raw_err_stat[split:]
            cx_test,  cy_test   = cx_raw[split:],  cy_raw[split:]
            rx_test,  ry_test   = rx_arr[split:],  ry_arr[split:]
            cx_test_o, cy_test_o = cx_out[split:], cy_out[split:]
            test_source = "statyczne-walidacja"
    else:
        X_test_sc, X_test_sc_o = X_val, X_val_o
        raw_errors_test = raw_err_stat[split:]
        cx_test,  cy_test   = cx_raw[split:],  cy_raw[split:]
        rx_test,  ry_test   = rx_arr[split:],  ry_arr[split:]
        cx_test_o, cy_test_o = cx_out[split:], cy_out[split:]
        test_source = "statyczne-walidacja"

    print(f"  Źródło danych testowych: {test_source}")

    # --- Predykcja ---
    # POWRÓT: Wynik sieci traktujemy jako poprawkę (deltę) i dodajemy ją do cx/cy
    pred_delta    = scaler_Y.inverse_transform(nn.predict(X_test_sc))
    pred_x        = cx_test + pred_delta[:, 0]
    pred_y        = cy_test + pred_delta[:, 1]
    nn_errors     = np.sqrt((pred_x - rx_test)**2 + (pred_y - ry_test)**2)

    pred_delta_o  = scaler_Y_out.inverse_transform(nn_out.predict(X_test_sc_o))
    pred_x_o      = cx_test_o + pred_delta_o[:, 0]
    pred_y_o      = cy_test_o + pred_delta_o[:, 1]
    nn_errors_out = np.sqrt((pred_x_o - rx_test)**2 + (pred_y_o - ry_test)**2)

    # --- Zapis poprawionych X, Y ---
    df_pred = pd.DataFrame({
        'pred_x':  pred_x_o,
        'pred_y':  pred_y_o,
        'ref_x':   rx_test,
        'ref_y':   ry_test,
        'blad_mm': nn_errors_out,
    })
    pred_path = os.path.join(sala_out, f"poprawione_xy_{sala}.xlsx")
    df_pred.to_excel(pred_path, index=False, engine='openpyxl')
    print(f"  Zapisano: {pred_path}")

    # --- Wykresy ---
    print(f"\n  Generowanie wykresów dla sali {sala}...")
    plot_cdf_comparison(
        raw_errors_test, nn_errors, nn_errors_out,
        os.path.join(sala_out, "cdf_comparison.png"),
        title=f"– {sala} ({test_source})")
    plot_trajectory(cx_test, cy_test, rx_test, ry_test, pred_x, pred_y,
                    os.path.join(sala_out, "trajectory_no_outlier.png"),
                    f"Trajektoria {sala} – NN bez outlier ({test_source})")
    plot_trajectory(cx_test_o, cy_test_o, rx_test, ry_test, pred_x_o, pred_y_o,
                    os.path.join(sala_out, "trajectory_with_outlier.png"),
                    f"Trajektoria {sala} – NN z outlier ({test_source})")

    # --- Statystyki ---
    print(f"\n  Statystyki błędów [{sala}] ({test_source}):")
    print(f"    Surowy      – mediana: {np.median(raw_errors_test):.1f} mm, "
          f"75%: {np.percentile(raw_errors_test, 75):.1f} mm")
    print(f"    NN bez out. – mediana: {np.median(nn_errors):.1f} mm, "
          f"75%: {np.percentile(nn_errors, 75):.1f} mm")
    print(f"    NN z out.   – mediana: {np.median(nn_errors_out):.1f} mm, "
          f"75%: {np.percentile(nn_errors_out, 75):.1f} mm")

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

    if not results:
        print("\nBłąd: brak wyników do zapisu.")
        return

    # --- Zbiorczy wykres CDF ---
    plot_cdf_comparison(
        np.array(all_raw), np.array(all_nn), np.array(all_nn_out),
        os.path.join(OUTPUT_DIR, "cdf_comparison_all.png"),
        title="– F8 + F10 łącznie")

    # --- Osobne pliki dystrybuant dla każdej sali ---
    for sala, (raw_err, nn_err, nn_err_out) in results.items():
        _, cdf_vals = compute_cdf(nn_err_out)
        df_cdf = pd.DataFrame({'dystrybuanta_bledu_NN': cdf_vals})
        cdf_path = os.path.join(OUTPUT_DIR, sala, f"dystrybuanta_NN_{sala}.xlsx")
        df_cdf.to_excel(cdf_path, index=False, engine='openpyxl')
        print(f"  Zapisano: {cdf_path}")

    print(f"\nZakończono. Wyniki w katalogu: {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        POMIARY_DIR = sys.argv[1]
    run_pipeline(POMIARY_DIR)