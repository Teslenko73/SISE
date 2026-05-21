"""
Korekcja pomiarów UWB przy użyciu sieci neuronowej
====================================================
Poprawki względem poprzedniej wersji:
  1. Cechy prędkości i przyspieszenia w oknie czasowym (lepsza generalizacja na dane dynamiczne)
  2. Predykcja bezpośrednich współrzędnych zamiast delt (stabilniejsze uczenie)
  3. Bezpieczne dopasowanie liczby cech (align_features z ostrzeżeniem)
  4. RobustScaler zamiast MinMaxScaler dla Y (odporność na zakresy spoza treningu)
  5. Wczesne zatrzymanie (early stopping) – zapobiega przeuczeniu
  6. Dropout w warstwach ukrytych (regularyzacja)
  7. Osobna diagnostyka jakości referencji dynamicznej
  8. Lepsza OutlierDetector: IQR + mediana krocząca (dwa kryteria łącznie)
  9. Selekcja cech: Usunięto zaszumione sygnały (>1.5σ), zostawiono stabilne (~0σ)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURACJA
# ============================================================
POMIARY_DIR       = "./pomiary"
OUTPUT_DIR        = "./wyniki"
EPOCHS            = 50
HIDDEN_LAYERS     = [256, 128, 64, 32]
LEARNING_RATE     = 0.0005
SEQUENCE_LEN      = 10
BATCH_SIZE        = 32
OUTLIER_THRESHOLD = 2.5      # próg z-score dla mediany kroczącej
IQR_FACTOR        = 1.5      # mnożnik IQR (drugie kryterium outlieru)
DROPOUT_RATE      = 0.1      # regularyzacja dropout
EARLY_STOP_PATIENCE = 30     # epoki bez poprawy → stop
RANDOM_SEED       = 42

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. WCZYTYWANIE DANYCH I FILTRACJA CECH
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
            print(f"    Wczytano: {os.path.basename(f)} ({len(df)} wierszy)")
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
    print(f"  [{sala}] Statyczne: {len(static_dfs)} plików, {len(df_static)} wierszy łącznie")

    dyn_dfs = []
    for pat in [f"{prefix}_[0-9]p.xlsx", f"{prefix}_[0-9]z.xlsx", f"{prefix}_random_*.xlsx"]:
        dyn_dfs.extend(load_xlsx_files(sala_dir, pattern=pat))

    if dyn_dfs:
        df_dynamic = pd.concat(dyn_dfs, ignore_index=True)
        print(f"  [{sala}] Dynamiczne: {len(dyn_dfs)} plików, {len(df_dynamic)} wierszy łącznie")
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
    print(f"    Pomiar X: '{coord_x}',  Pomiar Y: '{coord_y}'")
    print(f"    Ref.   X: '{ref_x}',  Ref.   Y: '{ref_y}'")
    return coord_x, coord_y, ref_x, ref_y


def get_feature_cols(df, coord_x, coord_y, ref_x, ref_y):
    """
    Selekcja cech na podstawie poziomu szumów (sigma).
    Zostawiamy stabilne sygnały (~0σ do 0.5σ), odrzucamy mocno zaszumione (>1.5σ).
    """
    ALLOWED_FEATURES = {
        'gyro__x', 'gyro__y', 'gyro__z',
        'quaternion__x', 'quaternion__w',
        'orientation__yaw'
    }

    # Dodatkowo jawnie zezwalamy na bazowe współrzędne wejściowe
    additional_allowed = {coord_x.lower().strip(), coord_y.lower().strip()}

    feature_cols = []
    for col in df.columns:
        col_clean = col.lower().strip()
        if col_clean in ALLOWED_FEATURES or col_clean in additional_allowed:
            feature_cols.append(col)

    print(f"  [Filtr cech] Zachowano {len(feature_cols)} stabilnych cech sygnałowych.")
    print(f"  Wybrane kolumny: {feature_cols}")
    return feature_cols


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
    """
    Dwuetapowy detektor outlierów:
      Krok 1 – IQR: próbki poza [Q1 - k*IQR, Q3 + k*IQR] dla błędu euklidesowego
               od mediany kroczącej są kandydatami na outliery.
      Krok 2 – Z-score: wyznaczamy z-score odległości od mediany kroczącej;
               próbki z z > threshold są potwierdzonymi outlierami.
    Próbka musi spełnić OBA kryteria, żeby zostać zastąpiona medianą.
    """
    def __init__(self, threshold=2.5, iqr_factor=1.5, window=15):
        self.threshold  = threshold
        self.iqr_factor = iqr_factor
        self.window     = window
        self.mean_dist  = None
        self.std_dist   = None
        self.q1         = None
        self.q3         = None

    def _rolling_median(self, arr):
        return pd.Series(arr).rolling(self.window, center=True, min_periods=1).median().values

    def fit(self, x_meas, y_meas):
        median_x  = self._rolling_median(x_meas)
        median_y  = self._rolling_median(y_meas)
        dist      = np.sqrt((x_meas - median_x)**2 + (y_meas - median_y)**2)
        self.mean_dist = np.mean(dist)
        self.std_dist  = np.std(dist) + 1e-9
        self.q1 = np.percentile(dist, 25)
        self.q3 = np.percentile(dist, 75)
        return self

    def clean(self, x_meas, y_meas):
        median_x = self._rolling_median(x_meas)
        median_y = self._rolling_median(y_meas)
        dist     = np.sqrt((x_meas - median_x)**2 + (y_meas - median_y)**2)

        z_score = (dist - self.mean_dist) / self.std_dist
        crit1   = z_score > self.threshold

        iqr   = self.q3 - self.q1
        upper = self.q3 + self.iqr_factor * iqr
        crit2 = dist > upper

        is_outlier = crit1 & crit2
        mask       = ~is_outlier

        n_removed = np.sum(is_outlier)
        print(f"    OutlierDetector: usunięto {n_removed} próbek "
              f"({100*n_removed/len(mask):.1f}%) | "
              f"z-score>{self.threshold} AND IQR>{self.iqr_factor}×IQR")

        x_clean, y_clean = x_meas.copy(), y_meas.copy()
        x_clean[is_outlier] = median_x[is_outlier]
        y_clean[is_outlier] = median_y[is_outlier]
        return x_clean, y_clean, mask


# ============================================================
# 3. BUDOWANIE MACIERZY CECH (time-window + dynamiczne cechy)
# ============================================================

def build_feature_matrix(df, feature_cols, coord_x, coord_y, ref_x, ref_y,
                          seq_len=10, outlier_detector=None):
    X_raw = df[feature_cols].values.astype(float)
    cx    = df[coord_x].values.astype(float)
    cy    = df[coord_y].values.astype(float)
    rx    = df[ref_x].values.astype(float)
    ry    = df[ref_y].values.astype(float)

    if outlier_detector is not None:
        cx, cy, _ = outlier_detector.clean(cx, cy)

    # Prędkości i przyspieszenia mierzonych pozycji
    vx = np.diff(cx, prepend=cx[0])
    vy = np.diff(cy, prepend=cy[0])
    ax = np.diff(vx, prepend=vx[0])
    ay = np.diff(vy, prepend=vy[0])

    features, targets, raw_errors = [], [], []
    for i in range(seq_len - 1, len(X_raw)):
        sl = slice(i - seq_len + 1, i + 1)
        window_feats = X_raw[sl].flatten()
        window_cx    = cx[sl]
        window_cy    = cy[sl]
        window_vx    = vx[sl]
        window_vy    = vy[sl]
        window_ax    = ax[sl]
        window_ay    = ay[sl]

        feat = np.concatenate([
            window_feats,
            window_cx, window_cy,
            window_vx, window_vy,
            window_ax, window_ay,
        ])
        features.append(feat)
        targets.append([rx[i], ry[i]])
        raw_errors.append(np.sqrt((cx[i] - rx[i])**2 + (cy[i] - ry[i])**2))

    return (np.array(features),
            np.array(targets),
            np.array(raw_errors),
            cx[seq_len-1:], cy[seq_len-1:],
            rx[seq_len-1:], ry[seq_len-1:])


def align_features(X, n_target, label=""):
    """Bezpieczne dopasowanie liczby kolumn z ostrzeżeniem."""
    actual = X.shape[1]
    if actual == n_target:
        return X
    if actual < n_target:
        diff = n_target - actual
        print(f"  UWAGA [{label}]: dopełniam {diff} brakujących cech zerami "
              f"({actual} → {n_target}). Sprawdź spójność kolumn!")
        return np.hstack([X, np.zeros((X.shape[0], diff))])
    else:
        diff = actual - n_target
        print(f"  UWAGA [{label}]: przycinam {diff} nadmiarowych cech "
              f"({actual} → {n_target}). Sprawdź spójność kolumn!")
        return X[:, :n_target]


# ============================================================
# 4. SIEĆ NEURONOWA (NumPy)
# ============================================================

def relu(x):         return np.maximum(0, x)
def relu_deriv(x):   return (x > 0).astype(float)
def linear(x):       return x
def linear_deriv(x): return np.ones_like(x)


class DenseLayer:
    def __init__(self, n_in, n_out, activation='relu', dropout_rate=0.0, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        self.activation_name = activation
        self.act   = relu   if activation == 'relu' else linear
        self.act_d = relu_deriv if activation == 'relu' else linear_deriv
        self.dropout_rate = dropout_rate
        self.z = self.a = self.dW = self.db = self.input = None
        self._dropout_mask = None

    def forward(self, x, training=True):
        self.input = x
        self.z     = x @ self.W + self.b
        self.a     = self.act(self.z)
        if training and self.dropout_rate > 0:
            self._dropout_mask = (np.random.rand(*self.a.shape) > self.dropout_rate).astype(float)
            self.a *= self._dropout_mask / (1.0 - self.dropout_rate + 1e-9)
        else:
            self._dropout_mask = None
        return self.a

    def backward(self, delta):
        if self._dropout_mask is not None:
            delta = delta * self._dropout_mask / (1.0 - self.dropout_rate + 1e-9)
        d       = delta * self.act_d(self.z)
        self.dW = self.input.T @ d
        self.db = d.sum(axis=0, keepdims=True)
        return d @ self.W.T


class NeuralNetwork:
    def __init__(self, n_features, hidden_layers, lr=0.001, momentum=0.9, dropout_rate=0.1):
        self.lr       = lr
        self.momentum = momentum
        sizes       = [n_features] + hidden_layers + [2]
        activations = ['relu'] * len(hidden_layers) + ['linear']
        dropouts    = [dropout_rate] * len(hidden_layers) + [0.0]
        self.layers = [
            DenseLayer(sizes[i], sizes[i+1], activations[i], dropouts[i], seed=RANDOM_SEED + i)
            for i in range(len(sizes) - 1)
        ]
        self.vW = [np.zeros_like(l.W) for l in self.layers]
        self.vb = [np.zeros_like(l.b) for l in self.layers]

    def forward(self, x, training=True):
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def predict(self, x):
        return self.forward(x, training=False)

    def backward(self, y_pred, y_true):
        delta = 2 * (y_pred - y_true) / len(y_true)
        for i, layer in reversed(list(enumerate(self.layers))):
            delta = layer.backward(delta)
        for i, layer in enumerate(self.layers):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * layer.dW
            self.vb[i] = self.momentum * self.vb[i] - self.lr * layer.db
            layer.W   += self.vW[i]
            layer.b   += self.vb[i]

    def fit(self, X_train, Y_train, X_val, Y_val, epochs=100, batch_size=32, patience=30):
        n = len(X_train)
        train_losses, val_losses = [], []
        best_val   = np.inf
        best_epoch = 0
        best_weights = [(l.W.copy(), l.b.copy()) for l in self.layers]
        no_improve   = 0

        for epoch in range(epochs):
            idx    = np.random.permutation(n)
            X_shuf = X_train[idx]
            Y_shuf = Y_train[idx]
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                Xb = X_shuf[start:start + batch_size]
                Yb = Y_shuf[start:start + batch_size]
                pred       = self.forward(Xb, training=True)
                epoch_loss += np.mean((pred - Yb)**2) * len(Xb)
                self.backward(pred, Yb)
            train_loss = epoch_loss / n
            val_loss   = np.mean((self.predict(X_val) - Y_val)**2)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val - 1e-7:
                best_val     = val_loss
                best_epoch   = epoch + 1
                best_weights = [(l.W.copy(), l.b.copy()) for l in self.layers]
                no_improve   = 0
            else:
                no_improve += 1

            if (epoch + 1) % 20 == 0:
                print(f"    Epoka {epoch+1:4d}/{epochs} | "
                      f"Train MSE: {train_loss:10.4f} | "
                      f"Val MSE: {val_loss:10.4f} | "
                      f"Best epoch: {best_epoch}")

            if no_improve >= patience:
                print(f"    Early stopping na epoce {epoch+1} "
                      f"(brak poprawy przez {patience} epok). "
                      f"Najlepsza epoka: {best_epoch}")
                break

        for layer, (W, b) in zip(self.layers, best_weights):
            layer.W = W
            layer.b = b
        print(f"    Przywrócono wagi z epoki {best_epoch} (val MSE={best_val:.4f})")
        return train_losses, val_losses

    def get_weights_info(self):
        rows = []
        for i, l in enumerate(self.layers):
            rows.append({
                'Warstwa':     i + 1,
                'n_wejść':     l.W.shape[0],
                'n_neuronów':  l.W.shape[1],
                'Aktywacja':   l.activation_name,
                'Dropout':     l.dropout_rate,
                'W_mean':      float(np.mean(l.W)),
                'W_std':       float(np.std(l.W)),
                'W_min':       float(np.min(l.W)),
                'W_max':       float(np.max(l.W)),
            })
        return rows


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
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_losses, label='Train MSE', color='steelblue')
    ax.plot(val_losses,   label='Walidacja MSE', color='orange')
    ax.set_xlabel('Epoka')
    ax.set_ylabel('MSE')
    ax.set_title(f'Krzywa uczenia {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_cdf_comparison(raw_errors, nn_errors, nn_outlier_errors, save_path, title=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    e1, c1 = compute_cdf(raw_errors)
    ax.plot(e1, c1, 'b-',  linewidth=2.5, label='Dane surowe UWB')
    e2, c2 = compute_cdf(nn_errors)
    ax.plot(e2, c2, 'g--', linewidth=2.5, label='Po korekcji NN (bez eliminacji outlierów)')
    if nn_outlier_errors is not None:
        e3, c3 = compute_cdf(nn_outlier_errors)
        ax.plot(e3, c3, 'r-.', linewidth=2.5, label='Po korekcji NN (z eliminacją outlierów)')

    for p, ls in [(0.5, ':'), (0.75, '--'), (0.9, '-.')]:
        ax.axhline(p, color='gray', linewidth=0.8, linestyle=ls, alpha=0.5)

    ax.set_xlabel('Błąd euklidesowy [mm]', fontsize=12)
    ax.set_ylabel('Dystrybuanta F(e)', fontsize=12)
    ax.set_title(f'Porównanie dystrybuant błędu lokalizacji UWB {title}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_trajectory(cx, cy, rx, ry, pred_x, pred_y, save_path, title="Trajektoria"):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(rx, ry, 'k-',  linewidth=2,             label='Referencja (prawda)')
    ax.plot(cx, cy, 'b.',  markersize=3, alpha=0.4, label='Pomiar UWB (surowy)')
    ax.plot(pred_x, pred_y, 'r.', markersize=3, alpha=0.5, label='Po korekcji NN')
    ax.set_xlabel('X [mm]', fontsize=11)
    ax.set_ylabel('Y [mm]', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_error_vs_time(raw_errors, nn_errors, nn_out_errors, save_path, title=""):
    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(len(raw_errors))
    ax.plot(t, raw_errors,    alpha=0.6, label='Surowy',         color='blue')
    ax.plot(t, nn_errors,     alpha=0.7, label='NN bez outlier', color='green')
    ax.plot(t, nn_out_errors, alpha=0.7, label='NN z outlier',   color='red')
    ax.set_xlabel('Próbka')
    ax.set_ylabel('Błąd [mm]')
    ax.set_title(f'Błąd w czasie {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


# ============================================================
# 7. POTOK DLA JEDNEJ SALI
# ============================================================

def run_sala_pipeline(sala, df_static, df_dynamic):
    print(f"\n{'='*65}")
    print(f"  SALA: {sala}")
    print(f"{'='*65}")
    sala_out = os.path.join(OUTPUT_DIR, sala)
    os.makedirs(sala_out, exist_ok=True)

    coord_x, coord_y, ref_x, ref_y = detect_column_names(df_static)
    feature_cols = get_feature_cols(df_static, coord_x, coord_y, ref_x, ref_y)

    use_cols = list(dict.fromkeys(feature_cols + [coord_x, coord_y, ref_x, ref_y]))
    df_stat  = prepare_df(df_static[use_cols].copy(), feature_cols, coord_x, coord_y, ref_x, ref_y)

    print(f"  Próbki statyczne (trening+walidacja): {len(df_stat)}")
    if len(df_stat) == 0:
        print("  BŁĄD: Brak danych po czyszczeniu. Pomijam salę.")
        return None, None, None

    od = OutlierDetector(threshold=OUTLIER_THRESHOLD, iqr_factor=IQR_FACTOR, window=15)
    od.fit(df_stat[coord_x].values.astype(float), df_stat[coord_y].values.astype(float))

    print(f"  Budowanie cech (seq_len={SEQUENCE_LEN}, + prędkość + przyspieszenie)...")

    (X_raw, Y_raw, raw_err_stat,
     cx_raw, cy_raw, rx_arr, ry_arr) = build_feature_matrix(
        df_stat, feature_cols, coord_x, coord_y, ref_x, ref_y,
        seq_len=SEQUENCE_LEN, outlier_detector=None)

    (X_out, Y_out, _,
     cx_out, cy_out, _, _) = build_feature_matrix(
        df_stat, feature_cols, coord_x, coord_y, ref_x, ref_y,
        seq_len=SEQUENCE_LEN, outlier_detector=od)

    scaler_X     = StandardScaler()
    scaler_Y     = RobustScaler()
    scaler_Y_out = RobustScaler()

    X_raw_sc = scaler_X.fit_transform(X_raw)
    X_out_sc = scaler_X.transform(X_out)
    Y_raw_sc = scaler_Y.fit_transform(Y_raw)
    Y_out_sc = scaler_Y_out.fit_transform(Y_out)

    split = int(0.7 * len(X_raw_sc))
    X_train,   X_val   = X_raw_sc[:split], X_raw_sc[split:]
    Y_train,   Y_val   = Y_raw_sc[:split], Y_raw_sc[split:]
    X_train_o, X_val_o = X_out_sc[:split], X_out_sc[split:]
    Y_train_o, Y_val_o = Y_out_sc[:split], Y_out_sc[split:]

    n_features = X_train.shape[1]
    print(f"  Cechy wejściowe: {n_features} | Train: {len(X_train)} | Val: {len(X_val)}")

    # ====== SIEĆ 1: bez eliminacji outlierów ======
    print(f"\n  [Sieć 1 – {sala}] Trening (BEZ eliminacji outlierów)...")
    nn = NeuralNetwork(n_features, HIDDEN_LAYERS, lr=LEARNING_RATE, dropout_rate=DROPOUT_RATE)
    tl, vl = nn.fit(X_train, Y_train, X_val, Y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=EARLY_STOP_PATIENCE)
    plot_training_curve(tl, vl, os.path.join(sala_out, "training_curve_no_outlier.png"), title=f"{sala} – bez outlier")

    # ====== SIEĆ 2: z eliminacją outlierów ======
    print(f"\n  [Sieć 2 – {sala}] Trening (Z eliminacją outlierów)...")
    nn_out = NeuralNetwork(n_features, HIDDEN_LAYERS, lr=LEARNING_RATE, dropout_rate=DROPOUT_RATE)
    tl2, vl2 = nn_out.fit(X_train_o, Y_train_o, X_val_o, Y_val_o, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=EARLY_STOP_PATIENCE)
    plot_training_curve(tl2, vl2, os.path.join(sala_out, "training_curve_with_outlier.png"), title=f"{sala} – z outlier")

    weights_info = nn_out.get_weights_info()
    df_weights   = pd.DataFrame(weights_info)
    w_path       = os.path.join(sala_out, f"wagi_sieci_{sala}.xlsx")
    df_weights.to_excel(w_path, index=False, engine='openpyxl')
    print(f"  Zapisano wagi: {w_path}")

    # ====== DANE TESTOWE ======
    if df_dynamic is not None and len(df_dynamic) > 0:
        avail_cols    = [c for c in use_cols if c in df_dynamic.columns]
        feat_cols_dyn = [c for c in feature_cols if c in df_dynamic.columns]
        df_dyn = prepare_df(df_dynamic[avail_cols].copy(), feat_cols_dyn, coord_x, coord_y, ref_x, ref_y)
        print(f"\n  Próbki dynamiczne (test): {len(df_dyn)}")

        print("  DIAGNOSTYKA danych dynamicznych:")
        print(df_dyn[[coord_x, coord_y, ref_x, ref_y]].describe().round(1))
        ref_changes = df_dyn[ref_x].diff().abs().sum() + df_dyn[ref_y].diff().abs().sum()
        print(f"  Łączna zmiana referencji (ref_x+ref_y): {ref_changes:.0f} mm "
              f"({'dynamiczna ✓' if ref_changes > 1000 else 'statyczna?'})")

        if len(df_dyn) >= SEQUENCE_LEN and len(feat_cols_dyn) > 0:
            (X_test, _, raw_errors_test,
             cx_test, cy_test, rx_test, ry_test) = build_feature_matrix(
                df_dyn, feat_cols_dyn, coord_x, coord_y, ref_x, ref_y,
                seq_len=SEQUENCE_LEN, outlier_detector=None)

            (X_test_o, _, _,
             cx_test_o, cy_test_o, _, _) = build_feature_matrix(
                df_dyn, feat_cols_dyn, coord_x, coord_y, ref_x, ref_y,
                seq_len=SEQUENCE_LEN, outlier_detector=od)

            X_test   = align_features(X_test,   X_raw.shape[1], label="test")
            X_test_o = align_features(X_test_o, X_raw.shape[1], label="test+outlier")
            X_test_sc   = scaler_X.transform(X_test)
            X_test_sc_o = scaler_X.transform(X_test_o)
            test_source = "dynamiczne"
        else:
            print("  UWAGA: Za mało danych dynamicznych — fallback na dane walidacyjne.")
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

    pred_raw   = scaler_Y.inverse_transform(nn.predict(X_test_sc))
    pred_x     = pred_raw[:, 0]
    pred_y     = pred_raw[:, 1]
    nn_errors  = np.sqrt((pred_x - rx_test)**2 + (pred_y - ry_test)**2)

    pred_raw_o  = scaler_Y_out.inverse_transform(nn_out.predict(X_test_sc_o))
    pred_x_o    = pred_raw_o[:, 0]
    pred_y_o    = pred_raw_o[:, 1]
    nn_errors_o = np.sqrt((pred_x_o - rx_test)**2 + (pred_y_o - ry_test)**2)

    df_pred = pd.DataFrame({
        'pred_x':       pred_x_o,
        'pred_y':       pred_y_o,
        'ref_x':        rx_test,
        'ref_y':        ry_test,
        'blad_nn_mm':   nn_errors_o,
        'blad_surowy_mm': raw_errors_test,
    })
    pred_path = os.path.join(sala_out, f"poprawione_xy_{sala}.xlsx")
    df_pred.to_excel(pred_path, index=False, engine='openpyxl')
    print(f"  Zapisano wyniki: {pred_path}")

    print(f"\n  Generowanie wykresów dla sali {sala}...")
    plot_cdf_comparison(raw_errors_test, nn_errors, nn_errors_o, os.path.join(sala_out, "cdf_comparison.png"), title=f"– {sala} ({test_source})")
    plot_trajectory(cx_test, cy_test, rx_test, ry_test, pred_x, pred_y, os.path.join(sala_out, "trajectory_no_outlier.png"), f"Trajektoria {sala} – NN bez outlier ({test_source})")
    plot_trajectory(cx_test_o, cy_test_o, rx_test, ry_test, pred_x_o, pred_y_o, os.path.join(sala_out, "trajectory_with_outlier.png"), f"Trajektoria {sala} – NN z outlier ({test_source})")
    plot_error_vs_time(raw_errors_test, nn_errors, nn_errors_o, os.path.join(sala_out, "error_vs_time.png"), title=f"– {sala} ({test_source})")

    def stats(err):
        return (f"mediana={np.median(err):.1f} mm | "
                f"75%={np.percentile(err,75):.1f} mm | "
                f"90%={np.percentile(err,90):.1f} mm | "
                f"śred.={np.mean(err):.1f} mm")

    print(f"\n  Statystyki błędów [{sala}] ({test_source}):")
    print(f"    Surowy UWB      : {stats(raw_errors_test)}")
    print(f"    NN bez outlier  : {stats(nn_errors)}")
    print(f"    NN z outlier    : {stats(nn_errors_o)}")

    imp_med = (np.median(raw_errors_test) - np.median(nn_errors_o)) / np.median(raw_errors_test) * 100
    imp_90  = (np.percentile(raw_errors_test, 90) - np.percentile(nn_errors_o, 90)) / np.percentile(raw_errors_test, 90) * 100
    print(f"    Poprawa mediany : {imp_med:+.1f}%")
    print(f"    Poprawa 90%     : {imp_90:+.1f}%")

    return raw_errors_test, nn_errors, nn_errors_o


# ============================================================
# 8. GŁÓWNY POTOK SCRIPTU
# ============================================================

def run_pipeline(pomiary_dir):
    print("=" * 65)
    print("  Korekcja pomiarów UWB – Sieć Neuronowa (poprawiona)")
    print("  Architektura: 256→128→64→32→2 (ReLU+Dropout / Linear)")
    print("  Trening: dane statyczne  |  Test: dane dynamiczne")
    print("  Poprawki: cechy prędkości, RobustScaler, Early Stopping,")
    print("            Dropout, OutlierDetector (mediana + IQR),")
    print("            Zoptymalizowana selekcja cech (niska sigma)")
    print("=" * 65)

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
        print("\nBłąd: brak wyników.")
        return

    plot_cdf_comparison(
        np.array(all_raw), np.array(all_nn), np.array(all_nn_out),
        os.path.join(OUTPUT_DIR, "cdf_comparison_all.png"),
        title="– F8 + F10 łącznie")

    for sala, (raw_err, nn_err, nn_err_out) in results.items():
        sorted_err, cdf_vals = compute_cdf(nn_err_out)
        df_cdf = pd.DataFrame({
            'dystrybuanta_bledu_NN': cdf_vals,
            'blad_mm':               sorted_err,
        })
        cdf_path = os.path.join(OUTPUT_DIR, sala, f"dystrybuanta_NN_{sala}.xlsx")
        df_cdf.to_excel(cdf_path, index=False, engine='openpyxl')
        print(f"  Zapisano dystrybuantę: {cdf_path}")

    print(f"\n{'='*65}")
    print("  PODSUMOWANIE KOŃCOWE")
    print(f"{'='*65}")
    for sala, (raw_err, nn_err, nn_err_out) in results.items():
        print(f"  {sala}:")
        print(f"    Surowy   : mediana={np.median(raw_err):.1f} mm, 90%={np.percentile(raw_err, 90):.1f} mm")
        print(f"    NN + Out : mediana={np.median(nn_err_out):.1f} mm, 90%={np.percentile(nn_err_out, 90):.1f} mm")
        print("-" * 40)


if __name__ == "__main__":
    # Możesz zmienić ścieżkę do katalogu, jeśli Twoje pliki są w innym miejscu
    run_pipeline(POMIARY_DIR)