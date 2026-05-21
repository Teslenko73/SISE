"""
Korekcja pomiarów UWB przy użyciu sieci neuronowej
====================================================
Architektura: wejście → 256 → 128 → 64 → 32 → 2 (liniowa)
Aktywacje:    ReLU w warstwach ukrytych, liniowa na wyjściu
Regularyzacja: Dropout(0.1), Early Stopping

Kluczowe decyzje projektowe:
  1. TARGET = delta (rx-cx, ry-cy) — sieć uczy się błędu systemu UWB,
     nie bezwzględnej pozycji. Delty mają podobny rozkład w danych
     statycznych i dynamicznych (std ~150-220mm), co umożliwia generalizację.

  2. SELEKCJA CECH AUTOMATYCZNA — obliczamy shift rozkładu każdej cechy
     między danymi statycznymi a dynamicznymi (w jednostkach std).
     Zachowujemy tylko cechy z shift < MAX_FEATURE_SHIFT (domyślnie 1.0σ).
     Dzięki temu sieć nie dostaje cech które wyglądają zupełnie inaczej
     podczas testowania niż podczas treningu.

  3. OUTLIER DETECTOR (podwójne kryterium: z-score + IQR) — eliminuje
     ewidentnie błędne próbki UWB przed uczeniem i testowaniem.
     Porównanie dwóch sieci (z/bez) udowadnia skuteczność mechanizmu.

  4. StandardScaler dla X, StandardScaler dla Y (delt) — delty mają
     zbliżony zakres w obu zbiorach, więc standardowe skalowanie jest
     bezpieczne i nie eksploduje przy ekstrapolacji.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURACJA
# ============================================================
POMIARY_DIR         = "./pomiary"
OUTPUT_DIR          = "./wyniki"
EPOCHS              = 400
HIDDEN_LAYERS       = [256, 128, 64, 32]
LEARNING_RATE       = 0.0005
SEQUENCE_LEN        = 10
BATCH_SIZE          = 32
OUTLIER_THRESHOLD   = 2.5    # z-score próg dla OutlierDetector
IQR_FACTOR          = 1.5    # IQR mnożnik dla OutlierDetector
DROPOUT_RATE        = 0.1
EARLY_STOP_PATIENCE = 40
MAX_FEATURE_SHIFT   = 1.0    # max dopuszczalny shift rozkładu cechy [σ]
RANDOM_SEED         = 42

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
    for pat in [f"{prefix}_[0-9]p.xlsx", f"{prefix}_[0-9]z.xlsx",
                f"{prefix}_random_*.xlsx"]:
        dyn_dfs.extend(load_xlsx_files(sala_dir, pattern=pat))

    if dyn_dfs:
        df_dynamic = pd.concat(dyn_dfs, ignore_index=True)
        print(f"  [{sala}] Dynamiczne: {len(dyn_dfs)} plików, {len(df_dynamic)} wierszy")
    else:
        print(f"  [{sala}] UWAGA: Brak plików dynamicznych.")
        df_dynamic = None

    return df_static, df_dynamic


def detect_column_names(df):
    cols     = [c.lower().strip() for c in df.columns]
    original = df.columns.tolist()
    coord_x = coord_y = ref_x = ref_y = None
    for i, c in enumerate(cols):
        if 'coordinate' in c and 'x' in c:  coord_x = original[i]
        elif 'coordinate' in c and 'y' in c: coord_y = original[i]
        elif 'reference' in c and 'x' in c:  ref_x   = original[i]
        elif 'reference' in c and 'y' in c:  ref_y   = original[i]
    if coord_x is None:
        coord_x, coord_y, ref_x, ref_y = original[-4], original[-3], original[-2], original[-1]
    print(f"  Kolumny: cx='{coord_x}', cy='{coord_y}', rx='{ref_x}', ry='{ref_y}'")
    return coord_x, coord_y, ref_x, ref_y


def get_all_numeric_cols(df, coord_x, coord_y, ref_x, ref_y):
    """Zwraca wszystkie kolumny numeryczne z wyłączeniem metadanych."""
    META = {
        'unnamed: 0.1', 'unnamed: 0', 'version', 'alive', 'tagid',
        'success', 'timestamp', 'data__anchordata', 'errorcode',
        'data__coordinates__z', '_source_file',
        ref_x.lower(), ref_y.lower(),
    }
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c.lower() not in META]


def select_stable_features(df_stat, df_dyn, all_cols, max_shift=1.0):
    """
    Automatyczna selekcja cech na podstawie shiftu rozkładu.

    Dla każdej cechy obliczamy:
        shift = |mean_dyn - mean_stat| / std_stat

    Jeśli shift > max_shift, cecha jest odrzucana — jej rozkład
    w danych dynamicznych jest zbyt różny od treningowego (statycznego),
    co powoduje że sieć dostaje podczas testu wartości których nie widziała.

    Zawsze zachowujemy cx i cy (podstawowe współrzędne UWB).
    """
    stable = []
    report = []
    cols_in_both = [c for c in all_cols if c in df_dyn.columns]

    for c in cols_in_both:
        s_vals = df_stat[c].dropna().values
        d_vals = df_dyn[c].dropna().values
        if len(s_vals) == 0 or len(d_vals) == 0:
            continue
        mean_s = np.mean(s_vals)
        std_s  = np.std(s_vals) + 1e-9
        shift  = abs(np.mean(d_vals) - mean_s) / std_s
        report.append((c, shift))
        if shift <= max_shift:
            stable.append(c)

    # Posortuj raport po shifcie
    report.sort(key=lambda x: x[1])
    print(f"\n  Selekcja cech (próg shift ≤ {max_shift}σ):")
    for c, sh in report:
        flag = "✓" if sh <= max_shift else "✗ ODRZUCONO"
        print(f"    {c:45s} shift={sh:.2f}σ  {flag}")
    print(f"  Zachowano {len(stable)}/{len(report)} cech\n")

    return stable


def prepare_df(df, feature_cols, coord_x, coord_y, ref_x, ref_y):
    for col in [ref_x, ref_y]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    valid = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=valid + [coord_x, coord_y, ref_x, ref_y])
    return df.reset_index(drop=True)


# ============================================================
# 2. MECHANIZM ELIMINACJI BŁĘDNYCH POMIARÓW
#    Podwójne kryterium: z-score mediany kroczącej + IQR
#    Próbka musi spełnić OBA kryteria żeby zostać zastąpiona.
# ============================================================

class OutlierDetector:
    """
    Eliminuje ewidentnie błędne pomiary UWB.

    Działanie:
      1. Oblicz odległość każdej próbki od lokalnej mediany kroczącej
         (okno 15 próbek) — to eliminuje wpływ rzeczywistego ruchu.
      2. Kryterium 1 (z-score): odległość > threshold * std_dist
      3. Kryterium 2 (IQR):     odległość > Q3 + iqr_factor * IQR
      4. Outlier = spełnione OBA kryteria jednocześnie.
      5. Zastąp outliery wartością mediany kroczącej (nie usuwa próbek,
         tylko koryguje ich pozycję — zachowuje długość szeregu).

    Parametry uczone na danych statycznych (fit), stosowane na obu zbiorach.
    """
    def __init__(self, threshold=2.5, iqr_factor=1.5, window=15):
        self.threshold  = threshold
        self.iqr_factor = iqr_factor
        self.window     = window

    def _rolling_median(self, arr):
        return pd.Series(arr).rolling(
            self.window, center=True, min_periods=1).median().values

    def fit(self, cx, cy):
        mx = self._rolling_median(cx)
        my = self._rolling_median(cy)
        dist = np.sqrt((cx - mx)**2 + (cy - my)**2)
        self.mean_dist = np.mean(dist)
        self.std_dist  = np.std(dist) + 1e-9
        self.q1        = np.percentile(dist, 25)
        self.q3        = np.percentile(dist, 75)
        return self

    def clean(self, cx, cy):
        mx   = self._rolling_median(cx)
        my   = self._rolling_median(cy)
        dist = np.sqrt((cx - mx)**2 + (cy - my)**2)

        crit1 = (dist - self.mean_dist) / self.std_dist > self.threshold
        crit2 = dist > self.q3 + self.iqr_factor * (self.q3 - self.q1)
        bad   = crit1 & crit2

        cx_c, cy_c = cx.copy(), cy.copy()
        cx_c[bad] = mx[bad]
        cy_c[bad] = my[bad]

        print(f"    OutlierDetector: usunięto {bad.sum()} próbek "
              f"({100*bad.sum()/len(bad):.1f}%) "
              f"[z>{self.threshold}σ AND dist>Q3+{self.iqr_factor}×IQR]")
        return cx_c, cy_c, ~bad


# ============================================================
# 3. BUDOWANIE MACIERZY CECH
#    Cechy w oknie czasowym (seq_len próbek):
#      - sygnały UWB z okna (feature_cols × seq_len)
#      - pozycje cx, cy w oknie
#      - prędkości vx, vy w oknie (diff cx/cy)
#      - przyspieszenia ax, ay w oknie (diff vx/vy)
#
#    Target: delta = [rx - cx, ry - cy] dla bieżącej próbki
#    Wynik predykcji: pred_x = cx + delta_x
# ============================================================

def build_feature_matrix(df, feature_cols, coord_x, coord_y, ref_x, ref_y,
                          seq_len=10, outlier_detector=None):
    Xr = df[feature_cols].values.astype(float)
    cx = df[coord_x].values.astype(float)
    cy = df[coord_y].values.astype(float)
    rx = df[ref_x].values.astype(float)
    ry = df[ref_y].values.astype(float)

    if outlier_detector is not None:
        cx, cy, _ = outlier_detector.clean(cx, cy)

    vx = np.diff(cx, prepend=cx[0])
    vy = np.diff(cy, prepend=cy[0])
    ax = np.diff(vx, prepend=vx[0])
    ay = np.diff(vy, prepend=vy[0])

    features, deltas, raw_errors = [], [], []
    for i in range(seq_len - 1, len(Xr)):
        sl = slice(i - seq_len + 1, i + 1)
        feat = np.concatenate([
            Xr[sl].flatten(),
            cx[sl], cy[sl],
            vx[sl], vy[sl],
            ax[sl], ay[sl],
        ])
        features.append(feat)
        # TARGET: delta (ile sieć musi dodać do cx/cy żeby trafić w rx/ry)
        deltas.append([rx[i] - cx[i], ry[i] - cy[i]])
        raw_errors.append(np.sqrt((cx[i]-rx[i])**2 + (cy[i]-ry[i])**2))

    return (np.array(features),
            np.array(deltas),
            np.array(raw_errors),
            cx[seq_len-1:], cy[seq_len-1:],
            rx[seq_len-1:], ry[seq_len-1:])


def align_features(X, n_target, label=""):
    if X.shape[1] == n_target:
        return X
    if X.shape[1] < n_target:
        print(f"  UWAGA [{label}]: dopełniam {n_target-X.shape[1]} cech zerami")
        return np.hstack([X, np.zeros((X.shape[0], n_target - X.shape[1]))])
    print(f"  UWAGA [{label}]: przycinam {X.shape[1]-n_target} nadmiarowych cech")
    return X[:, :n_target]


# ============================================================
# 4. SIEĆ NEURONOWA (NumPy, bez zewnętrznych frameworków ML)
#
# Architektura:
#   Wejście: n_features neuronów
#   Warstwa 1: 256 neuronów, ReLU, Dropout(0.1)
#   Warstwa 2: 128 neuronów, ReLU, Dropout(0.1)
#   Warstwa 3:  64 neurony,  ReLU, Dropout(0.1)
#   Warstwa 4:  32 neurony,  ReLU, Dropout(0.1)
#   Wyjście:     2 neurony,  liniowa (dx, dy)
#
# Inicjalizacja: He (sqrt(2/n_in)) — optymalna dla ReLU
# Optymalizator: SGD z momentum (momentum=0.9)
# Regularyzacja: Dropout podczas treningu, wyłączony podczas predykcji
# Early Stopping: zatrzymuje gdy val_loss nie spada przez PATIENCE epok,
#                 przywraca wagi z najlepszej epoki
# ============================================================

def relu(x):       return np.maximum(0, x)
def relu_d(x):     return (x > 0).astype(float)
def linear(x):     return x
def linear_d(x):   return np.ones_like(x)


class DenseLayer:
    def __init__(self, n_in, n_out, activation='relu', dropout=0.0, seed=0):
        rng      = np.random.default_rng(seed)
        self.W   = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b   = np.zeros((1, n_out))
        self.act  = relu   if activation == 'relu' else linear
        self.dact = relu_d if activation == 'relu' else linear_d
        self.activation_name = activation
        self.dropout = dropout
        self.input = self.z = self.a = self.dW = self.db = self._mask = None

    def forward(self, x, training=True):
        self.input = x
        self.z     = x @ self.W + self.b
        self.a     = self.act(self.z)
        if training and self.dropout > 0:
            self._mask = (np.random.rand(*self.a.shape) > self.dropout).astype(float)
            self.a    *= self._mask / (1.0 - self.dropout + 1e-9)
        else:
            self._mask = None
        return self.a

    def backward(self, delta):
        if self._mask is not None:
            delta = delta * self._mask / (1.0 - self.dropout + 1e-9)
        d       = delta * self.dact(self.z)
        self.dW = self.input.T @ d
        self.db = d.sum(axis=0, keepdims=True)
        return d @ self.W.T


class NeuralNetwork:
    def __init__(self, n_in, hidden, lr=0.001, momentum=0.9, dropout=0.1):
        self.lr = lr
        self.momentum = momentum
        sizes = [n_in] + hidden + [2]
        acts  = ['relu'] * len(hidden) + ['linear']
        drops = [dropout] * len(hidden) + [0.0]
        self.layers = [
            DenseLayer(sizes[i], sizes[i+1], acts[i], drops[i],
                       seed=RANDOM_SEED + i)
            for i in range(len(sizes) - 1)
        ]
        self.vW = [np.zeros_like(l.W) for l in self.layers]
        self.vb = [np.zeros_like(l.b) for l in self.layers]

    def forward(self, x, training=True):
        for l in self.layers:
            x = l.forward(x, training)
        return x

    def predict(self, x):
        return self.forward(x, training=False)

    def _backward(self, y_pred, y_true):
        delta = 2 * (y_pred - y_true) / len(y_true)
        for l in reversed(self.layers):
            delta = l.backward(delta)
        for i, l in enumerate(self.layers):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * l.dW
            self.vb[i] = self.momentum * self.vb[i] - self.lr * l.db
            l.W += self.vW[i]
            l.b += self.vb[i]

    def fit(self, Xtr, Ytr, Xv, Yv,
            epochs=300, batch_size=32, patience=40):
        n = len(Xtr)
        tl, vl = [], []
        best_val   = np.inf
        best_ep    = 0
        best_W     = [(l.W.copy(), l.b.copy()) for l in self.layers]
        no_improve = 0

        for ep in range(epochs):
            idx = np.random.permutation(n)
            Xs, Ys = Xtr[idx], Ytr[idx]
            loss = 0.0
            for s in range(0, n, batch_size):
                Xb, Yb = Xs[s:s+batch_size], Ys[s:s+batch_size]
                p       = self.forward(Xb, training=True)
                loss   += np.mean((p - Yb)**2) * len(Xb)
                self._backward(p, Yb)
            train_loss = loss / n
            val_loss   = np.mean((self.predict(Xv) - Yv)**2)
            tl.append(train_loss)
            vl.append(val_loss)

            if val_loss < best_val - 1e-7:
                best_val   = val_loss
                best_ep    = ep + 1
                best_W     = [(l.W.copy(), l.b.copy()) for l in self.layers]
                no_improve = 0
            else:
                no_improve += 1

            if (ep + 1) % 50 == 0:
                print(f"    Epoka {ep+1:4d}/{epochs} | "
                      f"Train={train_loss:.4f} | Val={val_loss:.4f} | "
                      f"Best={best_ep}")

            if no_improve >= patience:
                print(f"    Early stop epoka {ep+1} "
                      f"(bez poprawy przez {patience} epok). Best={best_ep}")
                break

        for l, (W, b) in zip(self.layers, best_W):
            l.W, l.b = W, b
        print(f"    Wagi z epoki {best_ep} (val={best_val:.4f})")
        return tl, vl

    def get_weights_info(self):
        return [{
            'Warstwa':    i + 1,
            'n_wejść':    l.W.shape[0],
            'n_neuronów': l.W.shape[1],
            'Aktywacja':  l.activation_name,
            'Dropout':    l.dropout,
            'W_mean':     float(np.mean(l.W)),
            'W_std':      float(np.std(l.W)),
            'W_min':      float(np.min(l.W)),
            'W_max':      float(np.max(l.W)),
        } for i, l in enumerate(self.layers)]


# ============================================================
# 5. DYSTRYBUANTA
# ============================================================

def compute_cdf(errors):
    s = np.sort(errors)
    return s, np.arange(1, len(s) + 1) / len(s)


# ============================================================
# 6. WYKRESY
# ============================================================

def plot_training_curve(tl, vl, path, title=""):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tl, label='Train MSE', color='steelblue')
    ax.plot(vl, label='Val MSE',   color='orange')
    ax.set_xlabel('Epoka'); ax.set_ylabel('MSE')
    ax.set_title(f'Krzywa uczenia {title}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Zapisano: {path}")


def plot_cdf(raw, nn_no, nn_out, path, title=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    e, c = compute_cdf(raw);   ax.plot(e, c, 'b-',  lw=2.5, label='Surowe UWB')
    e, c = compute_cdf(nn_no); ax.plot(e, c, 'g--', lw=2.5, label='NN bez eliminacji outlierów')
    if nn_out is not None:
        e, c = compute_cdf(nn_out)
        ax.plot(e, c, 'r-.', lw=2.5, label='NN z eliminacją outlierów')
    for p, ls in [(0.5,':'), (0.75,'--'), (0.9,'-.')]:
        ax.axhline(p, color='gray', lw=0.8, ls=ls, alpha=0.5)
    ax.set_xlabel('Błąd euklidesowy [mm]', fontsize=12)
    ax.set_ylabel('Dystrybuanta F(e)', fontsize=12)
    ax.set_title(f'Porównanie dystrybuant błędu UWB {title}', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Zapisano: {path}")


def plot_trajectory(cx, cy, rx, ry, px, py, path, title=""):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(rx, ry, 'k-',  lw=2,           label='Referencja')
    ax.plot(cx, cy, 'b.',  ms=3, alpha=0.4, label='UWB surowy')
    ax.plot(px, py, 'r.',  ms=3, alpha=0.5, label='Po korekcji NN')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Zapisano: {path}")


def plot_error_time(raw, nn_no, nn_out, path, title=""):
    fig, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(len(raw))
    ax.plot(t, raw,    alpha=0.5, color='blue',  label='Surowy')
    ax.plot(t, nn_no,  alpha=0.6, color='green', label='NN bez outlier')
    ax.plot(t, nn_out, alpha=0.6, color='red',   label='NN z outlier')
    ax.set_xlabel('Próbka'); ax.set_ylabel('Błąd [mm]')
    ax.set_title(f'Błąd w czasie {title}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Zapisano: {path}")


# ============================================================
# 7. POTOK DLA JEDNEJ SALI
# ============================================================

def run_sala(sala, df_static, df_dynamic):
    print(f"\n{'='*65}\n  SALA: {sala}\n{'='*65}")
    out = os.path.join(OUTPUT_DIR, sala)
    os.makedirs(out, exist_ok=True)

    coord_x, coord_y, ref_x, ref_y = detect_column_names(df_static)

    # --- Selekcja cech na podstawie shiftu stat→dyn ---
    all_cols = get_all_numeric_cols(df_static, coord_x, coord_y, ref_x, ref_y)

    if df_dynamic is not None:
        # Potrzebujemy df_dyn z uzupełnioną referencją do obliczenia shiftu
        df_dyn_tmp = df_dynamic.copy()
        df_dyn_tmp[ref_x] = df_dyn_tmp[ref_x].ffill().bfill()
        df_dyn_tmp[ref_y] = df_dyn_tmp[ref_y].ffill().bfill()
        feature_cols = select_stable_features(
            df_static, df_dyn_tmp, all_cols, MAX_FEATURE_SHIFT)
    else:
        # Brak danych dynamicznych — użyj wszystkich cech
        feature_cols = all_cols
        print("  Brak danych dynamicznych — używam wszystkich cech.")

    # Zawsze upewnij się że cx, cy są w feature_cols
    for c in [coord_x, coord_y]:
        if c not in feature_cols:
            feature_cols.append(c)

    if not feature_cols:
        print("  BŁĄD: Brak cech po selekcji!")
        return None, None, None

    use_cols = list(dict.fromkeys(
        feature_cols + [coord_x, coord_y, ref_x, ref_y]))
    df_stat = prepare_df(
        df_static[use_cols].copy(), feature_cols,
        coord_x, coord_y, ref_x, ref_y)

    print(f"  Próbki statyczne po czyszczeniu: {len(df_stat)}")
    if len(df_stat) < SEQUENCE_LEN * 2:
        print("  BŁĄD: Za mało danych.")
        return None, None, None

    # --- OutlierDetector (uczony na danych statycznych) ---
    od = OutlierDetector(OUTLIER_THRESHOLD, IQR_FACTOR, window=15)
    od.fit(df_stat[coord_x].values.astype(float),
           df_stat[coord_y].values.astype(float))

    # --- Macierze cech (z i bez outlier) ---
    print(f"  Budowanie cech (seq_len={SEQUENCE_LEN})...")
    (X, dY, err_stat,
     cx_s, cy_s, rx_s, ry_s) = build_feature_matrix(
        df_stat, feature_cols, coord_x, coord_y, ref_x, ref_y,
        seq_len=SEQUENCE_LEN, outlier_detector=None)

    (X_o, dY_o, _,
     cx_so, cy_so, _, _) = build_feature_matrix(
        df_stat, feature_cols, coord_x, coord_y, ref_x, ref_y,
        seq_len=SEQUENCE_LEN, outlier_detector=od)

    # --- Skalowanie ---
    # StandardScaler dla X i Y (delt) — rozkłady delt są podobne w obu zbiorach
    scX    = StandardScaler()
    scY    = StandardScaler()
    scY_o  = StandardScaler()

    Xsc    = scX.fit_transform(X)
    Xosc   = scX.transform(X_o)
    dYsc   = scY.fit_transform(dY)
    dYosc  = scY_o.fit_transform(dY_o)

    # --- Podział 70/30 ---
    sp = int(0.7 * len(Xsc))
    Xtr,  Xvl  = Xsc[:sp],   Xsc[sp:]
    Ytr,  Yvl  = dYsc[:sp],  dYsc[sp:]
    Xotr, Xovl = Xosc[:sp],  Xosc[sp:]
    Yotr, Yovl = dYosc[:sp], dYosc[sp:]

    nf = Xtr.shape[1]
    print(f"  Cechy: {nf} | Train: {len(Xtr)} | Val: {len(Xvl)}")

    # ====== SIEĆ 1 — BEZ eliminacji outlierów ======
    print(f"\n  [Sieć 1 – {sala}] Trening BEZ eliminacji outlierów...")
    nn1 = NeuralNetwork(nf, HIDDEN_LAYERS, LEARNING_RATE, dropout=DROPOUT_RATE)
    tl1, vl1 = nn1.fit(Xtr, Ytr, Xvl, Yvl,
                        EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE)
    plot_training_curve(tl1, vl1,
                        os.path.join(out, "training_curve_no_outlier.png"),
                        f"{sala} – bez outlier")

    # ====== SIEĆ 2 — Z eliminacją outlierów ======
    print(f"\n  [Sieć 2 – {sala}] Trening Z eliminacją outlierów...")
    nn2 = NeuralNetwork(nf, HIDDEN_LAYERS, LEARNING_RATE, dropout=DROPOUT_RATE)
    tl2, vl2 = nn2.fit(Xotr, Yotr, Xovl, Yovl,
                        EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE)
    plot_training_curve(tl2, vl2,
                        os.path.join(out, "training_curve_with_outlier.png"),
                        f"{sala} – z outlier")

    # Zapis wag
    pd.DataFrame(nn2.get_weights_info()).to_excel(
        os.path.join(out, f"wagi_{sala}.xlsx"), index=False, engine='openpyxl')

    # ====== DANE TESTOWE ======
    if df_dynamic is not None and len(df_dynamic) > 0:
        feat_dyn = [c for c in feature_cols if c in df_dynamic.columns]
        use_dyn  = list(dict.fromkeys(
            feat_dyn + [coord_x, coord_y, ref_x, ref_y]))
        use_dyn  = [c for c in use_dyn if c in df_dynamic.columns]
        df_dyn   = prepare_df(df_dynamic[use_dyn].copy(), feat_dyn,
                              coord_x, coord_y, ref_x, ref_y)
        print(f"\n  Próbki dynamiczne (test): {len(df_dyn)}")

        if len(df_dyn) >= SEQUENCE_LEN and feat_dyn:
            (Xt, _, err_test,
             cx_t, cy_t, rx_t, ry_t) = build_feature_matrix(
                df_dyn, feat_dyn, coord_x, coord_y, ref_x, ref_y,
                seq_len=SEQUENCE_LEN, outlier_detector=None)

            (Xto, _, _,
             cx_to, cy_to, _, _) = build_feature_matrix(
                df_dyn, feat_dyn, coord_x, coord_y, ref_x, ref_y,
                seq_len=SEQUENCE_LEN, outlier_detector=od)

            Xt  = align_features(Xt,  X.shape[1], "test")
            Xto = align_features(Xto, X.shape[1], "test+od")
            Xtsc  = scX.transform(Xt)
            Xtosc = scX.transform(Xto)
            src = "dynamiczne"
        else:
            print("  Za mało danych dynamicznych — fallback na walidację.")
            Xtsc, Xtosc = Xvl, Xovl
            err_test = err_stat[sp:]
            cx_t,  cy_t  = cx_s[sp:],  cy_s[sp:]
            cx_to, cy_to = cx_so[sp:], cy_so[sp:]
            rx_t,  ry_t  = rx_s[sp:],  ry_s[sp:]
            src = "statyczne-walidacja"
    else:
        Xtsc, Xtosc = Xvl, Xovl
        err_test = err_stat[sp:]
        cx_t,  cy_t  = cx_s[sp:],  cy_s[sp:]
        cx_to, cy_to = cx_so[sp:], cy_so[sp:]
        rx_t,  ry_t  = rx_s[sp:],  ry_s[sp:]
        src = "statyczne-walidacja"

    print(f"  Źródło testowe: {src}")

    # --- Predykcja: delta → dodaj do cx/cy ---
    def predict_pos(nn, scY_used, Xtest, cx_ref, cy_ref):
        d = scY_used.inverse_transform(nn.predict(Xtest))
        return cx_ref + d[:, 0], cy_ref + d[:, 1]

    px1, py1 = predict_pos(nn1, scY,   Xtsc,  cx_t,  cy_t)
    px2, py2 = predict_pos(nn2, scY_o, Xtosc, cx_to, cy_to)

    err1 = np.sqrt((px1 - rx_t)**2 + (py1 - ry_t)**2)
    err2 = np.sqrt((px2 - rx_t)**2 + (py2 - ry_t)**2)

    # Zapis wyników
    pd.DataFrame({
        'pred_x': px2, 'pred_y': py2,
        'ref_x':  rx_t, 'ref_y': ry_t,
        'blad_nn_mm': err2, 'blad_surowy_mm': err_test,
    }).to_excel(os.path.join(out, f"wyniki_{sala}.xlsx"),
                index=False, engine='openpyxl')

    # Wykresy
    plot_cdf(err_test, err1, err2,
             os.path.join(out, "cdf_comparison.png"),
             f"– {sala} ({src})")
    plot_trajectory(cx_t, cy_t, rx_t, ry_t, px1, py1,
                    os.path.join(out, "trajectory_no_outlier.png"),
                    f"Trajektoria {sala} – NN bez outlier ({src})")
    plot_trajectory(cx_to, cy_to, rx_t, ry_t, px2, py2,
                    os.path.join(out, "trajectory_with_outlier.png"),
                    f"Trajektoria {sala} – NN z outlier ({src})")
    plot_error_time(err_test, err1, err2,
                    os.path.join(out, "error_vs_time.png"),
                    f"– {sala} ({src})")

    # Statystyki
    def st(e):
        return (f"med={np.median(e):.0f} | "
                f"75%={np.percentile(e,75):.0f} | "
                f"90%={np.percentile(e,90):.0f} | "
                f"mean={np.mean(e):.0f} mm")

    print(f"\n  Statystyki [{sala}] ({src}):")
    print(f"    Surowy UWB     : {st(err_test)}")
    print(f"    NN bez outlier : {st(err1)}")
    print(f"    NN z outlier   : {st(err2)}")
    med_imp = (np.median(err_test)-np.median(err2))/np.median(err_test)*100
    p90_imp = (np.percentile(err_test,90)-np.percentile(err2,90))/np.percentile(err_test,90)*100
    print(f"    Poprawa mediany: {med_imp:+.1f}%")
    print(f"    Poprawa 90%:     {p90_imp:+.1f}%")

    return err_test, err1, err2


# ============================================================
# 8. GŁÓWNY POTOK
# ============================================================

def run_pipeline(pomiary_dir):
    print("=" * 65)
    print("  Korekcja UWB – Sieć Neuronowa v4")
    print(f"  Architektura: {HIDDEN_LAYERS} → 2 (ReLU+Dropout / Linear)")
    print(f"  Target: delta (rx-cx, ry-cy)")
    print(f"  Selekcja cech: automatyczna (shift ≤ {MAX_FEATURE_SHIFT}σ)")
    print(f"  Outlier: z>{OUTLIER_THRESHOLD}σ AND dist>Q3+{IQR_FACTOR}×IQR")
    print("=" * 65)

    all_raw, all_nn1, all_nn2 = [], [], []
    results = {}

    for sala in ['F8', 'F10']:
        print(f"\n[Wczytywanie {sala}]...")
        try:
            df_s, df_d = load_sala_data(pomiary_dir, sala)
        except FileNotFoundError as e:
            print(f"  POMINIĘTO: {e}")
            continue

        r, n1, n2 = run_sala(sala, df_s, df_d)
        if r is None:
            continue
        all_raw.extend(r.tolist())
        all_nn1.extend(n1.tolist())
        all_nn2.extend(n2.tolist())
        results[sala] = (r, n1, n2)

    if not results:
        print("Brak wyników.")
        return

    # Zbiorczy CDF
    plot_cdf(np.array(all_raw), np.array(all_nn1), np.array(all_nn2),
             os.path.join(OUTPUT_DIR, "cdf_comparison_all.png"),
             "– F8 + F10 łącznie")

    # Pliki dystrybuant (jedna kolumna — wymóg zadania)
    for sala, (r, n1, n2) in results.items():
        se, cv = compute_cdf(n2)
        pd.DataFrame({
            'dystrybuanta_bledu_NN': cv,
            'blad_mm': se,
        }).to_excel(
            os.path.join(OUTPUT_DIR, sala, f"dystrybuanta_NN_{sala}.xlsx"),
            index=False, engine='openpyxl')
        print(f"  Dystrybuanta: {sala}")

    # Podsumowanie
    print(f"\n{'='*65}\n  PODSUMOWANIE\n{'='*65}")
    for sala, (r, n1, n2) in results.items():
        print(f"  {sala}:")
        print(f"    Surowy     med={np.median(r):.0f} mm, 90%={np.percentile(r,90):.0f} mm")
        print(f"    NN+outlier med={np.median(n2):.0f} mm, 90%={np.percentile(n2,90):.0f} mm")
    print(f"\n  Wyniki: {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        POMIARY_DIR = sys.argv[1]
    run_pipeline(POMIARY_DIR)