"""
Diagnostyka danych UWB - przed naprawą sieci
=============================================
Sprawdzamy:
  1. Rozkład wartości cech w danych statycznych vs dynamicznych
  2. Rozkład błędów surowego UWB (statyczne vs dynamiczne)
  3. Wartości docelowe (rx, ry) - czy pokrywają się z zakresem treningu
  4. Korelacja między cechami a błędem - czy w ogóle da się nauczyć korekcji
  5. Przykładowe wiersze danych
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

POMIARY_DIR = "./pomiary"
OUTPUT_DIR  = "./diagnostyka"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# WCZYTYWANIE (uproszczone)
# ============================================================

def load_xlsx_files(data_dir, pattern):
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, engine='openpyxl')
            df['_plik'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Błąd: {f}: {e}")
    return dfs


def detect_cols(df):
    cols = [c.lower().strip() for c in df.columns]
    orig = df.columns.tolist()
    coord_x = coord_y = ref_x = ref_y = None
    for i, c in enumerate(cols):
        if 'coordinate' in c and 'x' in c:  coord_x = orig[i]
        elif 'coordinate' in c and 'y' in c: coord_y = orig[i]
        elif 'reference' in c and 'x' in c:  ref_x   = orig[i]
        elif 'reference' in c and 'y' in c:  ref_y   = orig[i]
    if coord_x is None:
        coord_x, coord_y, ref_x, ref_y = orig[-4], orig[-3], orig[-2], orig[-1]
    return coord_x, coord_y, ref_x, ref_y


# ============================================================
# DIAGNOSTYKA DLA JEDNEJ SALI
# ============================================================

def diagnose_sala(sala):
    sala_dir = os.path.join(POMIARY_DIR, sala)
    prefix   = sala.lower()
    out_dir  = os.path.join(OUTPUT_DIR, sala)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DIAGNOSTYKA: {sala}")
    print(f"{'='*60}")

    # Wczytaj
    stat_dfs = load_xlsx_files(sala_dir, f"{prefix}_stat_*.xlsx")
    dyn_dfs  = []
    for pat in [f"{prefix}_[0-9]p.xlsx", f"{prefix}_[0-9]z.xlsx", f"{prefix}_random_*.xlsx"]:
        dyn_dfs.extend(load_xlsx_files(sala_dir, pat))

    if not stat_dfs:
        print("  Brak danych statycznych!")
        return
    if not dyn_dfs:
        print("  Brak danych dynamicznych!")
        return

    df_stat = pd.concat(stat_dfs, ignore_index=True)
    df_dyn  = pd.concat(dyn_dfs,  ignore_index=True)

    coord_x, coord_y, ref_x, ref_y = detect_cols(df_stat)
    print(f"  Kolumny: cx={coord_x}, cy={coord_y}, rx={ref_x}, ry={ref_y}")

    # Wypełnij referencję
    for df in [df_stat, df_dyn]:
        df[ref_x] = df[ref_x].ffill().bfill()
        df[ref_y] = df[ref_y].ffill().bfill()

    df_stat = df_stat.dropna(subset=[coord_x, coord_y, ref_x, ref_y])
    df_dyn  = df_dyn.dropna(subset=[coord_x, coord_y, ref_x, ref_y])

    # ---- 1. PODSTAWOWE STATYSTYKI ----
    print(f"\n--- 1. LICZNOŚĆ ---")
    print(f"  Statyczne:  {len(df_stat)} próbek")
    print(f"  Dynamiczne: {len(df_dyn)} próbek")

    print(f"\n--- 2. ZAKRESY POZYCJI ---")
    for label, df in [("Statyczne ", df_stat), ("Dynamiczne", df_dyn)]:
        print(f"  {label} | "
              f"cx: [{df[coord_x].min():.0f}, {df[coord_x].max():.0f}] mm | "
              f"cy: [{df[coord_y].min():.0f}, {df[coord_y].max():.0f}] mm | "
              f"rx: [{df[ref_x].min():.0f}, {df[ref_x].max():.0f}] mm | "
              f"ry: [{df[ref_y].min():.0f}, {df[ref_y].max():.0f}] mm")

    # ---- 2. BŁĘDY SUROWEGO UWB ----
    err_stat = np.sqrt((df_stat[coord_x] - df_stat[ref_x])**2 +
                       (df_stat[coord_y] - df_stat[ref_y])**2)
    err_dyn  = np.sqrt((df_dyn[coord_x]  - df_dyn[ref_x])**2  +
                       (df_dyn[coord_y]  - df_dyn[ref_y])**2)

    print(f"\n--- 3. BŁĄD SUROWEGO UWB ---")
    for label, err in [("Statyczne ", err_stat), ("Dynamiczne", err_dyn)]:
        print(f"  {label} | "
              f"mediana={np.median(err):.0f} mm | "
              f"75%={np.percentile(err,75):.0f} mm | "
              f"90%={np.percentile(err,90):.0f} mm | "
              f"max={np.max(err):.0f} mm")

    # ---- 3. DELTY (target sieci) ----
    dx_stat = df_stat[ref_x] - df_stat[coord_x]
    dy_stat = df_stat[ref_y] - df_stat[coord_y]
    dx_dyn  = df_dyn[ref_x]  - df_dyn[coord_x]
    dy_dyn  = df_dyn[ref_y]  - df_dyn[coord_y]

    print(f"\n--- 4. DELTY (rx-cx, ry-cy) ---")
    print(f"  Statyczne  dx: [{dx_stat.min():.0f}, {dx_stat.max():.0f}] mm, "
          f"śred={dx_stat.mean():.0f}, std={dx_stat.std():.0f}")
    print(f"  Statyczne  dy: [{dy_stat.min():.0f}, {dy_stat.max():.0f}] mm, "
          f"śred={dy_stat.mean():.0f}, std={dy_stat.std():.0f}")
    print(f"  Dynamiczne dx: [{dx_dyn.min():.0f}, {dx_dyn.max():.0f}] mm, "
          f"śred={dx_dyn.mean():.0f}, std={dx_dyn.std():.0f}")
    print(f"  Dynamiczne dy: [{dy_dyn.min():.0f}, {dy_dyn.max():.0f}] mm, "
          f"śred={dy_dyn.mean():.0f}, std={dy_dyn.std():.0f}")

    # ---- 4. UNIKALNE POZYCJE REFERENCYJNE ----
    ref_stat_unique = df_stat[[ref_x, ref_y]].drop_duplicates()
    ref_dyn_unique  = df_dyn[[ref_x,  ref_y]].drop_duplicates()
    print(f"\n--- 5. UNIKALNE POZYCJE REFERENCYJNE ---")
    print(f"  Statyczne:  {len(ref_stat_unique)} unikalnych pozycji")
    print(ref_stat_unique.to_string(index=False))
    print(f"  Dynamiczne: {len(ref_dyn_unique)} unikalnych pozycji "
          f"({'ciągła trajektoria' if len(ref_dyn_unique) > 50 else 'kilka punktów'})")

    # ---- 5. KOLUMNY CECH ----
    META = {'unnamed: 0.1','unnamed: 0','version','alive','tagid','success',
            'timestamp','data__anchordata','errorcode','data__coordinates__z',
            '_source_file','_plik', ref_x.lower(), ref_y.lower()}
    num_cols = df_stat.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c.lower() not in META]
    print(f"\n--- 6. KOLUMNY CECH ({len(feat_cols)} szt.) ---")
    print(f"  {feat_cols}")

    # Sprawdź czy te same kolumny są w danych dynamicznych
    feat_in_dyn = [c for c in feat_cols if c in df_dyn.columns]
    feat_missing = [c for c in feat_cols if c not in df_dyn.columns]
    print(f"  Cechy obecne w dynamicznych: {len(feat_in_dyn)}/{len(feat_cols)}")
    if feat_missing:
        print(f"  BRAKUJĄCE w dynamicznych: {feat_missing}")

    # ---- 6. ROZKŁAD CECH stat vs dyn ----
    print(f"\n--- 7. PORÓWNANIE ROZKŁADÓW CECH (stat vs dyn) ---")
    shift_report = []
    for c in feat_in_dyn[:20]:  # max 20 cech
        ms = df_stat[c].mean()
        ss = df_stat[c].std() + 1e-9
        md = df_dyn[c].mean()
        sd = df_dyn[c].std()
        shift = abs(md - ms) / ss  # przesunięcie w jednostkach std
        shift_report.append({'cecha': c, 'mean_stat': ms, 'mean_dyn': md,
                              'std_stat': ss, 'std_dyn': sd, 'shift_std': shift})
        flag = " ← DUŻE PRZESUNIĘCIE" if shift > 1.0 else ""
        print(f"  {c:40s} | shift={shift:.2f}σ{flag}")

    # ============================================================
    # WYKRESY
    # ============================================================

    # Wykres 1: CDF błędu surowego stat vs dyn
    fig, ax = plt.subplots(figsize=(9, 5))
    for err, label, color in [(err_stat, 'Statyczne', 'blue'),
                               (err_dyn,  'Dynamiczne', 'red')]:
        se = np.sort(err)
        cdf = np.arange(1, len(se)+1) / len(se)
        ax.plot(se, cdf, color=color, linewidth=2, label=label)
    ax.set_xlabel('Błąd euklidesowy surowego UWB [mm]')
    ax.set_ylabel('CDF')
    ax.set_title(f'{sala} – CDF błędu surowego UWB: statyczne vs dynamiczne')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cdf_surowy_stat_vs_dyn.png'), dpi=150)
    plt.close()

    # Wykres 2: Rozkład delt dx, dy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dx_s, dx_d, label in [
        (axes[0], dx_stat, dx_dyn, 'delta X (rx-cx)'),
        (axes[1], dy_stat, dy_dyn, 'delta Y (ry-cy)'),
    ]:
        bins = np.linspace(
            min(dx_s.min(), dx_d.min()),
            max(dx_s.max(), dx_d.max()), 80)
        ax.hist(dx_s, bins=bins, alpha=0.5, color='blue',
                label='Statyczne', density=True)
        ax.hist(dx_d, bins=bins, alpha=0.5, color='red',
                label='Dynamiczne', density=True)
        ax.set_title(f'{sala} – Rozkład {label}')
        ax.set_xlabel('mm'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rozklad_delt.png'), dpi=150)
    plt.close()

    # Wykres 3: Pozycje referencyjne na mapie
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, label in [(axes[0], df_stat, 'Statyczne'),
                           (axes[1], df_dyn,  'Dynamiczne')]:
        ax.scatter(df[coord_x], df[coord_y], s=1, alpha=0.3,
                   color='blue', label='UWB (cx, cy)')
        ax.scatter(df[ref_x],   df[ref_y],   s=1, alpha=0.3,
                   color='red',  label='Referencja (rx, ry)')
        ax.set_title(f'{sala} – Pozycje {label}')
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]')
        ax.legend(markerscale=5); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pozycje_na_mapie.png'), dpi=150)
    plt.close()

    # Wykres 4: Rozkłady kilku kluczowych cech stat vs dyn
    n_plot = min(6, len(feat_in_dyn))
    # Sortuj po największym przesunięciu żeby zobaczyć najgorsze
    shift_report.sort(key=lambda x: x['shift_std'], reverse=True)
    top_feats = [r['cecha'] for r in shift_report[:n_plot]]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, c in enumerate(top_feats):
        ax = axes[i]
        vals_s = df_stat[c].dropna().values
        vals_d = df_dyn[c].dropna().values
        lo = np.percentile(np.concatenate([vals_s, vals_d]), 1)
        hi = np.percentile(np.concatenate([vals_s, vals_d]), 99)
        bins = np.linspace(lo, hi, 60)
        ax.hist(vals_s, bins=bins, alpha=0.5, color='blue',
                label='Stat', density=True)
        ax.hist(vals_d, bins=bins, alpha=0.5, color='red',
                label='Dyn', density=True)
        shift = shift_report[i]['shift_std']
        ax.set_title(f'{c}\nshift={shift:.2f}σ', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    for j in range(n_plot, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f'{sala} – Cechy z największym przesunięciem rozkładu (stat vs dyn)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rozklad_cech_top.png'), dpi=150)
    plt.close()

    print(f"\n  Wykresy zapisane w: {out_dir}")
    print(f"  Pliki: cdf_surowy_stat_vs_dyn.png, rozklad_delt.png, "
          f"pozycje_na_mapie.png, rozklad_cech_top.png")

    return {
        'err_stat': err_stat,
        'err_dyn':  err_dyn,
        'dx_stat':  dx_stat, 'dy_stat': dy_stat,
        'dx_dyn':   dx_dyn,  'dy_dyn':  dy_dyn,
        'shift_report': shift_report,
        'feat_missing': feat_missing,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        POMIARY_DIR = sys.argv[1]

    for sala in ['F8', 'F10']:
        sala_dir = os.path.join(POMIARY_DIR, sala)
        if os.path.isdir(sala_dir):
            diagnose_sala(sala)
        else:
            print(f"Brak katalogu: {sala_dir}")

    print(f"\nDiagnostyka zakończona. Wyniki w: {OUTPUT_DIR}")