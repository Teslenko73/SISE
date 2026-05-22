import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

SALA = "F8" # zmienic na f8/f10
FILE_PATH = fr"C:\Users\user\PycharmProjects\SISE\zad2_norm\wyniki\{SALA}\5_Wyniki_{SALA}.xlsx"
OUT_DIR = fr"C:\Users\user\PycharmProjects\SISE\zad2_norm\wyniki\{SALA}"

df = pd.read_excel(FILE_PATH)

err_nn = df['Dystrybuanta_Bledu_Euklidesowego_NN'].values
err_raw = df['Blad_Surowy_Euklidesowy'].values

if np.mean(err_nn) > np.mean(err_raw):
    err_nn, err_raw = err_raw, err_nn

plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#e74c3c', '#3498db']

# BOXPLOT
plt.figure(figsize=(8, 6))
sns.boxplot(data=[err_raw, err_nn], palette=colors, width=0.5, fliersize=3)
plt.xticks([0, 1], ['Surowy UWB', 'Po korekcji NN'], fontsize=12)
plt.ylabel('Błąd euklidesowy [mm]', fontsize=12)
plt.title(f'Rozkład błędów i wartości odstające (Boxplot) - {SALA}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplot_{SALA}.png"), dpi=200)
plt.close()

# HISTOGRAM

plt.figure(figsize=(10, 6))
sns.histplot(err_raw, color=colors[0], label='Surowy UWB', kde=True, stat="density", linewidth=0, alpha=0.5)
sns.histplot(err_nn, color=colors[1], label='Po korekcji NN', kde=True, stat="density", linewidth=0, alpha=0.5)
plt.xlabel('Błąd euklidesowy [mm]', fontsize=12)
plt.ylabel('Gęstość prawdopodobieństwa', fontsize=12)
plt.title(f'Histogram częstotliwości występowania błędu - {SALA}', fontsize=14)
plt.legend(fontsize=12)
plt.xlim(0, np.percentile(err_raw, 99))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hist_{SALA}.png"), dpi=200)
plt.close()

# LOGARYTMICZNA DYSTRYBUANTA

plt.figure(figsize=(10, 6))
# Считаем CDF
sort_raw = np.sort(err_raw)
sort_nn = np.sort(err_nn)
p_raw = np.arange(1, len(sort_raw) + 1) / len(sort_raw)
p_nn = np.arange(1, len(sort_nn) + 1) / len(sort_nn)

plt.plot(sort_raw, p_raw, color=colors[0], label='Surowy UWB', lw=2.5)
plt.plot(sort_nn, p_nn, color=colors[1], label='Po korekcji NN', lw=2.5)

plt.xscale('log')
plt.xlabel('Błąd euklidesowy [mm] (skala logarytmiczna)', fontsize=12)
plt.ylabel('Prawdopodobieństwo skumulowane (CDF)', fontsize=12)
plt.title(f'Dystrybuanta błędu w skali logarytmicznej - {SALA}', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "log_{SALA}.png"), dpi=200)
plt.close()