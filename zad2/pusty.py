import pandas as pd

# Sprawdź wszystkie arkusze
xl = pd.ExcelFile("./pomiary/F8/f8_stat_1.xlsx")
print("Arkusze:", xl.sheet_names)

# Wczytaj bez założeń o nagłówku
df_raw = pd.read_excel("./pomiary/F8/f8_stat_1.xlsx", header=None, nrows=5)
print("\nPierwsze 5 wierszy bez parsowania nagłówka:")
print(df_raw.to_string())
