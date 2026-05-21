"""
benchmark.py
------------
Generuje plik CSV ze statystykami dla wszystkich algorytmów
na losowych planszach o rosnącym poziomie trudności.

Użycie:
    python benchmark.py [--output wyniki.csv] [--order RDLU]
                        [--depths 1-7] [--repeats 59]

Domyślna konfiguracja odpowiada oryginalnym badaniom w projekcie.
"""

import csv
import random
import time
import argparse

from Puzzle import Puzzle
from Solver import solve_bfs, solve_dfs, solve_astar


# ================================================================== #
# Generowanie planszy testowej                                         #
# ================================================================== #

ROWS, COLS = 4, 4
GOAL       = tuple(range(1, ROWS * COLS)) + (0,)


def scramble(goal: tuple, steps: int) -> tuple:
    """
    Tasuje stan docelowy wykonując `steps` losowych legalnych ruchów.
    Gwarantuje osiągalność (brak parzystości inversion).
    """
    board = list(goal)
    z     = board.index(0)

    for _ in range(steps):
        neighbors = []
        r, c = z // COLS, z % COLS
        if r > 0:            neighbors.append(z - COLS)
        if r < ROWS - 1:     neighbors.append(z + COLS)
        if c > 0:            neighbors.append(z - 1)
        if c < COLS - 1:     neighbors.append(z + 1)

        t = random.choice(neighbors)
        board[z], board[t] = board[t], board[z]
        z = t

    return tuple(board)


# ================================================================== #
# Jeden pomiar                                                          #
# ================================================================== #

def measure(board: tuple, rows: int, cols: int, order: str) -> list[dict]:
    """
    Uruchamia BFS, DFS, A*(manh) i A*(hamm) na tej samej planszy.
    Zwraca listę 4 słowników gotowych do zapisu w CSV.
    """
    configs = [
        ("BFS",       lambda: solve_bfs(board, rows, cols, order)),
        ("DFS",       lambda: solve_dfs(board, rows, cols, order)),
        ("A* (Manh)", lambda: solve_astar(board, rows, cols, "manh")),
        ("A* (Hamm)", lambda: solve_astar(board, rows, cols, "hamm")),
    ]

    rows_out = []
    for name, fn in configs:
        t0 = time.perf_counter()
        _, v, p, d = fn()
        ms = (time.perf_counter() - t0) * 1000.0
        rows_out.append({
            "Algorytm":          name,
            "Poziom_trudnosci":  None,   # uzupełni wywołujący
            "Czas_ms":           round(ms, 3),
            "Odwiedzone_stany":  v,
            "Przetworzone_stany": p,
            "Max_glebokosc":     d,
        })
    return rows_out


# ================================================================== #
# Główna pętla benchmarku                                              #
# ================================================================== #

def run_benchmark(output_file: str,
                  order: str,
                  depths: range,
                  repeats: int) -> None:

    fieldnames = [
        "Algorytm", "Poziom_trudnosci", "Czas_ms",
        "Odwiedzone_stany", "Przetworzone_stany", "Max_glebokosc",
    ]

    total_runs = len(depths) * repeats
    done       = 0

    print(f"Benchmark: głębokości {list(depths)}, "
          f"{repeats} powtórzeń każda, kolejność='{order}'")
    print(f"Łącznie planszy: {total_runs}  →  {output_file}")
    print("Proszę czekać...")

    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()

        for depth in depths:
            for rep in range(repeats):
                board = scramble(GOAL, depth)
                rows_data = measure(board, ROWS, COLS, order)
                for row in rows_data:
                    row["Poziom_trudnosci"] = depth
                    writer.writerow(row)

                done += 1
                if done % 10 == 0 or done == total_runs:
                    pct = done / total_runs * 100
                    print(f"  {done}/{total_runs} planszy ({pct:.0f}%)")

    print(f'\nGotowe → "{output_file}"')


# ================================================================== #
# CLI                                                                   #
# ================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark algorytmów N-puzzle")
    parser.add_argument("--output",  default="wyniki.csv", help="Plik wyjściowy CSV")
    parser.add_argument("--order",   default="RDLU",       help="Kolejność ruchów dla BFS/DFS")
    parser.add_argument("--depths",  default="1-7",        help="Zakres głębokości np. '1-7' lub '3-5'")
    parser.add_argument("--repeats", default=59, type=int, help="Liczba powtórzeń na głębokość")
    return parser.parse_args()


def parse_depths(s: str) -> range:
    parts = s.split("-")
    if len(parts) == 2:
        return range(int(parts[0]), int(parts[1]) + 1)
    return range(int(parts[0]), int(parts[0]) + 1)


if __name__ == "__main__":
    args   = parse_args()
    depths = parse_depths(args.depths)
    run_benchmark(args.output, args.order, depths, args.repeats)