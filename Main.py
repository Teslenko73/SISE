"""
main.py
-------
Punkt wejścia programu (CLI).

Użycie:
    python main.py <algorytm> <parametr> <plik_wejściowy> <plik_rozwiązania> <plik_statystyk> [--gui]

Algorytmy:
    bfs   <kolejność>    np. bfs RDLU
    dfs   <kolejność>    np. dfs RDLU
    astr  <heurystyka>   np. astr manh   |  astr hamm

Przykład:
    python main.py bfs RDLU puzzle.txt solution.txt stats.txt
    python main.py astr manh puzzle.txt solution.txt stats.txt --gui
"""

import sys
import time

from Puzzle import Puzzle
from Solver import solve_bfs, solve_dfs, solve_astar, extract_path


def parse_args():
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)

    algo      = sys.argv[1].lower()
    param     = sys.argv[2]
    file_in   = sys.argv[3]
    file_sol  = sys.argv[4]
    file_stat = sys.argv[5]
    show_gui  = "--gui" in sys.argv

    return algo, param, file_in, file_sol, file_stat, show_gui


def run_solver(algo: str, param: str, board: tuple, rows: int, cols: int):
    """Wybiera i uruchamia odpowiedni algorytm. Zwraca (node, v, p, d, czas_ms)."""
    t0 = time.perf_counter()

    if algo == "bfs":
        result = solve_bfs(board, rows, cols, order=param)
    elif algo == "dfs":
        result = solve_dfs(board, rows, cols, order=param)
    elif algo == "astr":
        result = solve_astar(board, rows, cols, heuristic=param)
    else:
        print(f"Nieznany algorytm: {algo}")
        sys.exit(1)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return *result, elapsed_ms   # (node, visited, processed, depth, ms)


def write_solution(filepath: str, path: list[str]) -> None:
    with open(filepath, "w") as f:
        if path is None:
            f.write("-1\n")
        else:
            f.write(f"{len(path)}\n")
            f.write("".join(path) + "\n")


def write_stats(filepath: str, path_len: int, visited: int,
                processed: int, depth: int, elapsed_ms: float) -> None:
    with open(filepath, "w") as f:
        f.write(f"{path_len}\n")
        f.write(f"{visited}\n")
        f.write(f"{processed}\n")
        f.write(f"{depth}\n")
        f.write(f"{elapsed_ms:.3f}\n")


def main():
    algo, param, file_in, file_sol, file_stat, show_gui = parse_args()

    # Wczytaj planszę
    board, rows, cols = Puzzle.load_as_tuple(file_in)

    # Uruchom solver
    node, visited, processed, depth, elapsed_ms = run_solver(
        algo, param, board, rows, cols
    )

    # Odtwórz ścieżkę
    if node is not None:
        path = extract_path(node)
        path_len = len(path)
    else:
        path = None
        path_len = -1

    # Zapisz wyniki
    write_solution(file_sol, path)
    write_stats(file_stat, path_len, visited, processed, depth, elapsed_ms)

    # Podsumowanie w konsoli
    if path_len == -1:
        print("Nie znaleziono rozwiązania.")
    else:
        print(f"Rozwiązanie: {path_len} ruchów  |  "
              f"odwiedzone: {visited}  |  przetworzone: {processed}  |  "
              f"czas: {elapsed_ms:.3f} ms")
        print(f"Ruchy: {''.join(path)}")

    # Opcjonalna wizualizacja
    if show_gui and path is not None:
        from gui import run_gui
        run_gui(board, rows, cols, path)


if __name__ == "__main__":
    main()