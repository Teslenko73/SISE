import solver
import random
import csv
import time

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
ROWS, COLS = 4, 4
ORDER = "RDLU"
PLIK = "RDLU.csv"


def scramble_board(goal, steps):
    board = list(goal)
    for _ in range(steps):
        z_idx = board.index(0)
        r, c = z_idx // COLS, z_idx % COLS
        moves = []
        if r > 0: moves.append(-COLS)
        if r < ROWS - 1: moves.append(COLS)
        if c > 0: moves.append(-1)
        if c < COLS - 1: moves.append(1)

        swap_idx = z_idx + random.choice(moves)
        board[z_idx], board[swap_idx] = board[swap_idx], board[z_idx]
    return tuple(board)


print("Trwają operacje, czekaj od 50 sekund do 1 godziny...")

with open(PLIK, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(
        ['Algorytm', 'Poziom_trudnosci', 'Czas_ms', 'Odwiedzone_stany', 'Przetworzone_stany', 'Max_glebokosc'])

    for depth in range(1, 8):
        for _ in range(59):
            test_board = scramble_board(GOAL, depth)

            # 1. BFS
            start_t = time.perf_counter()
            res, v, p, d = solver.solve_bfs(test_board, ROWS, COLS, ORDER)
            t_ms = (time.perf_counter() - start_t) * 1000.0
            writer.writerow(['BFS', depth, round(t_ms, 3), v, p, d])

            # 2. DFS
            start_t = time.perf_counter()
            res, v, p, d = solver.solve_dfs(test_board, ROWS, COLS, ORDER)
            t_ms = (time.perf_counter() - start_t) * 1000.0
            writer.writerow(['DFS', depth, round(t_ms, 3), v, p, d])

            # 3. A* (Manhattan)
            start_t = time.perf_counter()
            res, v, p, d = solver.solve_astar(test_board, ROWS, COLS, "manh")
            t_ms = (time.perf_counter() - start_t) * 1000.0
            writer.writerow(['A* (Manh)', depth, round(t_ms, 3), v, p, d])

            start_t = time.perf_counter()
            res, v, p, d = solver.solve_astar(test_board, ROWS, COLS, "hamm")
            t_ms = (time.perf_counter() - start_t) * 1000.0
            writer.writerow(['A* (Hamm)', depth, round(t_ms, 3), v, p, d])

print(f'Patrz na "{ORDER}".csv plik')