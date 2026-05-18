"""
solver.py
---------
Czyste funkcje rozwiązujące układankę N-puzzle.
Nie importują nic z pygame ani csv — tylko logika algorytmów.

Każda funkcja zwraca:
    (node | None, visited_count, processed_count, max_depth)

node.parent tworzy łańcuch do odczytania ścieżki ruchów.
"""

import heapq
from collections import deque


# ======================================================================= #
# Węzeł drzewa przeszukiwań                                                #
# ======================================================================= #

class Node:
    __slots__ = ("board", "parent", "move", "depth")

    def __init__(self, board: tuple, parent, move: str | None, depth: int):
        self.board  = board
        self.parent = parent
        self.move   = move
        self.depth  = depth

    def __hash__(self):
        return hash(self.board)

    def __eq__(self, other):
        return self.board == other.board

    def __lt__(self, other):          # wymagane przez heapq
        return False


# ======================================================================= #
# Pomocnicze                                                                #
# ======================================================================= #

def get_target(rows: int, cols: int) -> tuple:
    return tuple(range(1, rows * cols)) + (0,)


def extract_path(node: Node) -> list[str]:
    """Odtwarza listę ruchów od korzenia do node."""
    path = []
    while node.parent is not None:
        path.append(node.move)
        node = node.parent
    path.reverse()
    return path


def get_neighbors(node: Node, rows: int, cols: int, order: str) -> list[Node]:
    """
    Generuje sąsiadów węzła w kolejności zdefiniowanej przez `order`
    (łańcuch liter U/D/L/R, np. 'RDLU').
    """
    board  = list(node.board)
    z_idx  = board.index(0)
    r, c   = z_idx // cols, z_idx % cols

    MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    neighbors = []

    for m in order:
        dr, dc = MOVES[m]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            new_idx   = nr * cols + nc
            nb        = board[:]
            nb[z_idx], nb[new_idx] = nb[new_idx], nb[z_idx]
            neighbors.append(Node(tuple(nb), node, m, node.depth + 1))

    return neighbors


# ======================================================================= #
# BFS                                                                       #
# ======================================================================= #

def solve_bfs(start_board: tuple, rows: int, cols: int, order: str = "RDLU"):
    """
    Przeszukiwanie wszerz (BFS).
    Gwarantuje optymalne rozwiązanie.
    """
    target  = get_target(rows, cols)
    root    = Node(start_board, None, None, 0)

    if start_board == target:
        return root, 1, 1, 0

    queue   = deque([root])
    visited = {start_board}
    visited_count   = 1
    processed_count = 0
    max_depth       = 0

    while queue:
        current = queue.popleft()
        processed_count += 1
        max_depth = max(max_depth, current.depth)

        for nxt in get_neighbors(current, rows, cols, order):
            if nxt.board == target:
                return nxt, visited_count + 1, processed_count, nxt.depth
            if nxt.board not in visited:
                visited.add(nxt.board)
                visited_count += 1
                queue.append(nxt)

    return None, visited_count, processed_count, max_depth


# ======================================================================= #
# DFS                                                                       #
# ======================================================================= #

DFS_MAX_DEPTH = 20


def solve_dfs(start_board: tuple, rows: int, cols: int, order: str = "RDLU"):
    """
    Przeszukiwanie wgłąb (DFS) z limitem głębokości DFS_MAX_DEPTH.
    Kolejność sąsiadów odwrócona (stos LIFO → efektywna kolejność = order).
    """
    target       = get_target(rows, cols)
    root         = Node(start_board, None, None, 0)
    order_rev    = order[::-1]

    stack           = [root]
    best_depth      = {start_board: 0}
    visited_count   = 1
    processed_count = 0
    max_depth       = 0

    while stack:
        current = stack.pop()
        processed_count += 1
        max_depth = max(max_depth, current.depth)

        if current.board == target:
            return current, visited_count, processed_count, max_depth

        if current.depth < DFS_MAX_DEPTH:
            for nxt in get_neighbors(current, rows, cols, order_rev):
                if nxt.board not in best_depth or nxt.depth < best_depth[nxt.board]:
                    best_depth[nxt.board] = nxt.depth
                    stack.append(nxt)
                    visited_count += 1

    return None, visited_count, processed_count, max_depth


# ======================================================================= #
# Heurystyki dla A*                                                          #
# ======================================================================= #

def hamming(board: tuple, target: tuple) -> int:
    return sum(1 for i in range(len(board)) if board[i] != 0 and board[i] != target[i])


def manhattan(board: tuple, target: tuple, rows: int, cols: int) -> int:
    # Przelicz pozycje docelowe raz
    goal_pos = {v: (i // cols, i % cols) for i, v in enumerate(target) if v != 0}
    dist = 0
    for i, v in enumerate(board):
        if v != 0:
            gr, gc = goal_pos[v]
            cr, cc = i // cols, i % cols
            dist += abs(gr - cr) + abs(gc - cc)
    return dist


# ======================================================================= #
# A*                                                                        #
# ======================================================================= #

def solve_astar(start_board: tuple, rows: int, cols: int, heuristic: str = "manh"):
    """
    Algorytm A*.
    heuristic: 'manh' (Manhattan) lub 'hamm' (Hamming).
    """
    target = get_target(rows, cols)
    root   = Node(start_board, None, None, 0)

    h0 = (manhattan(start_board, target, rows, cols)
          if heuristic == "manh"
          else hamming(start_board, target))

    counter         = 0
    open_set        = [(h0, counter, root)]
    g_scores        = {start_board: 0}
    visited_count   = 1
    processed_count = 0
    max_depth       = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)
        processed_count += 1
        max_depth = max(max_depth, current.depth)

        if current.board == target:
            return current, visited_count, processed_count, max_depth

        for nxt in get_neighbors(current, rows, cols, "LUDR"):
            tentative_g = current.depth + 1
            if nxt.board not in g_scores or tentative_g < g_scores[nxt.board]:
                g_scores[nxt.board] = tentative_g
                h = (manhattan(nxt.board, target, rows, cols)
                     if heuristic == "manh"
                     else hamming(nxt.board, target))
                counter += 1
                heapq.heappush(open_set, (tentative_g + h, counter, nxt))
                visited_count += 1

    return None, visited_count, processed_count, max_depth