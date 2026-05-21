import sys
import time
import heapq
from collections import deque


class Node:
    def __init__(self, board, parent, move, depth):
        self.board = board
        self.parent = parent
        self.move = move
        self.depth = depth

    def __hash__(self):
        return hash(self.board)

    def __eq__(self, other):
        return self.board == other.board

    def __lt__(self, other):
        return False


def get_neighbors(node, rows, cols, order):
    board = list(node.board)
    z_idx = board.index(0)
    r, c = z_idx // cols, z_idx % cols
    neighbors = []

    moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

    for move_name in order:
        dr, dc = moves[move_name]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            new_idx = nr * cols + nc
            new_board = board[:]
            new_board[z_idx], new_board[new_idx] = new_board[new_idx], new_board[z_idx]
            neighbors.append(Node(tuple(new_board), node, move_name, node.depth + 1))
    return neighbors


def get_target(rows, cols):
    return tuple(list(range(1, rows * cols)) + [0])


def solve_bfs(start_board, rows, cols, order):
    target = get_target(rows, cols)
    start_node = Node(start_board, None, None, 0)
    queue = deque([start_node])
    visited = {start_board}
    processed = 0
    visited_count = 1
    max_depth = 0

    while queue:
        current = queue.popleft()
        processed += 1
        if current.depth > max_depth:
            max_depth = current.depth

        if current.board == target:
            return current, visited_count, processed, max_depth

        for nxt in get_neighbors(current, rows, cols, order):
            if nxt.board not in visited:
                visited.add(nxt.board)
                queue.append(nxt)
                visited_count += 1

    return None, visited_count, processed, max_depth


def solve_dfs(start_board, rows, cols, order):
    target = get_target(rows, cols)
    start_node = Node(start_board, None, None, 0)
    stack = [start_node]
    visited = {start_board: 0}
    processed = 0
    visited_count = 1
    max_depth = 0

    order_rev = order[::-1]

    while stack:
        current = stack.pop()
        processed += 1
        if current.depth > max_depth:
            max_depth = current.depth

        if current.board == target:
            return current, visited_count, processed, max_depth

        if current.depth < 20:
            for nxt in get_neighbors(current, rows, cols, order_rev):
                if nxt.board not in visited or nxt.depth < visited[nxt.board]:
                    visited[nxt.board] = nxt.depth
                    stack.append(nxt)
                    visited_count += 1

    return None, visited_count, processed, max_depth


def hamming(board, target):
    return sum(1 for i in range(len(board)) if board[i] != 0 and board[i] != target[i])


def manhattan(board, target, rows, cols):
    dist = 0
    for i in range(len(board)):
        if board[i] != 0:
            tr, tc = target.index(board[i]) // cols, target.index(board[i]) % cols
            cr, cc = i // cols, i % cols
            dist += abs(tr - cr) + abs(tc - cc)
    return dist


def solve_astar(start_board, rows, cols, heuristic):
    target = get_target(rows, cols)
    start_node = Node(start_board, None, None, 0)

    tie_breaker = 0
    open_set = []
    heapq.heappush(open_set, (0, tie_breaker, start_node))

    g_scores = {start_board: 0}
    processed = 0
    visited_count = 1
    max_depth = 0

    order = "LUDR"

    while open_set:
        _, _, current = heapq.heappop(open_set)
        processed += 1
        if current.depth > max_depth:
            max_depth = current.depth

        if current.board == target:
            return current, visited_count, processed, max_depth

        for nxt in get_neighbors(current, rows, cols, order):
            tentative_g = current.depth + 1

            if nxt.board not in g_scores or tentative_g < g_scores[nxt.board]:
                g_scores[nxt.board] = tentative_g

                if heuristic == "hamm":
                    h = hamming(nxt.board, target)
                else:
                    h = manhattan(nxt.board, target, rows, cols)

                f = tentative_g + h
                tie_breaker += 1
                heapq.heappush(open_set, (f, tie_breaker, nxt))
                visited_count += 1

    return None, visited_count, processed, max_depth


def main():
    if len(sys.argv) < 6:
        return

    algo = sys.argv[1]
    param = sys.argv[2]
    file_in = sys.argv[3]
    file_sol = sys.argv[4]
    file_stats = sys.argv[5]

    with open(file_in, 'r') as f:
        lines = f.readlines()
        rows, cols = map(int, lines[0].split())
        board = []
        for line in lines[1:]:
            board.extend(list(map(int, line.split())))
        start_board = tuple(board)

    start_time = time.perf_counter()

    if algo == "bfs":
        res, v, p, d = solve_bfs(start_board, rows, cols, param)
    elif algo == "dfs":
        res, v, p, d = solve_dfs(start_board, rows, cols, param)
    elif algo == "astr":
        res, v, p, d = solve_astar(start_board, rows, cols, param)
    else:
        return

    elapsed = (time.perf_counter() - start_time) * 1000.0

    path = []
    if res:
        curr = res
        while curr.parent:
            path.append(curr.move)
            curr = curr.parent
        path.reverse()
        path_str = "".join(path)
        path_len = len(path)
    else:
        path_str = ""
        path_len = -1

    with open(file_sol, 'w') as f:
        if path_len == -1:
            f.write("-1\n")
        else:
            f.write(f"{path_len}\n{path_str}\n")

    with open(file_stats, 'w') as f:
        f.write(f"{path_len}\n{v}\n{p}\n{d}\n{elapsed:.3f}\n")


if __name__ == "__main__":
    main()