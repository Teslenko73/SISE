import time
from collections import deque

class Node:
    def __init__(self, board, parent=None, move=None, depth=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.depth = depth

    def __hash__(self):
        return hash(self.board)

    def __eq__(self, other):
        return self.board == other.board

def get_neighbors(node, size):
    board = list(node.board)
    z_idx = board.index(0)
    r, c = z_idx // size, z_idx % size
    neighbors = []

    moves = [(-1, 0, 'U'), (1, 0, 'D'), (0, -1, 'L'), (0, 1, 'R')]

    for dr, dc, move_name in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            new_idx = nr * size + nc
            new_board = board[:]
            new_board[z_idx], new_board[new_idx] = new_board[new_idx], new_board[z_idx]
            neighbors.append(Node(tuple(new_board), node, move_name, node.depth + 1))
    return neighbors

def solve_bfs(start_board, target_board):
    size = int(len(start_board) ** 0.5)
    start_node = Node(start_board)
    queue = deque([start_node])
    visited = {start_board}
    nodes_visited = 0
    start_time = time.time()

    while queue:
        current = queue.popleft()
        nodes_visited += 1

        if current.board == target_board:
            path = []
            while current.parent:
                path.append(current.move)
                current = current.parent
            return "".join(path[::-1]), nodes_visited, time.time() - start_time

        for next_node in get_neighbors(current, size):
            if next_node.board not in visited:
                visited.add(next_node.board)
                queue.append(next_node)

    return None, nodes_visited, time.time() - start_time

if __name__ == "__main__":
    start = (1, 2, 3, 4, 0, 6, 7, 5, 8)
    goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    path, count, duration = solve_bfs(start, goal)
    print(f"Path: {path}")
    print(f"Nodes: {count}")
    print(f"Time: {duration:.4f}s")


def solve_dfs(start_board, target_board, limit=20):
    size = int(len(start_board) ** 0.5)
    start_node = Node(start_board)
    stack = [start_node]
    visited = {start_board: 0}
    nodes_visited = 0
    start_time = time.time()

    while stack:
        current = stack.pop()
        nodes_visited += 1

        if current.board == target_board:
            path = []
            while current.parent:
                path.append(current.move)
                current = current.parent
            return "".join(path[::-1]), nodes_visited, time.time() - start_time

        if current.depth < limit:
            for next_node in get_neighbors(current, size):
                if next_node.board not in visited or next_node.depth < visited[next_node.board]:
                    visited[next_node.board] = next_node.depth
                    stack.append(next_node)

    return None, nodes_visited, time.time() - start_time