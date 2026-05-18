import os
import random
import numpy as np


class Puzzle:
    UP    = (-1,  0)
    DOWN  = ( 1,  0)
    LEFT  = ( 0, -1)
    RIGHT = ( 0,  1)
    DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            for j in range(cols):
                self.board[i][j] = i * cols + j + 1
        self.board[rows - 1][cols - 1] = 0
        self.blank = (rows - 1, cols - 1)

    # ------------------------------------------------------------------ #
    # Reprezentacja                                                         #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        return str(self.board)

    def __getitem__(self, index):
        return self.board[index]

    # ------------------------------------------------------------------ #
    # Ruch / tasowanie                                                      #
    # ------------------------------------------------------------------ #

    def move(self, direction: tuple) -> bool:
        """Przesuwa puste pole w podanym kierunku. Zwraca True przy sukcesie."""
        r, c = self.blank
        nr, nc = r + direction[0], c + direction[1]
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return False
        self.board[r][c], self.board[nr][nc] = self.board[nr][nc], self.board[r][c]
        self.blank = (nr, nc)
        return True

    def shuffle(self, steps: int = 1000) -> None:
        """Tasuje planszę wykonując losowe legalne ruchy."""
        for _ in range(steps):
            self.move(random.choice(self.DIRECTIONS))

    def is_solved(self) -> bool:
        """Sprawdza, czy plansza jest w stanie docelowym."""
        expected = list(range(1, self.rows * self.cols)) + [0]
        return list(self.board.flatten()) == expected

    def as_tuple(self) -> tuple:
        """Zwraca stan planszy jako niemutowalną krotkę (dla solverów)."""
        return tuple(self.board.flatten().tolist())

    # ------------------------------------------------------------------ #
    # Zapis / odczyt                                                        #
    # ------------------------------------------------------------------ #

    def save(self, filepath: str) -> None:
        """Zapisuje planszę do pliku tekstowego."""
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(filepath, "w") as f:
            f.write(f"{self.rows} {self.cols}\n")
            for row in range(self.rows):
                f.write(" ".join(str(int(self.board[row][c])) for c in range(self.cols)))
                f.write("\n")

    @classmethod
    def load(cls, filepath: str) -> "Puzzle":
        """Wczytuje planszę z pliku i zwraca obiekt Puzzle."""
        with open(filepath, "r") as f:
            rows, cols = map(int, f.readline().split())
            puzzle = cls.__new__(cls)
            puzzle.rows = rows
            puzzle.cols = cols
            data = []
            for _ in range(rows):
                data.append(list(map(int, f.readline().split())))
            puzzle.board = np.array(data, dtype=int)
            # znajdź pozycję pustego pola
            result = np.argwhere(puzzle.board == 0)
            puzzle.blank = tuple(result[0])
        return puzzle

    @classmethod
    def load_as_tuple(cls, filepath: str):
        """Wczytuje plik i zwraca (board_tuple, rows, cols) — do użycia w solverach."""
        puzzle = cls.load(filepath)
        return puzzle.as_tuple(), puzzle.rows, puzzle.cols