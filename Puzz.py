from random import choice
import numpy as np
class Puzz:
    UP = (1,0)
    DOWN = (-1,0)
    LEFT = (0,-1)
    RIGHT = (0,1)

    DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

    def __init__(self, board_rows, board_columns ):
        self.board_rows = board_rows
        self.board_columns = board_columns
        self.board = np.zeros((board_rows, board_columns))
        self.blankPos = (board_rows - 1, board_columns - 1)


        for i in range(board_rows):
            for j in range(board_columns):
                self.board[i][j] = i*board_columns + j+1
        self.board[self.blankPos[0]][self.blankPos[1]] = 0
        self.shuffle()


    def __str__(self):
        return str(self.board)


    def __getitem__(self,index):
        return self.board[index]


    def shuffle(self):
        nShuffle = 1000

        for _ in range(nShuffle):
            dir = choice(self.DIRECTIONS)
            self.move(dir)

    def move(self, direction):
        new_x = self.blankPos[0] + direction[0]
        new_y = self.blankPos[1] + direction[1]

        if new_x < 0 or new_x >= self.board_rows or new_y < 0 or new_y >= self.board_columns:
            return False
        self.board[self.blankPos[0]][self.blankPos[1]] = self.board[new_x][new_y]
        self.board[new_x][new_y] = 0
        self.blankPos = (new_x, new_y)
        self.WriteInFile("bin/generPuzz.txt")
        return True

    def checkWin(self):
        for row in range(self.board_rows):
            for col in range(self.board_columns):
                if self.board[row][col] != i * self.board_rows + j + 1 and self.board[row][col] != 0:
                    return False
        return True

    def WriteInFile(self, filename):
        with open(filename, 'w') as file:
            file.write(f"{self.board_rows} {self.board_columns}\n")

            for row in range(self.board_rows):
                for col in range(self.board_columns):
                    file.write(str(int(self.board[row][col])) + " ")
                file.write("\n")

    def loadFromFile(self, filename="generPuzz.txt"):
        with open(filename, 'r') as file:
            first_line = file.readline().split()
            self.board_rows = int(first_line[0])
            self.board_columns = int(first_line[1])

            self.board = []

            for i in range(self.board_rows):
                row = list(map(int, file.readline().split()))
                self.board.append(row)

                for j in range(self.board_columns):
                    if row[j] == 0:
                        self.blankPos = (i, j)