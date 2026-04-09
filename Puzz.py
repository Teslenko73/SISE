import numpy as np
class Puzz:

    def __init__(self, board_rows, board_columns ):
        self.board_rows = board_rows
        self.board_columns = board_columns
        self.board = np.zeros((board_rows, board_columns))
        self.blankPos = (board_rows - 1, board_columns - 1)
        self.board = list(range(1, board_rows * board_columns)) + [0]

    def __str__(self):
        return str(self.board)


    def __getitem__(self,index):
        return self.board[index]


    def get_board(self):
        return self.board
