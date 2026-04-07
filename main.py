import Puzz
import pygame
import sys
#import ai

pygame.init()
row = int(input("Press Rows"))
column = int(input("Press Columns"))
BOARD_SIZE = 4
NANO_TO_SEC = 1000000000

# UI
size = width, height = 480, 480
screen = pygame.display.set_mode(size)
if column < row:
    pygame.display.set_caption('{} Puzzle'.format(BOARD_SIZE**2-1))
FPS = 30

# Fonts
tileFont = pygame.font.SysFont("", 72)

# Colors
black = (0,0,0)
borderColor = (92, 90, 86)
tileColor = (242, 197, 133)
fontColor = black

# # ai
# ai.init(BOARD_SIZE)
# aiMoveIndex = 0
# aiMoves = []

def gameLoop():
    clock = pygame.time.Clock()

    puzzle = Puzz.Puzz(row,column)
    puzzle.loadFromFile("generPuzz.txt")
    while True:
        # for event in pygame.event.get():
        #     handleInput(event, puzzle)

        drawPuzzle(puzzle)
        pygame.display.flip()
        clock.tick(FPS)
# тут змінити на іа і переревірити щоб працювала з матрицями н н
# def handleInput(event, puzzle):
#     global aiMoveIndex
#     global aiMoves
#
#     if event.type == pygame.QUIT: sys.exit()
#     elif event.type == pygame.KEYDOWN:
#         if event.key == pygame.K_r:
#             puzzle.shuffle()
#             aiMoveIndex = 0
#             aiMoves = []
#         elif event.key == pygame.K_h:
#             if len(aiMoves) == 0:
#                 aiMoves = ai.idaStar(puzzle)
#                 aiMoveIndex = 0
#
#             if len(aiMoves) != 0:
#                 puzzle.move(aiMoves[aiMoveIndex])
#                 if puzzle.checkWin():
#                     aiMoveIndex = 0
#                     aiMoves = []
#                 else:
#                     aiMoveIndex += 1
#
#     elif event.type == pygame.MOUSEBUTTONUP:
#         pos = pygame.mouse.get_pos()
#         puzzleCoord = (pos[1]*puzzle.board_columns//height,
#                         pos[0]*puzzle.board_rows//width)
#         dir = (puzzleCoord[0] - puzzle.blankPos[0],
#                 puzzleCoord[1] - puzzle.blankPos[1])
#
#         if dir == puzzle.RIGHT:
#             puzzle.move(puzzle.RIGHT)
#         elif dir == puzzle.LEFT:
#             puzzle.move(puzzle.LEFT)
#         elif dir == puzzle.DOWN:
#             puzzle.move(puzzle.DOWN)
#         elif dir == puzzle.UP:
#             puzzle.move(puzzle.UP)


def drawPuzzle(puzzle):
    screen.fill(black)

    tile_w = width / puzzle.board_columns
    tile_h = height / puzzle.board_rows

    for i in range(puzzle.board_rows):
        for j in range(puzzle.board_columns):
            value = puzzle.board[i][j]

            currentTileColor = tileColor
            numberText = str(int(value))

            if value == 0:
                currentTileColor = borderColor
                numberText = ''

            rect = pygame.Rect(
                j * tile_w,
                i * tile_h,
                tile_w,
                tile_h
            )

            pygame.draw.rect(screen, currentTileColor, rect)
            pygame.draw.rect(screen, borderColor, rect, 1)

            fontImg = tileFont.render(numberText, True, fontColor)

            screen.blit(
                fontImg,
                (
                    j * tile_w + (tile_w - fontImg.get_width()) / 2,
                    i * tile_h + (tile_h - fontImg.get_height()) / 2
                )
            )
if __name__ =="__main__":
    gameLoop()