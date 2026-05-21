"""
gui.py
------
Wizualizacja rozwiązania układanki N-puzzle w Pygame.

Publiczne API:
    run_gui(start_board, rows, cols, moves, tile_size=120, fps=4)

    start_board  – krotka reprezentująca stan startowy
    rows, cols   – wymiary planszy
    moves        – lista ruchów ['R', 'D', 'L', ...] z solvera
    tile_size    – rozmiar kafelka w pikselach
    fps          – szybkość animacji (ruchów na sekundę)

Sterowanie podczas wyświetlania:
    SPACJA  – pauza / wznowienie
    →       – krok naprzód (gdy pauza)
    ←       – krok wstecz (gdy pauza)
    R       – restart (powrót do stanu startowego)
    ESC / Q – wyjście
"""

import sys
import pygame


# ================================================================== #
# Stałe wizualne                                                       #
# ================================================================== #

BG_COLOR     = (30,  30,  30)
BORDER_COLOR = (92,  90,  86)
TILE_COLOR   = (242, 197, 133)
BLANK_COLOR  = (60,  60,  60)
TEXT_COLOR   = (30,  30,  30)
HEADER_COLOR = (200, 200, 200)
PAUSE_COLOR  = (255, 200,  50)
DONE_COLOR   = (100, 220, 100)

PADDING = 4          # px między kafelkami
HEADER  = 50         # px na pasek górny z informacjami


# ================================================================== #
# Pomocnicze                                                           #
# ================================================================== #

def _apply_move(board: list[int], move: str, cols: int) -> list[int]:
    """Zwraca nową planszę po wykonaniu ruchu (mutuje kopię)."""
    DELTA = {"U": -cols, "D": cols, "L": -1, "R": 1}
    z = board.index(0)
    t = z + DELTA[move]
    nb = board[:]
    nb[z], nb[t] = nb[t], nb[z]
    return nb


def _build_states(start_board: tuple, moves: list[str], cols: int) -> list[list[int]]:
    """Buduje listę wszystkich stanów planszy (stan[0] = start, stan[-1] = cel)."""
    states = [list(start_board)]
    for m in moves:
        states.append(_apply_move(states[-1], m, cols))
    return states


# ================================================================== #
# Rysowanie                                                            #
# ================================================================== #

def _draw_board(surface: pygame.Surface,
                board: list[int],
                rows: int, cols: int,
                tile_size: int,
                font: pygame.font.Font) -> None:
    for idx, value in enumerate(board):
        r, c = idx // cols, idx % cols
        x = c * (tile_size + PADDING) + PADDING
        y = r * (tile_size + PADDING) + PADDING + HEADER

        color = BLANK_COLOR if value == 0 else TILE_COLOR
        rect  = pygame.Rect(x, y, tile_size, tile_size)
        pygame.draw.rect(surface, color, rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, rect, width=1, border_radius=8)

        if value != 0:
            text_surf = font.render(str(value), True, TEXT_COLOR)
            tx = x + (tile_size - text_surf.get_width())  // 2
            ty = y + (tile_size - text_surf.get_height()) // 2
            surface.blit(text_surf, (tx, ty))


def _draw_header(surface: pygame.Surface,
                 step: int, total: int,
                 paused: bool, done: bool,
                 small_font: pygame.font.Font,
                 width: int) -> None:
    pygame.draw.rect(surface, (20, 20, 20), (0, 0, width, HEADER))

    if done:
        label = "Rozwiązano!"
        color = DONE_COLOR
    elif paused:
        label = "PAUZA  (← → = krok,  SPACJA = wznów)"
        color = PAUSE_COLOR
    else:
        label = "Odtwarzanie...  (SPACJA = pauza)"
        color = HEADER_COLOR

    info  = small_font.render(f"Krok {step}/{total}   {label}", True, color)
    ctrl  = small_font.render("R = restart   ESC / Q = wyjście", True, (120, 120, 120))
    surface.blit(info,  (10, 8))
    surface.blit(ctrl,  (10, 30))


# ================================================================== #
# Główna pętla                                                          #
# ================================================================== #

def run_gui(start_board: tuple,
            rows: int, cols: int,
            moves: list[str],
            tile_size: int = 120,
            fps: int = 4) -> None:
    """
    Uruchamia okno Pygame i animuje rozwiązanie układanki krok po kroku.
    """
    pygame.init()

    win_w = cols * (tile_size + PADDING) + PADDING
    win_h = rows * (tile_size + PADDING) + PADDING + HEADER
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"{rows}×{cols} Puzzle — rozwiązanie")

    tile_font  = pygame.font.SysFont("", max(20, tile_size // 2))
    small_font = pygame.font.SysFont("", 18)
    clock      = pygame.time.Clock()

    states = _build_states(start_board, moves, cols)
    total  = len(moves)

    step        = 0   # aktualnie wyświetlany stan
    paused      = False
    auto_timer  = 0.0  # czas [s] od ostatniego automatycznego kroku

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # delta time w sekundach

        # ---- zdarzenia ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    auto_timer = 0.0

                elif event.key == pygame.K_r:
                    step       = 0
                    paused     = False
                    auto_timer = 0.0

                elif event.key == pygame.K_RIGHT and paused:
                    step = min(step + 1, total)

                elif event.key == pygame.K_LEFT and paused:
                    step = max(step - 1, 0)

        # ---- automatyczne przejście ----
        done = (step == total)
        if not paused and not done:
            auto_timer += dt
            if auto_timer >= 1.0 / fps:
                step      += 1
                auto_timer = 0.0

        # ---- rysowanie ----
        screen.fill(BG_COLOR)
        _draw_board(screen, states[step], rows, cols, tile_size, tile_font)
        _draw_header(screen, step, total, paused or done, done, small_font, win_w)
        pygame.display.flip()

    pygame.quit()