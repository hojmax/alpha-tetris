"""
Tetris game with Rust backend and Pygame visualization.
Implements proper SRS rotation and DAS/ARR for responsive controls.
"""

import pygame

from tetris_core.tetris_core import TetrisEnv
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PIECE_COLORS,
)

# Constants
CELL_SIZE = 30
LEFT_SIDEBAR_WIDTH = 150
RIGHT_SIDEBAR_WIDTH = 150
TOP_PADDING = 40
BOTTOM_PADDING = 20

WINDOW_WIDTH = LEFT_SIDEBAR_WIDTH + BOARD_WIDTH * CELL_SIZE + RIGHT_SIDEBAR_WIDTH
WINDOW_HEIGHT = TOP_PADDING + BOARD_HEIGHT * CELL_SIZE + BOTTOM_PADDING

# Piece slot spacing (hold / queue)
SLOT_HEIGHT = 50
SLOT_GAP = 16

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (80, 80, 80)
GRID_COLOR = (40, 40, 40)
BORDER_COLOR = (80, 80, 80)
LABEL_COLOR = (140, 140, 140)


class TetrisGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.label_font = pygame.font.Font(None, 22)
        self.stats_font = pygame.font.Font(None, 22)

        self.env = TetrisEnv(BOARD_WIDTH, BOARD_HEIGHT)
        self.fall_time = 0
        self.paused = False

        # DAS (Delayed Auto Shift) and ARR (Auto Repeat Rate) settings
        self.das = 100
        self.arr = 0

        # Soft drop settings
        self.soft_drop_das = 0
        self.soft_drop_arr = 25
        self.soft_drop_factor = 20

        # Key state tracking for DAS/ARR
        self.key_states = {}
        self.down_held = False

        # Mini block size for sidebar pieces
        self.mini_cell_size = 20

        # Move inspection mode
        self.inspect_mode = False
        self.placements = []
        self.placement_index = 0

    def _piece_color(self, piece_type: int) -> tuple[int, int, int]:
        return PIECE_COLORS[piece_type]

    def get_fall_speed(self):
        """Get fall speed in milliseconds (constant gravity)."""
        return 1000

    def handle_events(self):
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

                if event.key == pygame.K_p:
                    self.paused = not self.paused
                    continue

                if event.key == pygame.K_r:
                    self.env.reset()
                    self.fall_time = 0
                    continue

                if event.key == pygame.K_m and not self.env.game_over:
                    self.inspect_mode = not self.inspect_mode
                    if self.inspect_mode:
                        # Get all placements for current piece
                        self.placements = self.env.get_possible_placements()
                        self.placement_index = 0
                    continue

                if self.paused or self.env.game_over:
                    continue

                # In inspect mode, left/right cycle through placements
                if self.inspect_mode:
                    if event.key == pygame.K_LEFT:
                        if self.placements:
                            self.placement_index = (self.placement_index - 1) % len(
                                self.placements
                            )
                    elif event.key == pygame.K_RIGHT:
                        if self.placements:
                            self.placement_index = (self.placement_index + 1) % len(
                                self.placements
                            )
                    continue

                if event.key == pygame.K_LEFT:
                    self.env.move_left()
                    self.key_states[pygame.K_LEFT] = {
                        "pressed_time": current_time,
                        "das_charged": False,
                        "last_move_time": current_time,
                    }

                elif event.key == pygame.K_RIGHT:
                    self.env.move_right()
                    self.key_states[pygame.K_RIGHT] = {
                        "pressed_time": current_time,
                        "das_charged": False,
                        "last_move_time": current_time,
                    }

                elif event.key == pygame.K_DOWN:
                    self.down_held = True
                    self.env.move_down()
                    self.key_states[pygame.K_DOWN] = {
                        "pressed_time": current_time,
                        "das_charged": False,
                        "last_move_time": current_time,
                    }

                elif event.key == pygame.K_UP or event.key == pygame.K_s:
                    self.env.rotate_cw()

                elif event.key == pygame.K_a:
                    self.env.rotate_ccw()

                elif event.key == pygame.K_w:
                    self.env.rotate_180()

                elif event.key == pygame.K_SPACE:
                    self.env.hard_drop()
                    self.fall_time = 0

                elif event.key == pygame.K_d:
                    self.env.hold()

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    if pygame.K_LEFT in self.key_states:
                        del self.key_states[pygame.K_LEFT]

                elif event.key == pygame.K_RIGHT:
                    if pygame.K_RIGHT in self.key_states:
                        del self.key_states[pygame.K_RIGHT]

                elif event.key == pygame.K_DOWN:
                    self.down_held = False
                    if pygame.K_DOWN in self.key_states:
                        del self.key_states[pygame.K_DOWN]

        if not self.paused and not self.env.game_over:
            self._handle_das_arr(current_time)

        return True

    def _handle_das_arr(self, current_time):
        """Handle Delayed Auto Shift and Auto Repeat Rate for movement keys."""
        # Determine which horizontal key to process (most recently pressed wins)
        horizontal_key = None
        if pygame.K_LEFT in self.key_states and pygame.K_RIGHT in self.key_states:
            # Both held - use most recently pressed
            left_time = self.key_states[pygame.K_LEFT]["pressed_time"]
            right_time = self.key_states[pygame.K_RIGHT]["pressed_time"]
            horizontal_key = pygame.K_LEFT if left_time > right_time else pygame.K_RIGHT
        elif pygame.K_LEFT in self.key_states:
            horizontal_key = pygame.K_LEFT
        elif pygame.K_RIGHT in self.key_states:
            horizontal_key = pygame.K_RIGHT

        if horizontal_key:
            state = self.key_states[horizontal_key]
            time_held = current_time - state["pressed_time"]
            time_since_move = current_time - state["last_move_time"]

            if not state["das_charged"]:
                if time_held >= self.das:
                    state["das_charged"] = True
                    if horizontal_key == pygame.K_LEFT:
                        self.env.move_left()
                    else:
                        self.env.move_right()
                    state["last_move_time"] = current_time
            else:
                if self.arr == 0:
                    if horizontal_key == pygame.K_LEFT:
                        while self.env.move_left():
                            pass
                    else:
                        while self.env.move_right():
                            pass
                    state["last_move_time"] = current_time
                elif time_since_move >= self.arr:
                    if horizontal_key == pygame.K_LEFT:
                        self.env.move_left()
                    else:
                        self.env.move_right()
                    state["last_move_time"] = current_time

        if pygame.K_DOWN in self.key_states:
            state = self.key_states[pygame.K_DOWN]
            time_held = current_time - state["pressed_time"]
            time_since_move = current_time - state["last_move_time"]

            if not state["das_charged"]:
                if time_held >= self.soft_drop_das:
                    state["das_charged"] = True
                    self.env.move_down()
                    state["last_move_time"] = current_time
            else:
                if time_since_move >= self.soft_drop_arr:
                    self.env.move_down()
                    state["last_move_time"] = current_time

    def update(self, dt):
        if self.paused or self.env.game_over or self.inspect_mode:
            return

        self.env.update_lock_delay(dt)
        self.fall_time += dt
        fall_speed = self.get_fall_speed()

        if self.down_held:
            fall_speed = fall_speed // self.soft_drop_factor

        if self.fall_time >= fall_speed:
            self.env.tick()
            self.fall_time = 0

    def draw_board(self):
        board_x = LEFT_SIDEBAR_WIDTH
        board_y = TOP_PADDING

        # Draw board background
        board_rect = pygame.Rect(
            board_x, board_y, BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE
        )
        pygame.draw.rect(self.screen, BLACK, board_rect)

        # Draw grid lines
        for x in range(BOARD_WIDTH + 1):
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (board_x + x * CELL_SIZE, board_y),
                (board_x + x * CELL_SIZE, board_y + BOARD_HEIGHT * CELL_SIZE),
            )
        for y in range(BOARD_HEIGHT + 1):
            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (board_x, board_y + y * CELL_SIZE),
                (board_x + BOARD_WIDTH * CELL_SIZE, board_y + y * CELL_SIZE),
            )

        # Draw border around the board
        pygame.draw.rect(self.screen, BORDER_COLOR, board_rect, 2)

        # Draw locked pieces
        board = self.env.get_board()
        board_piece_types = self.env.get_board_piece_types()

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if board[y][x] != 0:
                    color_idx = board_piece_types[y][x]
                    if color_idx is not None:
                        color = self._piece_color(color_idx)
                    else:
                        color = GRAY

                    self.draw_cell(
                        board_x + x * CELL_SIZE, board_y + y * CELL_SIZE, color
                    )

        if self.inspect_mode:
            # Draw placement ghost in inspect mode
            if self.placements:
                placement = self.placements[self.placement_index]
                piece = placement.piece
                color = self._piece_color(piece.piece_type)
                for x, y in piece.get_cells():
                    if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                        rect = pygame.Rect(
                            board_x + x * CELL_SIZE + 1,
                            board_y + y * CELL_SIZE + 1,
                            CELL_SIZE - 2,
                            CELL_SIZE - 2,
                        )
                        pygame.draw.rect(self.screen, color, rect, 2)
        else:
            # Draw ghost piece (outline only)
            ghost = self.env.get_ghost_piece()
            if ghost:
                color = self._piece_color(ghost.piece_type)
                for x, y in ghost.get_cells():
                    if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                        rect = pygame.Rect(
                            board_x + x * CELL_SIZE + 1,
                            board_y + y * CELL_SIZE + 1,
                            CELL_SIZE - 2,
                            CELL_SIZE - 2,
                        )
                        pygame.draw.rect(self.screen, color, rect, 2)

            # Draw current piece
            piece = self.env.get_current_piece()
            if piece:
                color = self._piece_color(piece.piece_type)
                for x, y in piece.get_cells():
                    if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                        self.draw_cell(
                            board_x + x * CELL_SIZE, board_y + y * CELL_SIZE, color
                        )

    def draw_cell(self, px, py, color):
        """Draw a single cell (flat style) at pixel coordinates."""
        rect = pygame.Rect(px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2)
        pygame.draw.rect(self.screen, color, rect)

    def draw_mini_piece(
        self,
        piece,
        center_x: int,
        center_y: int,
        color_override: tuple[int, int, int] | None = None,
    ):
        """Draw a mini piece centered at the given position."""
        cells = piece.get_cells()
        color = color_override or self._piece_color(piece.piece_type)

        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        piece_width = (max_x - min_x + 1) * self.mini_cell_size
        piece_height = (max_y - min_y + 1) * self.mini_cell_size

        start_x = center_x - piece_width // 2
        start_y = center_y - piece_height // 2

        for cx, cy in cells:
            rect = pygame.Rect(
                start_x + (cx - min_x) * self.mini_cell_size,
                start_y + (cy - min_y) * self.mini_cell_size,
                self.mini_cell_size - 1,
                self.mini_cell_size - 1,
            )
            pygame.draw.rect(self.screen, color, rect)

    def draw_sidebar(self):
        left_cx = LEFT_SIDEBAR_WIDTH // 2

        # --- Left sidebar: HOLD label + piece ---
        hold_label = self.label_font.render("HOLD", True, LABEL_COLOR)
        hold_label_rect = hold_label.get_rect(center=(left_cx, TOP_PADDING + 12))
        self.screen.blit(hold_label, hold_label_rect)

        hold_piece = self.env.get_hold_piece()
        if hold_piece:
            color = None
            if self.env.is_hold_used():
                base = self._piece_color(hold_piece.piece_type)
                color = (base[0] // 3, base[1] // 3, base[2] // 3)
            self.draw_mini_piece(
                hold_piece,
                left_cx,
                TOP_PADDING + 28 + SLOT_HEIGHT // 2,
                color_override=color,
            )

        # --- Stats under hold piece ---
        stats_y = TOP_PADDING + 28 + SLOT_HEIGHT + 16
        stats_x = left_cx - 40
        b2b = "Yes" if self.env.back_to_back else "No"
        stats = [
            ("ATK", str(self.env.attack)),
            ("Lines", str(self.env.lines_cleared)),
            ("Combo", str(self.env.combo)),
            ("B2B", b2b),
        ]
        for label, value in stats:
            text = self.stats_font.render(f"{label}: {value}", True, LABEL_COLOR)
            self.screen.blit(text, (stats_x, stats_y))
            stats_y += 20

        # --- Right sidebar: NEXT label + panels ---
        right_x = LEFT_SIDEBAR_WIDTH + BOARD_WIDTH * CELL_SIZE
        right_cx = right_x + RIGHT_SIDEBAR_WIDTH // 2

        next_label = self.label_font.render("NEXT", True, LABEL_COLOR)
        next_label_rect = next_label.get_rect(center=(right_cx, TOP_PADDING + 12))
        self.screen.blit(next_label, next_label_rect)

        next_pieces = self.env.get_next_pieces(5)
        slot_start_y = TOP_PADDING + 28
        for i, piece in enumerate(next_pieces):
            slot_cy = slot_start_y + i * (SLOT_HEIGHT + SLOT_GAP) + SLOT_HEIGHT // 2
            self.draw_mini_piece(piece, right_cx, slot_cy)

    def draw_overlay(self):
        if self.inspect_mode:
            # Draw placement counter above the board (in padding area)
            board_x = LEFT_SIDEBAR_WIDTH
            total = len(self.placements)
            if total > 0:
                counter_text = f"{self.placement_index + 1}/{total}"
            else:
                counter_text = "0/0"
            counter_surface = self.font.render(counter_text, True, WHITE)
            counter_rect = counter_surface.get_rect(
                center=(board_x + BOARD_WIDTH * CELL_SIZE // 2, TOP_PADDING // 2)
            )
            self.screen.blit(counter_surface, counter_rect)
        elif self.paused:
            self.draw_message("PAUSED", "Press P to continue")
        elif self.env.game_over:
            self.draw_message(
                "GAME OVER", f"Attack: {self.env.attack} | Press R to restart"
            )

    def draw_message(self, title, subtitle):
        board_x = LEFT_SIDEBAR_WIDTH
        board_y = TOP_PADDING
        board_center_x = board_x + BOARD_WIDTH * CELL_SIZE // 2
        board_center_y = board_y + BOARD_HEIGHT * CELL_SIZE // 2

        overlay = pygame.Surface(
            (BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (board_x, board_y))

        title_text = self.font.render(title, True, WHITE)
        title_rect = title_text.get_rect(center=(board_center_x, board_center_y - 20))
        self.screen.blit(title_text, title_rect)

        small_font = pygame.font.Font(None, 24)
        subtitle_text = small_font.render(subtitle, True, WHITE)
        subtitle_rect = subtitle_text.get_rect(
            center=(board_center_x, board_center_y + 20)
        )
        self.screen.blit(subtitle_text, subtitle_rect)

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_board()
        self.draw_sidebar()
        self.draw_overlay()
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60)

            running = self.handle_events()
            self.update(dt)
            self.draw()

        pygame.quit()


def main():
    game = TetrisGame()
    game.run()


if __name__ == "__main__":
    main()
