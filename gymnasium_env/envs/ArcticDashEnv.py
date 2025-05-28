import gymnasium as gym
from gymnasium import spaces, utils
import numpy as np
import pygame
from io import StringIO
from contextlib import closing
import os
from PIL import Image

MOVE_LEFT = 0
MOVE_DOWN = 1
MOVE_RIGHT = 2
MOVE_UP = 3
JUMP_LEFT = 4
JUMP_DOWN = 5
JUMP_RIGHT = 6
JUMP_UP = 7

MAPS = {
    "e0": [
        "FFFFFFFFFFFFFFFFFFFF",
        "FFHHHHHHHFFFFHHHHFFF",
        "FFHFFFFFFHFFHFFFFFFH",
        "SFFHFFFFFHFFHFFFFFFG",
        "FFHFFFFFFHFFHFFFFFFH",
        "FFHHHHHHHFFFFHHHHFFF",
        "FFFFFFFFFFFFFFFFFFFF",
    ],
    "e1": [
        "SVFVVVVV",
        "HWFFFWFF",
        "FFWHFFWF",
        "FVFWFHFF",
        "FFWHWWFF",
        "FHWHFWHF",
        "WHFFHFHH",
        "FFFHWFWG",
    ],
    "e2": [
        "WWFHFWFHFHWWWFF",
        "WHFFFHWFFFHHWFF",
        "WHWVFHVVWHWFWHF",
        "SHHFHWFWWWWWHHG",
        "WHFFWFFWFWFFFHF",
        "WHWHFFVFWHFWFFF",
        "WWWVFHFFWWFHFFW",
    ],
    "e3": [
        "SFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH",
        "FFFWWWVFWWHFWWWHFWWWHFWWWHFWWWHFWWWHWFFH",
        "HFFFFFWFFFFHFFFFFWFFFFHFFFFFWFFFFHFFFFFH",
        "HHHFWWHHHVFWWHHHVFWWHHHVFWWHHHVFWWHHHHHH",
        "HFWWWFWVFWWWHFWVFWWWHFWVFWWWHFWVFWWWFHHH",
        "HFFFFFWVFFFHFFFFFWVFFFHFFFFFWVFFFHFFGFHH",
        "HHHFWWHHHVFWWHHHVFWWHHHVFWWHHHVFWWWWHHHH",
        "HFWWWVFWWWHFWWWHFWWWHFWWWHFWWWHFWWHWHWFH",
        "HFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFH",
        "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
    ],
    "e5": [
        "WWWWWSWWWWW",
        "FWWWWWWWWWF",
        "FHHHHHHHHHF",
        "FWFHHHHHFWW",
        "FWWFHHHFWWF",
        "WWHWWHWWHWH",
        "WWHHWWWWWWH",
        "WWFWWWHHWWW",
        "WWWFWWWWWHW",
        "HHHHHHWWWFW",
        "WFFWWWFWWFW",
        "WWHHWWFHHHH",
        "WWWFHWWWFFW",
        "FWHHHHHHWWH",
        "WFFHHHHWWFW",
        "FFWWHHWWFHW",
        "WFWWHWHWFWW",
        "WHHWWWFWWFW",
        "WFWFWWHHHHW",
        "HHHWWFWFHWH",
        "WFFWWWWFFWH",
        "WWHHHWWFHWW",
        "WWFHHHWWWFF",
        "HWWFHHHWFWW",
        "WFWWWFWWFWW",
        "HHHHWHFHHFH",
        "WFFHWHWHHHH",
        "FFWHWHWWFFF",
        "HHHHWWWFHHW",
        "FFFHFWWWFFF",
        "FFFHHHHHHHF",
        "WFFHHHHFHFF",
        "WFFHWWWFHFW",
        "WHWHFFHHHFW",
        "HHWWFFFFWWH",
        "WWFFWFHWWHW",
        "WFHWWWFHWFH",
        "WHHHFHWFFHF",
        "WFWFHHHHWWH",
        "HWWFFGFWWFH",
    ],
}


class ArcticDashEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, desc=None, map_name="e5",
                 goal_reward=100_000.0, pickup_reward=1_000.0, hole_penalty=-200.0,
                 move_cost=-1.0, jump_cost=-7.5,
                 max_jumps=2,
                 interactive_mode=False):

        if desc is None and map_name is None:
            desc = MAPS["e5"]
        elif desc is None:
            if map_name in MAPS:
                desc = MAPS[map_name]
            else:
                raise ValueError(f"Nazwa mapy '{map_name}' nie znaleziona. Dostępne: {list(MAPS.keys())}")

        self.desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = self.desc.shape

        self.goal_reward = goal_reward
        self.pickup_reward = pickup_reward
        self.hole_penalty = hole_penalty
        self.move_cost = move_cost
        self.jump_cost = jump_cost
        self.max_jumps = max_jumps

        self.jumps_left = 0
        self.has_treasure = False
        self.current_map_state = np.copy(self.desc)

        self.n_actions = 8
        self.n_states = self.nrow * self.ncol
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states)

        initial_state_coord = np.argwhere(self.desc == b'S')
        if initial_state_coord.size == 0:
            raise ValueError("Mapa musi zawierać punkt startowy 'S'.")
        self.initial_row, self.initial_col = initial_state_coord[0]
        self.initial_state_idx = self._to_s(self.initial_row, self.initial_col)

        goal_coords = np.argwhere(self.desc == b'G')
        if goal_coords.size == 0:
            raise ValueError("Mapa musi zawierać przynajmniej jeden cel 'G'.")
        self.goal_pos_row, self.goal_pos_col = goal_coords[0]

        self.current_row = 0
        self.current_col = 0
        self.s = 0

        self.render_mode = render_mode

        self.base_window_width = 850
        self.base_window_height = 600
        self.game_area_width_ratio = 0.7
        self.info_panel_width_ratio = 0.3

        self.game_area_width = int(self.base_window_width * self.game_area_width_ratio)
        self.game_area_height = self.base_window_height
        self.info_panel_width = int(self.base_window_width * self.info_panel_width_ratio)

        self.window_width = self.base_window_width
        self.window_height = self.base_window_height

        self.window = None
        self.clock = None
        self.game_font = None
        self.small_font = None
        self.lastaction = None
        self.last_step_info = {}

        self.assets_loaded = False
        self.scaled_sprites = {}
        self.original_sprites_pil = {}
        self.current_pix_square_size = 0

        self.sprite_files = {
            'S': "start_tile.png", 'G': "goal_tile.png", 'F': "ice_tile.png",
            'H': "hole_tile.png", 'W': "weak_ice_tile.png", 'V': "very_weak_ice_tile.png",
            'AGENT': "agent.png", 'TREASURE': "treasure_icon.png"
        }
        self.tile_colors = {
            b'S': (124, 252, 0), b'G': (255, 215, 0), b'F': (200, 220, 255),
            b'H': (50, 50, 100), b'W': (173, 216, 230), b'V': (224, 255, 255)
        }
        self.agent_color = (255, 100, 100)
        self.treasure_color = (255, 223, 0)

        self.is_fullscreen = False
        self.zoom_level = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 4.0
        self.zoom_step = 0.1

        self.camera_on_agent = True
        self.current_fps = self.metadata["render_fps"]
        self.min_fps = 2
        self.max_fps = 120
        self.fps_step = 4

        self.camera_offset_x = 0
        self.camera_offset_y = 0

        self.interactive_mode = interactive_mode
        self.path_taken = []
        self.game_is_over_manual_reset_pending = False

        if self.render_mode == "human":
            if not pygame.get_init():
                pygame.init()
            if not pygame.display.get_init():
                pygame.display.init()

        self.reset()

    def _toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        current_display_caption = pygame.display.get_caption()[0]

        pygame.display.quit()
        pygame.display.init()

        if self.is_fullscreen:
            info = pygame.display.Info()
            self.window_width = info.current_w
            self.window_height = info.current_h
            self.window = pygame.display.set_mode((self.window_width, self.window_height),
                                                  pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self.window_width = self.base_window_width
            self.window_height = self.base_window_height
            self.window = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)

        self.game_area_width = int(self.window_width * self.game_area_width_ratio)
        self.game_area_height = self.window_height
        self.info_panel_width = self.window_width - self.game_area_width
        pygame.display.set_caption(current_display_caption)

        self.assets_loaded = False

    def _adjust_zoom(self, direction):
        if direction > 0:
            self.zoom_level = min(self.max_zoom, self.zoom_level + self.zoom_step)
        elif direction < 0:
            self.zoom_level = max(self.min_zoom, self.zoom_level - self.zoom_step)
        self.assets_loaded = False

    def _adjust_fps(self, direction):
        if direction > 0:
            self.current_fps = min(self.max_fps, self.current_fps + self.fps_step)
        elif direction < 0:
            self.current_fps = max(self.min_fps, self.current_fps - self.fps_step)

    def _handle_input_events(self):
        if self.render_mode != "human" or not pygame.display.get_init():
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                import sys
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    self._adjust_zoom(1)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    self._adjust_zoom(-1)
                elif event.key == pygame.K_c:
                    self.camera_on_agent = not self.camera_on_agent
                elif event.key == pygame.K_PAGEUP:
                    self._adjust_fps(1)
                elif event.key == pygame.K_PAGEDOWN:
                    self._adjust_fps(-1)
                elif event.key == pygame.K_r:
                    if self.interactive_mode and self.game_is_over_manual_reset_pending:
                        self.reset()

            if event.type == pygame.VIDEORESIZE and not self.is_fullscreen:
                self.window_width = event.w
                self.window_height = event.h
                self.game_area_width = int(self.window_width * self.game_area_width_ratio)
                self.game_area_height = self.window_height
                self.info_panel_width = self.window_width - self.game_area_width

                current_display_caption = pygame.display.get_caption()[0]
                self.window = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
                pygame.display.set_caption(current_display_caption)

                self.assets_loaded = False

    def _load_and_scale_assets(self, size):
        self.scaled_sprites = {}
        self.original_sprites_pil = {}
        self.assets_loaded = True
        base_path = "assets"

        if not os.path.exists(base_path):
            self.assets_loaded = False
            return

        for key, filename in self.sprite_files.items():
            try:
                img_path = os.path.join(base_path, filename)
                if not os.path.exists(img_path):
                    self.scaled_sprites[key] = None
                    self.original_sprites_pil[key] = None
                    continue

                pil_img = Image.open(img_path).convert("RGBA")
                self.original_sprites_pil[key] = pil_img

                pygame_surface = pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode).convert_alpha()
                self.scaled_sprites[key] = pygame.transform.scale(pygame_surface, (size, size))

            except Exception as e:
                self.scaled_sprites[key] = None
                self.original_sprites_pil[key] = None

        self.current_pix_square_size = size

    def _to_s(self, row, col):
        return row * self.ncol + col

    def _to_rc(self, s):
        return s // self.ncol, s % self.ncol

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_row, self.current_col = self.initial_row, self.initial_col
        self.s = self._to_s(self.current_row, self.current_col)
        self.lastaction = None
        self.jumps_left = self.max_jumps
        self.has_treasure = False
        self.current_map_state = np.copy(self.desc)
        self.last_step_info = {}
        self.camera_offset_x = 0
        self.camera_offset_y = 0

        self.path_taken = [(self.current_row, self.current_col)]
        self.game_is_over_manual_reset_pending = False

        if self.render_mode == "human":
            self._render_frame()

        return self.s, {"prob": 1.0, "jumps_left": self.jumps_left, "has_treasure": self.has_treasure}

    def step(self, action):
        if self.interactive_mode and self.game_is_over_manual_reset_pending:
            if not hasattr(self, 'last_step_info') or not self.last_step_info:
                self.last_step_info = {"prob": 1.0, "jumps_left": self.jumps_left, "has_treasure": self.has_treasure}

            return self.s, 0.0, True, False, self.last_step_info

        self.lastaction = action
        reward = 0.0
        terminated = False
        current_step_info = {"prob": 1.0}
        prev_r, prev_c = self.current_row, self.current_col

        is_jump_action = JUMP_LEFT <= action <= JUMP_UP

        dr, dc = 0, 0
        if action == MOVE_LEFT:
            dc = -1
        elif action == MOVE_DOWN:
            dr = 1
        elif action == MOVE_RIGHT:
            dc = 1
        elif action == MOVE_UP:
            dr = -1
        elif action == JUMP_LEFT:
            dc = -2
        elif action == JUMP_DOWN:
            dr = 2
        elif action == JUMP_RIGHT:
            dc = 2
        elif action == JUMP_UP:
            dr = -2

        intended_r, intended_c = prev_r + dr, prev_c + dc
        agent_actually_moved = False
        new_agent_r, new_agent_c = prev_r, prev_c

        if is_jump_action:
            if self.jumps_left <= 0:
                reward = -400.0
                terminated = True
                current_step_info["error"] = "Brak skoków! Kara, koniec."
            else:
                is_valid_jump_path_and_landing = False
                if abs(dr) == 2 and dc == 0:
                    if 0 <= intended_r < self.nrow and intended_c == prev_c:
                        is_valid_jump_path_and_landing = True
                elif abs(dc) == 2 and dr == 0:
                    if 0 <= intended_c < self.ncol and intended_r == prev_r:
                        is_valid_jump_path_and_landing = True

                if is_valid_jump_path_and_landing:
                    reward = self.jump_cost
                    self.jumps_left -= 1
                    new_agent_r, new_agent_c = intended_r, intended_c
                    agent_actually_moved = True
                else:
                    reward = -400.0
                    terminated = True
                    current_step_info["error"] = "Niewłaściwy skok! Kara, koniec."
        else:
            if 0 <= intended_r < self.nrow and 0 <= intended_c < self.ncol:
                reward = self.move_cost
                new_agent_r, new_agent_c = intended_r, intended_c
                if new_agent_r != prev_r or new_agent_c != prev_c:
                    agent_actually_moved = True
            else:
                reward = -400.0
                terminated = True
                current_step_info["error"] = "Ruch poza mapę! Kara, koniec."

        self.current_row, self.current_col = new_agent_r, new_agent_c
        self.s = self._to_s(self.current_row, self.current_col)

        if agent_actually_moved:
            if not self.path_taken or self.path_taken[-1] != (self.current_row, self.current_col):
                self.path_taken.append((self.current_row, self.current_col))

        if terminated and reward == -400.0:
            current_step_info.update({"jumps_left": self.jumps_left, "has_treasure": self.has_treasure})
            self.last_step_info = current_step_info.copy()
            if self.interactive_mode:
                self.game_is_over_manual_reset_pending = True
            if self.render_mode == "human": self._render_frame()
            return self.s, reward, terminated, False, self.last_step_info

        if agent_actually_moved:
            r_check, c_check = self.current_row, self.current_col
            original_tile = self.desc[r_check, c_check]
            if original_tile != b'S' and original_tile != b'G':
                current_tile_on_map = self.current_map_state[r_check, c_check]
                action_type = "jump" if is_jump_action else "move"

                if action_type == "jump":
                    if current_tile_on_map == b'F':
                        self.current_map_state[r_check, c_check] = b'W'
                    elif current_tile_on_map == b'W':
                        self.current_map_state[r_check, c_check] = b'H'
                    elif current_tile_on_map == b'V':
                        self.current_map_state[r_check, c_check] = b'H'
                else:
                    if current_tile_on_map == b'F':
                        self.current_map_state[r_check, c_check] = b'W'
                    elif current_tile_on_map == b'W':
                        self.current_map_state[r_check, c_check] = b'V'
                    elif current_tile_on_map == b'V':
                        self.current_map_state[r_check, c_check] = b'H'

        effective_tile_now = self.current_map_state[self.current_row, self.current_col]

        if effective_tile_now == b'H':
            reward += self.hole_penalty
            terminated = True
        elif effective_tile_now == b'G':
            if not self.has_treasure:
                self.has_treasure = True
                reward += self.pickup_reward
        elif effective_tile_now == b'S':
            if self.has_treasure:
                reward += self.goal_reward
                terminated = True

        current_step_info.update({"jumps_left": self.jumps_left, "has_treasure": self.has_treasure})
        self.last_step_info = current_step_info.copy()

        if terminated:
            if self.interactive_mode:
                self.game_is_over_manual_reset_pending = True

        if self.render_mode == "human": self._render_frame()
        return self.s, reward, terminated, False, self.last_step_info

    def _render_frame(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return

        if self.render_mode == "human":
            if not pygame.get_init(): pygame.init()
            if not pygame.display.get_init(): pygame.display.init()

            if self.window is None:
                if self.is_fullscreen:
                    info = pygame.display.Info()
                    self.window_width, self.window_height = info.current_w, info.current_h
                    self.window = pygame.display.set_mode((self.window_width, self.window_height),
                                                          pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
                else:
                    self.window_width, self.window_height = self.base_window_width, self.base_window_height
                    self.window = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)

                self.game_area_width = int(self.window_width * self.game_area_width_ratio)
                self.game_area_height = self.window_height
                self.info_panel_width = self.window_width - self.game_area_width
                pygame.display.set_caption(f"ArcticDashEnv - {self.nrow}x{self.ncol}")

                try:
                    self.game_font = pygame.font.SysFont("Consolas", 26)
                    self.small_font = pygame.font.SysFont("Consolas", 18)
                except pygame.error as e:
                    self.game_font = pygame.font.Font(None, 28)
                    self.small_font = pygame.font.Font(None, 20)

                if self.clock is None:
                    self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self._handle_input_events()

        base_pix_square_size_w = self.game_area_width / self.ncol
        base_pix_square_size_h = self.game_area_height / self.nrow
        effective_pix_square_size = int(min(base_pix_square_size_w, base_pix_square_size_h) * self.zoom_level)
        if effective_pix_square_size <= 0: effective_pix_square_size = 1

        if not self.assets_loaded or self.current_pix_square_size != effective_pix_square_size:
            if effective_pix_square_size > 0:
                self._load_and_scale_assets(effective_pix_square_size)
            else:
                self.assets_loaded = False
                self.scaled_sprites = {}

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((50, 50, 80))

        game_surface = pygame.Surface((self.game_area_width, self.game_area_height))
        game_surface.fill((100, 140, 180))

        zoomed_map_width = self.ncol * effective_pix_square_size
        zoomed_map_height = self.nrow * effective_pix_square_size

        agent_map_x_on_zoomed_map = self.current_col * effective_pix_square_size
        agent_map_y_on_zoomed_map = self.current_row * effective_pix_square_size

        view_offset_x = 0
        view_offset_y = 0

        if self.camera_on_agent:
            agent_tile_center_x_on_zoomed_map = agent_map_x_on_zoomed_map + effective_pix_square_size / 2
            view_offset_x = (self.game_area_width / 2) - agent_tile_center_x_on_zoomed_map

            agent_tile_center_y_on_zoomed_map = agent_map_y_on_zoomed_map + effective_pix_square_size / 2
            view_offset_y = (self.game_area_height / 2) - agent_tile_center_y_on_zoomed_map

            if zoomed_map_width > self.game_area_width:
                view_offset_x = max(self.game_area_width - zoomed_map_width, min(0, view_offset_x))
            else:
                view_offset_x = (self.game_area_width - zoomed_map_width) / 2

            if zoomed_map_height > self.game_area_height:
                view_offset_y = max(self.game_area_height - zoomed_map_height, min(0, view_offset_y))
            else:
                view_offset_y = (self.game_area_height - zoomed_map_height) / 2

        else:
            if zoomed_map_width <= self.game_area_width:
                view_offset_x = (self.game_area_width - zoomed_map_width) / 2
            else:
                view_offset_x = self.camera_offset_x
                view_offset_x = max(self.game_area_width - zoomed_map_width, min(0, view_offset_x))

            if zoomed_map_height <= self.game_area_height:
                view_offset_y = (self.game_area_height - zoomed_map_height) / 2
            else:
                view_offset_y = self.camera_offset_y
                view_offset_y = max(self.game_area_height - zoomed_map_height, min(0, view_offset_y))

        for r_idx in range(self.nrow):
            for c_idx in range(self.ncol):
                tile_map_x = c_idx * effective_pix_square_size
                tile_map_y = r_idx * effective_pix_square_size

                screen_x = tile_map_x + view_offset_x
                screen_y = tile_map_y + view_offset_y

                if screen_x + effective_pix_square_size < 0 or screen_x > self.game_area_width or \
                        screen_y + effective_pix_square_size < 0 or screen_y > self.game_area_height:
                    continue

                rect = pygame.Rect(
                    screen_x, screen_y,
                    effective_pix_square_size, effective_pix_square_size,
                )
                char_byte = self.current_map_state[r_idx, c_idx]
                char_str = char_byte.decode('utf-8')
                sprite_to_draw = self.scaled_sprites.get(char_str) if self.assets_loaded else None

                if sprite_to_draw:
                    game_surface.blit(sprite_to_draw, rect)
                else:
                    color = self.tile_colors.get(char_byte, (128, 128, 128))
                    pygame.draw.rect(game_surface, color, rect)
                    if effective_pix_square_size > 2:
                        pygame.draw.rect(game_surface, tuple(min(255, x + 20) for x in color), rect, 1)
                        if char_byte == b'F':
                            pygame.draw.line(game_surface, (230, 240, 255), (rect.left + 2, rect.top + 2),
                                             (rect.right - 2, rect.bottom - 2), 1)
                            pygame.draw.line(game_surface, (230, 240, 255), (rect.right - 2, rect.top + 2),
                                             (rect.left + 2, rect.bottom - 2), 1)

                if char_byte == b'G' and not self.has_treasure:
                    treasure_sprite_on_goal = self.scaled_sprites.get('TREASURE') if self.assets_loaded else None
                    if treasure_sprite_on_goal:
                        icon_rect = treasure_sprite_on_goal.get_rect(center=rect.center)
                        game_surface.blit(treasure_sprite_on_goal, icon_rect)
                    elif effective_pix_square_size > 4:
                        pygame.draw.circle(game_surface, self.treasure_color, rect.center,
                                           effective_pix_square_size // 4)
                        if effective_pix_square_size > 8:
                            pygame.draw.circle(game_surface, tuple(max(0, x - 30) for x in self.treasure_color),
                                               rect.center, effective_pix_square_size // 4, 2)

        agent_screen_x = self.current_col * effective_pix_square_size + view_offset_x
        agent_screen_y = self.current_row * effective_pix_square_size + view_offset_y
        agent_base_rect = pygame.Rect(
            agent_screen_x, agent_screen_y,
            effective_pix_square_size, effective_pix_square_size
        )
        agent_sprite_to_draw = self.scaled_sprites.get('AGENT') if self.assets_loaded else None

        if agent_sprite_to_draw:
            game_surface.blit(agent_sprite_to_draw, agent_base_rect)
            if self.has_treasure:
                original_treasure_pil = self.original_sprites_pil.get('TREASURE')
                if original_treasure_pil:
                    try:
                        small_icon_size = effective_pix_square_size // 3
                        if small_icon_size > 0:
                            small_treasure_pil = original_treasure_pil.resize((small_icon_size, small_icon_size),
                                                                              Image.Resampling.LANCZOS)
                            small_treasure_surface = pygame.image.fromstring(small_treasure_pil.tobytes(),
                                                                             small_treasure_pil.size,
                                                                             small_treasure_pil.mode).convert_alpha()

                            icon_pos_x = agent_base_rect.right - small_icon_size - (
                                effective_pix_square_size // 16 if effective_pix_square_size > 15 else 1)
                            icon_pos_y = agent_base_rect.top + (
                                effective_pix_square_size // 16 if effective_pix_square_size > 15 else 1)
                            game_surface.blit(small_treasure_surface, (icon_pos_x, icon_pos_y))
                    except Exception as e:
                        if effective_pix_square_size > 4:
                            pygame.draw.rect(game_surface, self.treasure_color,
                                             (agent_base_rect.centerx - effective_pix_square_size // 8,
                                              agent_base_rect.top + effective_pix_square_size // 10,
                                              effective_pix_square_size // 4, effective_pix_square_size // 4))
                elif effective_pix_square_size > 4:
                    pygame.draw.rect(game_surface, self.treasure_color,
                                     (agent_base_rect.centerx - effective_pix_square_size // 8,
                                      agent_base_rect.top + effective_pix_square_size // 10,
                                      effective_pix_square_size // 4, effective_pix_square_size // 4))
        else:
            center_x = agent_screen_x + effective_pix_square_size / 2
            center_y = agent_screen_y + effective_pix_square_size / 2
            radius = effective_pix_square_size / 3
            if radius > 1:
                pygame.draw.circle(game_surface, self.agent_color, (center_x, center_y), radius)
                eye_radius = radius / 4
                if eye_radius > 1:
                    pygame.draw.circle(game_surface, (255, 255, 255), (center_x - radius / 2.5, center_y - radius / 3),
                                       eye_radius)
                    pygame.draw.circle(game_surface, (255, 255, 255), (center_x + radius / 2.5, center_y - radius / 3),
                                       eye_radius)
                    pygame.draw.circle(game_surface, (0, 0, 0), (center_x - radius / 2.5, center_y - radius / 3),
                                       max(1, eye_radius / 2))
                    pygame.draw.circle(game_surface, (0, 0, 0), (center_x + radius / 2.5, center_y - radius / 3),
                                       max(1, eye_radius / 2))
                if self.has_treasure and radius > 2:
                    pygame.draw.rect(game_surface, self.treasure_color,
                                     (center_x - radius / 2, center_y + radius / 1.5, radius, radius / 2))

        if self.interactive_mode and hasattr(self, 'path_taken') and len(self.path_taken) > 1:
            if effective_pix_square_size > 0:
                path_line_color = (255, 0, 0)
                path_line_thickness = max(1,
                                          int(effective_pix_square_size * 0.05) + 1)

                for i in range(len(self.path_taken) - 1):
                    r1, c1 = self.path_taken[i]
                    r2, c2 = self.path_taken[i + 1]

                    start_x_on_screen = c1 * effective_pix_square_size + effective_pix_square_size / 2 + view_offset_x
                    start_y_on_screen = r1 * effective_pix_square_size + effective_pix_square_size / 2 + view_offset_y
                    end_x_on_screen = c2 * effective_pix_square_size + effective_pix_square_size / 2 + view_offset_x
                    end_y_on_screen = r2 * effective_pix_square_size + effective_pix_square_size / 2 + view_offset_y

                    pygame.draw.line(game_surface, path_line_color,
                                     (start_x_on_screen, start_y_on_screen),
                                     (end_x_on_screen, end_y_on_screen),
                                     path_line_thickness)

        if self.zoom_level >= 0.5 and effective_pix_square_size > 2:
            grid_line_color = (150, 170, 200, 100)
            temp_surface_for_grid = pygame.Surface(game_surface.get_size(),
                                                   pygame.SRCALPHA)

            for r_idx in range(self.nrow + 1):
                y = r_idx * effective_pix_square_size + view_offset_y
                pygame.draw.line(temp_surface_for_grid, grid_line_color,
                                 (view_offset_x, y),
                                 (view_offset_x + zoomed_map_width, y), 1)

            for c_idx in range(self.ncol + 1):
                x = c_idx * effective_pix_square_size + view_offset_x
                pygame.draw.line(temp_surface_for_grid, grid_line_color,
                                 (x, view_offset_y),
                                 (x, view_offset_y + zoomed_map_height), 1)
            game_surface.blit(temp_surface_for_grid, (0, 0))

        canvas.blit(game_surface, (0, 0))

        info_panel_surf = pygame.Surface((self.info_panel_width, self.window_height))
        info_panel_surf.fill((60, 60, 90))
        text_y = 20
        txt_color_main = (220, 220, 255)
        txt_color_value = (200, 255, 200)
        txt_color_error = (255, 100, 100)

        def render_text_panel(surface, text, y_pos, font, color=(220, 220, 255), x_offset=15):
            try:
                if font:
                    text_surface = font.render(text, True, color)
                    surface.blit(text_surface, (x_offset, y_pos))
                    return y_pos + font.get_height() + 5
            except pygame.error:
                pass
            return y_pos + 20

        if self.game_font and self.small_font:
            text_y = render_text_panel(info_panel_surf, f"Skoki: {self.jumps_left}/{self.max_jumps}", text_y,
                                       self.game_font, txt_color_value if self.jumps_left > 0 else txt_color_error)
            text_y = render_text_panel(info_panel_surf, f"Skarb: {'TAK' if self.has_treasure else 'NIE'}", text_y,
                                       self.game_font, txt_color_value if self.has_treasure else txt_color_main)
            text_y = render_text_panel(info_panel_surf, f"Zoom: {self.zoom_level:.2f}x (+/=, -)", text_y,
                                       self.small_font, txt_color_main)
            text_y = render_text_panel(info_panel_surf,
                                       f"Kamera na agencie (C): {'ON' if self.camera_on_agent else 'OFF'}", text_y,
                                       self.small_font, txt_color_main)
            text_y = render_text_panel(info_panel_surf, f"FPS: {self.current_fps} (PgUp/PgDn)", text_y, self.small_font,
                                       txt_color_main)
            text_y = render_text_panel(info_panel_surf, f"Pełny ekran (F11)", text_y, self.small_font, txt_color_main)
            if self.interactive_mode:
                text_y = render_text_panel(info_panel_surf, f"Reset (R) - gdy koniec", text_y, self.small_font,
                                           txt_color_main)

            text_y += 10

            if self.lastaction is not None:
                action_names = ["LEWO", "DÓŁ", "PRAWO", "GÓRA", "SKOK L", "SKOK D", "SKOK P", "SKOK G"]
                text_y = render_text_panel(info_panel_surf, f"Akcja: {action_names[self.lastaction]}", text_y,
                                           self.small_font, txt_color_main)

            text_y += 5
            if hasattr(self, 'last_step_info') and self.last_step_info:
                if "error" in self.last_step_info and self.last_step_info["error"]:
                    text_y = render_text_panel(info_panel_surf, self.last_step_info["error"], text_y, self.small_font,
                                               txt_color_error, x_offset=10)

        canvas.blit(info_panel_surf, (self.game_area_width, 0))

        if self.render_mode == "human":
            if self.window:
                self.window.blit(canvas, canvas.get_rect())
                pygame.display.update()
            if self.clock: self.clock.tick(self.current_fps)
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(game_surface)), axes=(1, 0, 2))

    def _render_text(self):
        desc_list = self.current_map_state.tolist()
        outfile = StringIO()
        row, col = self.current_row, self.current_col
        desc_text = [[c.decode("utf-8") for c in line] for line in desc_list]
        try:
            desc_text[row][col] = utils.colorize(desc_text[row][col], "red", highlight=True)
        except Exception:
            desc_text[row][col] = f"[{desc_text[row][col]}]"

        outfile.write(
            f"Pozycja: ({row},{col}), Skoki: {self.jumps_left}/{self.max_jumps}, Skarb: {'TAK' if self.has_treasure else 'NIE'}\n")
        if self.lastaction is not None:
            action_names = ["Ruch L", "Ruch D", "Ruch P", "Ruch G", "Skok L", "Skok D", "Skok P", "Skok G"]
            outfile.write(f"Ostatnia akcja: {action_names[self.lastaction]}\n")

        if hasattr(self, 'last_step_info') and self.last_step_info:
            if "error" in self.last_step_info and self.last_step_info["error"]:
                outfile.write(f"Info: {self.last_step_info['error']}\n")

        if self.interactive_mode and self.game_is_over_manual_reset_pending:
            outfile.write("Koniec gry. Naciśnij 'R' aby zresetować (w trybie human).\n")

        outfile.write("\n".join("".join(line) for line in desc_text) + "\n")
        with closing(outfile):
            return outfile.getvalue()

    def render(self):
        if self.render_mode is None: return
        if self.render_mode == "ansi":
            rendered_text = self._render_text()
            print(rendered_text)
        elif self.render_mode in ["human", "rgb_array"]:
            return self._render_frame()
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}. Dostępne: 'human', 'ansi', 'rgb_array'.")

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.window = None
        self.clock = None
        self.game_font = None
        self.small_font = None
        self.scaled_sprites = {}
        self.assets_loaded = False
        self.original_sprites_pil = {}
