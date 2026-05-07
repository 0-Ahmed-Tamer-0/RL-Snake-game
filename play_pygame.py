"""
play_pygame.py — Watch the trained DQN agent play Snake visually.

Requirements:
    pip install pygame torch

Run:
    python play_pygame.py                  # uses checkpoints/best_model.pth
    python play_pygame.py --random         # watch random agent instead
    python play_pygame.py --human          # play yourself (arrow keys)
"""

import sys
import time
import random
import argparse
import pygame

from environment import SnakeEnv
from model import QNetwork
from replay_buffer import ReplayBuffer
from agent import DQNAgent

# ─── Display constants ───────────────────────────────────────────────
CELL        = 48          # pixels per grid cell
GRID        = 10          # must match training
WIDTH       = CELL * GRID
PANEL       = 160         # right-side info panel width
HEIGHT      = CELL * GRID
FPS_DEFAULT = 8

# ─── Colors ──────────────────────────────────────────────────────────
BG          = (15,  17,  26)
GRID_LINE   = (25,  28,  42)
SNAKE_HEAD  = (80,  220, 140)
SNAKE_BODY  = (40,  160,  90)
FOOD_COL    = (255, 90,   90)
OBSTACLE    = (80,  80,  110)
PANEL_BG    = (20,  22,  35)
TEXT_COL    = (200, 210, 230)
ACCENT      = (80,  220, 140)
DIM         = (80,   90, 110)

pygame.font.init()
FONT_LARGE  = pygame.font.SysFont("consolas", 22, bold=True)
FONT_SMALL  = pygame.font.SysFont("consolas", 14)
FONT_TINY   = pygame.font.SysFont("consolas", 12)


def draw_rounded_rect(surface, color, rect, radius=6):
    pygame.draw.rect(surface, color, rect, border_radius=radius)


def draw_frame(surface, env, score, episode, steps, mode_label, fps):
    surface.fill(BG)

    # ── Grid lines ──
    for i in range(GRID + 1):
        pygame.draw.line(surface, GRID_LINE, (i * CELL, 0), (i * CELL, HEIGHT))
        pygame.draw.line(surface, GRID_LINE, (0, i * CELL), (WIDTH, i * CELL))

    # ── Obstacles ──
    for (r, c) in env.obstacles:
        rect = pygame.Rect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4)
        draw_rounded_rect(surface, OBSTACLE, rect, 4)

    # ── Snake body ──
    for i, (r, c) in enumerate(env.snake):
        color = SNAKE_HEAD if i == 0 else SNAKE_BODY
        rect  = pygame.Rect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4)
        draw_rounded_rect(surface, color, rect, 6 if i == 0 else 4)

        # Eyes on head
        if i == 0:
            er, ec = r, c
            eye_size = 5
            offsets = {
                ( 0,  1): [(CELL-12, 8), (CELL-12, CELL-14)],  # RIGHT
                ( 0, -1): [(8, 8), (8, CELL-14)],              # LEFT
                (-1,  0): [(8, 8), (CELL-14, 8)],              # UP
                ( 1,  0): [(8, CELL-12), (CELL-14, CELL-12)],  # DOWN
            }
            eye_positions = offsets.get(env.direction, [(8, 8), (CELL-14, 8)])
            for (ex, ey) in eye_positions:
                pygame.draw.circle(surface, BG,
                                   (ec * CELL + ex, er * CELL + ey), eye_size)

    # ── Food ──
    if env.food:
        fr, fc = env.food
        cx = fc * CELL + CELL // 2
        cy = fr * CELL + CELL // 2
        pygame.draw.circle(surface, FOOD_COL, (cx, cy), CELL // 2 - 4)
        pygame.draw.circle(surface, (255, 160, 160), (cx - 4, cy - 4), 4)

    # ── Panel ──
    panel_rect = pygame.Rect(WIDTH, 0, PANEL, HEIGHT)
    pygame.draw.rect(surface, PANEL_BG, panel_rect)
    pygame.draw.line(surface, ACCENT, (WIDTH, 0), (WIDTH, HEIGHT), 2)

    def label(text, x, y, color=TEXT_COL, font=FONT_SMALL):
        surface.blit(font.render(text, True, color), (x, y))

    px = WIDTH + 12
    label("SNAKE  DQN", px, 18, ACCENT, FONT_LARGE)
    pygame.draw.line(surface, GRID_LINE, (WIDTH + 8, 48), (WIDTH + PANEL - 8, 48))

    label("MODE",    px, 60,  DIM)
    label(mode_label, px, 76, ACCENT)

    label("EPISODE", px, 108, DIM)
    label(str(episode), px, 124, TEXT_COL)

    label("SCORE",   px, 156, DIM)
    label(str(score),  px, 172, ACCENT, FONT_LARGE)

    label("LENGTH",  px, 210, DIM)
    label(str(len(env.snake)), px, 226, TEXT_COL)

    label("STEPS",   px, 258, DIM)
    label(str(steps),  px, 274, TEXT_COL)

    label("SPEED",   px, 306, DIM)
    label(f"x{env.speed_level}", px, 322, TEXT_COL)

    label("FPS",     px, 354, DIM)
    label(str(fps),    px, 370, TEXT_COL)

    label("[+-] speed", px, HEIGHT - 50, DIM, FONT_TINY)
    label("[R] restart", px, HEIGHT - 34, DIM, FONT_TINY)
    label("[Q] quit",    px, HEIGHT - 18, DIM, FONT_TINY)


def build_agent(checkpoint="checkpoints/best_model.pth"):
    buffer = ReplayBuffer(capacity=1000, batch_size=64)
    agent  = DQNAgent(
        state_size  = 11,
        action_size = 3,
        model_class = QNetwork,
        buffer      = buffer,
        epsilon     = 0.0,
        epsilon_min = 0.0,
    )
    try:
        agent.q_network.load(checkpoint)
        print(f"✅ Loaded: {checkpoint}")
    except FileNotFoundError:
        print(f"⚠️  Checkpoint not found ({checkpoint}). Running untrained agent.")
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="Run random agent")
    parser.add_argument("--human",  action="store_true", help="Play with arrow keys")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH + PANEL, HEIGHT))
    pygame.display.set_caption("Snake DQN")
    clock  = pygame.time.Clock()

    env   = SnakeEnv(grid_size=GRID, obstacles=True)
    agent = None if (args.random or args.human) else build_agent(args.checkpoint)

    if args.human:
        mode_label = "HUMAN"
    elif args.random:
        mode_label = "RANDOM"
    else:
        mode_label = "DQN"

    fps     = args.fps
    episode = 1
    state   = env.reset()
    score   = 0
    steps   = 0
    pending_human_action = None

    running = True
    while running:
        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    state = env.reset(); score = 0; steps = 0; episode += 1
                if event.key == pygame.K_KP_PLUS:
                    fps = min(fps + 2, 30)
                if event.key == pygame.K_KP_MINUS:
                    fps = max(fps - 2, 1)

                # Human controls → translate absolute to relative action
                if args.human:
                    dir_map = {
                        pygame.K_UP:    (-1, 0),
                        pygame.K_DOWN:  ( 1, 0),
                        pygame.K_LEFT:  ( 0,-1),
                        pygame.K_RIGHT: ( 0, 1),
                    }
                    wanted = dir_map.get(event.key)
                    if wanted:
                        from environment import TURN_LEFT, TURN_RIGHT
                        cur = env.direction
                        if wanted == cur:
                            pending_human_action = 0
                        elif wanted == TURN_LEFT[cur]:
                            pending_human_action = 1
                        elif wanted == TURN_RIGHT[cur]:
                            pending_human_action = 2
                        else:
                            pending_human_action = 0   # 180° → go straight

        # ── Choose action ──
        if args.human:
            action = pending_human_action if pending_human_action is not None else 0
            pending_human_action = None
        elif args.random:
            action = random.randint(0, 2)
        else:
            action = agent.select_action(state)

        # ── Step ──
        next_state, reward, done, info = env.step(action)
        state  = next_state
        score  = info["score"]
        steps  = info["steps"]

        if done:
            time.sleep(0.4)
            state = env.reset(); score = 0; steps = 0; episode += 1

        # ── Draw ──
        draw_frame(screen, env, score, episode, steps, mode_label, fps)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
