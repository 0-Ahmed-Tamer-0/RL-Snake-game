import os
import csv
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from environment import SnakeEnv
from model import QNetwork
from replay_buffer import ReplayBuffer
from agent import DQNAgent

# ─── Hyperparameters ────────────────────────────────────────────────
EPISODES       = 1000
STATE_SIZE     = 11
ACTION_SIZE    = 3
GAMMA          = 0.99
LR             = 0.0005
BATCH_SIZE     = 64
BUFFER_SIZE    = 10000
EPSILON        = 1.0
EPSILON_MIN    = 0.01
EPSILON_DECAY  = 0.5
TAU            = 0.001
TARGET_SCORE   = 15      # early stopping threshold
MOVING_AVG_N   = 50
SAVE_PATH      = "checkpoints/best_model.pth"
CSV_PATH       = "results/training_log.csv"
# ────────────────────────────────────────────────────────────────────

def train():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results",      exist_ok=True)

    env    = SnakeEnv(grid_size=10, obstacles=True)
    buffer = ReplayBuffer(capacity=BUFFER_SIZE, batch_size=BATCH_SIZE,device='cuda')
    agent  = DQNAgent(
        state_size    = STATE_SIZE,
        action_size   = ACTION_SIZE,
        model_class   = QNetwork,
        buffer        = buffer,
        gamma         = GAMMA,
        lr            = LR,
        batch_size    = BATCH_SIZE,
        epsilon       = EPSILON,
        epsilon_min   = EPSILON_MIN,
        epsilon_decay = EPSILON_DECAY,
        tau           = TAU,
    )

    scores, losses, epsilons = [], [], []
    moving_avgs = []
    score_window = deque(maxlen=MOVING_AVG_N)
    best_avg = -float("inf")

    # ── CSV header ──
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "avg_score", "loss", "epsilon"])

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        ep_losses    = []
        done         = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            state        = next_state
            total_reward += reward

        agent.decay_epsilon()

        score      = info["score"]
        avg_loss   = float(np.mean(ep_losses)) if ep_losses else 0.0
        scores.append(score)
        losses.append(avg_loss)
        epsilons.append(agent.epsilon)
        score_window.append(score)
        avg_score = float(np.mean(score_window))
        moving_avgs.append(avg_score)

        # ── Save best model ──
        if avg_score > best_avg and len(score_window) == MOVING_AVG_N:
            best_avg = avg_score
            agent.q_network.save(SAVE_PATH)

        # ── CSV row ──
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, score, round(avg_score, 2), round(avg_loss, 4), round(agent.epsilon, 4)])

        # ── Console log ──
        if ep % 50 == 0:
            print(f"Ep {ep:4d} | Score {score:3d} | "
                  f"Avg(50) {avg_score:5.2f} | "
                  f"Loss {avg_loss:.4f} | ε {agent.epsilon:.3f}")

        # ── Early stopping ──
        if avg_score >= TARGET_SCORE and len(score_window) == MOVING_AVG_N:
            print(f"\n✅ Early stopping at episode {ep} — avg score {avg_score:.2f}")
            break

    plot_results(scores, moving_avgs, losses, epsilons)
    print(f"\nTraining complete. Best model saved to: {SAVE_PATH}")
    return agent, scores, moving_avgs, losses, epsilons


def plot_results(scores, moving_avgs, losses, epsilons):
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Snake — Training Results", fontsize=16, fontweight="bold")

    axes[0, 0].plot(scores, alpha=0.4, color="steelblue", label="Score")
    axes[0, 0].set_title("Score per Episode")
    axes[0, 0].set_xlabel("Episode"); axes[0, 0].set_ylabel("Score")
    axes[0, 0].legend()

    axes[0, 1].plot(moving_avgs, color="darkorange", label=f"Moving Avg ({MOVING_AVG_N})")
    axes[0, 1].axhline(TARGET_SCORE, color="green", linestyle="--", label="Target")
    axes[0, 1].set_title("Moving Average Score")
    axes[0, 1].set_xlabel("Episode"); axes[0, 1].set_ylabel("Avg Score")
    axes[0, 1].legend()

    axes[1, 0].plot(epsilons, color="purple", label="Epsilon")
    axes[1, 0].set_title("Epsilon Decay")
    axes[1, 0].set_xlabel("Episode"); axes[1, 0].set_ylabel("Epsilon")
    axes[1, 0].legend()

    axes[1, 1].plot(losses, alpha=0.6, color="crimson", label="Loss")
    axes[1, 1].set_title("Training Loss")
    axes[1, 1].set_xlabel("Episode"); axes[1, 1].set_ylabel("MSE Loss")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    plt.show()
    print("Plots saved to results/training_curves.png")


if __name__ == "__main__":
    train()
