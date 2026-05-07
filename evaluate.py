import random
import numpy as np
import matplotlib.pyplot as plt
import os

from environment import SnakeEnv
from model import QNetwork
from replay_buffer import ReplayBuffer
from agent import DQNAgent

CHECKPOINT  = "checkpoints/best_model.pth"
EVAL_EPS    = 100
STATE_SIZE  = 11
ACTION_SIZE = 3


# ─── Random Agent baseline ──────────────────────────────────────────
def run_random_agent(episodes=EVAL_EPS):
    env    = SnakeEnv(grid_size=10, obstacles=True)
    scores = []
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(random.randint(0, 2))
        scores.append(info["score"])
    return scores


# ─── DQN Agent evaluation ───────────────────────────────────────────
def run_dqn_agent(episodes=EVAL_EPS):
    env    = SnakeEnv(grid_size=10, obstacles=True)
    buffer = ReplayBuffer(capacity=1000, batch_size=64)
    agent  = DQNAgent(
        state_size  = STATE_SIZE,
        action_size = ACTION_SIZE,
        model_class = QNetwork,
        buffer      = buffer,
        epsilon     = 0.0,      # pure exploitation
        epsilon_min = 0.0,
    )
    if os.path.isfile(CHECKPOINT):
        agent.q_network.load(CHECKPOINT)
        print(f"✅ Loaded checkpoint: {CHECKPOINT}")
    else:
        print("⚠️  No checkpoint found — evaluating untrained agent")

    scores = []
    for _ in range(episodes):
        state = env.reset()
        done  = False
        while not done:
            action = agent.select_action(state)
            state, _, done, info = env.step(action)
        scores.append(info["score"])
    return scores


# ─── Comparison plot ────────────────────────────────────────────────
def plot_comparison(dqn_scores, random_scores):
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DQN vs Random Agent — Evaluation", fontsize=15, fontweight="bold")

    # Score distribution
    axes[0].hist(dqn_scores,    bins=15, alpha=0.7, color="steelblue",  label="DQN")
    axes[0].hist(random_scores, bins=15, alpha=0.7, color="darkorange", label="Random")
    axes[0].axvline(np.mean(dqn_scores),    color="steelblue",  linestyle="--")
    axes[0].axvline(np.mean(random_scores), color="darkorange", linestyle="--")
    axes[0].set_title("Score Distribution")
    axes[0].set_xlabel("Score"); axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Episode-by-episode line
    axes[1].plot(dqn_scores,    alpha=0.7, color="steelblue",  label="DQN")
    axes[1].plot(random_scores, alpha=0.7, color="darkorange", label="Random")
    axes[1].set_title("Score per Episode")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/dqn_vs_random.png", dpi=150)
    plt.show()
    print("Comparison plot saved to results/dqn_vs_random.png")


def evaluate():
    print("=" * 50)
    print("  Evaluating Random Agent ...")
    random_scores = run_random_agent()
    print(f"  Random → Mean: {np.mean(random_scores):.2f} | "
          f"Max: {max(random_scores)} | Std: {np.std(random_scores):.2f}")

    print("\n  Evaluating DQN Agent ...")
    dqn_scores = run_dqn_agent()
    print(f"  DQN    → Mean: {np.mean(dqn_scores):.2f} | "
          f"Max: {max(dqn_scores)} | Std: {np.std(dqn_scores):.2f}")

    improvement = np.mean(dqn_scores) - np.mean(random_scores)
    print(f"\n  Improvement over Random: {improvement:+.2f} points")
    print("=" * 50)

    plot_comparison(dqn_scores, random_scores)
    return dqn_scores, random_scores


if __name__ == "__main__":
    evaluate()
