# file: play_random_games.py
import random
from risk_env import RiskEnv

def run_random_episodes(num_episodes: int = 20):
    env = RiskEnv(max_steps=5000)  # etwas kleiner für schnellere Episoden
    wins = {"Lila": 0, "Rot": 0, "time_limit": 0, "no_progress": 0}

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_steps = 0

        while not done:
            legal = env.get_legal_actions()
            a = random.choice(legal)
            obs, r, done, info = env.step(a)
            total_steps += 1

        if "winner" in info and info["winner"] is not None:
            wins[info["winner"]] += 1
        elif info.get("no_progress_termination"):
            wins["no_progress"] += 1
        else:
            wins["time_limit"] += 1

        print(f"EP {ep}: info={info}, reward={r}, steps={total_steps}")

    print("\nSummary after", num_episodes, "episodes:")
    print(wins)

if __name__ == "__main__":
    run_random_episodes()
