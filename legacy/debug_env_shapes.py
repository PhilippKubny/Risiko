from risk_env import RiskEnv

def main():
    env = RiskEnv()
    obs, info = env.reset()
    print("flat_features shape:", obs["flat_features"].shape)
    print("per_territory shape:", obs["per_territory"].shape)
    print("extra shape:", obs["extra"].shape)

if __name__ == "__main__":
    main()
