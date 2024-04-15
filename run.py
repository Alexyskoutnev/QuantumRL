from quantum_rl.envs.env_frozen_lake import QuantumGridWorld
from quantum_rl.train import train

if __name__ == "__main__":
    lr = 0.95
    gamma = 0.9
    epsilon = 0.9
    n_episodes = 100
    max_steps = 2500
    n_test = 2
    batch_size = 4
    # ===== Config =====
    config = {
        'lr': lr,
        'gamma': gamma,
        'epsilon': epsilon,
        'n_episodes': n_episodes,
        'max_steps': max_steps,
        'n_test': n_test,
        'batch_size': batch_size
    }
    train(config)
