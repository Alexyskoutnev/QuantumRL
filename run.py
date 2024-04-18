import pennylane as qml
from quantum_rl.envs.env_frozen_lake import QuantumGridWorld
from quantum_rl.train_QQN import train, plot_rewards, plot_loss, plot_eval_rewards

if __name__ == "__main__":
    # ===== Config =====
    lr = 0.01
    gamma = 0.999
    epsilon = 1.0
    n_episodes = 100 # keep below 100 for testing with real QC hardware
    max_steps = 50
    n_test = 5
    batch_size = 4
    eval_update = 5
    device = ""
    # ===== Config =====
    config = {
        'lr': lr,
        'gamma': gamma,
        'epsilon': epsilon,
        'n_episodes': n_episodes,
        'max_steps': max_steps,
        'n_test': n_test,
        'batch_size': batch_size,
        'eval_update': eval_update,
        'n_test': 5,
    }
    iter_index, iter_reward, iter_total_steps, timestep_reward, var_Q_circuit, var_Q_bias, iter_loss, eval_rewards = train(config)
    plot_rewards(iter_index, iter_reward, filename='plots/reward_plot.png')
    plot_loss(iter_index, iter_loss, filename='plots/loss_plot.png')
    plot_eval_rewards(eval_rewards, filename='plots/eval_plot.png')