from src.envs.env_frozen_lake import QuantumGridWorld



if __name__ == "__main__":
    env = QuantumGridWorld(render_mode='human')
    env.reset()
    env.render()
    env.close()
    print('done')