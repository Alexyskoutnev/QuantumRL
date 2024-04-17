from gymnasium.envs.registration import register

register(
     id="QuantumGridWorld-v0",
     entry_point="quantum_rl.envs.env_frozen_lake:QuantumGridWorld",
     max_episode_steps=50,
)