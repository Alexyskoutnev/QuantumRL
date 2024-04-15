from gymnasium.envs.registration import register

register(
     id="gym_examples/QuantumGridWorld-v0",
     entry_point="src.envs:QuantumGridWorld",
     max_episode_steps=300,
)