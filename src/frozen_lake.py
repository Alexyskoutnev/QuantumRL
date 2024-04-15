import numpy as np
import random



import pennylane as qml
import torch
import torch.nn as nn
from torch.autograd import Variable
import gymnasium as gym

import numpy as np
import random
import pickle
from collections import namedtuple

dev = qml.device("default.qubit", wires=4)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class QuantumFrozenLakeEnv(gym.Env):
    def __int__(self):
        super(QuantumFrozenLakeEnv, self).__init__()
        self.env = gym.make('FrozenLake-v1')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    

class QReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __str__(self) -> str:
        return str(self.memory)

    def __len__(self):
        return  len(self.memory)

def preprocess_state(a):
    for ind in range(len(a)):
        qml.RX(np.pi * a[ind], wires=ind)
        qml.RZ(np.pi * a[ind], wires=ind)

def variational_circuit(params, wires):
    pass


def epsilon_greedy(var_Q_circuit, var_Q_bias, epsilon, n_actions, state):
    pass


def train(config):
    breakpoint()
    env = gym.make('QauntumFrozenLake')
    n_state, n_actions = env.observation_space.n, env.action_space.n


if __name__ == "__main__":
    # ===== Config =====
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


