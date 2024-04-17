from quantum_rl.envs.env_frozen_lake import QuantumGridWorld

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
import matplotlib.pyplot as plt
from collections import namedtuple

dev = qml.device("default.qubit", wires=4)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))    

class QReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
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

def layer(W):
    """ Single layer of the variational classifier."""
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

def decimalToBinaryFixLength(_length, _decimal):
    binNum = bin(int(_decimal))[2:]
    outputNum = [int(item) for item in binNum]
    if len(outputNum) < _length:
        outputNum = np.concatenate((np.zeros((_length-len(outputNum),)),np.array(outputNum)))
    else:
        outputNum = np.array(outputNum)
    return outputNum

@qml.qnode(dev, interface='torch')
def circuit(weights, angles=None):
    preprocess_state(angles)
    for W in weights:
        layer(W)
    return [qml.expval(qml.PauliZ(ind)) for ind in range(4)]
 

def variational_classifier(var_Q_circuit, var_Q_bias, angles):
    """The variational classifier."""
    weights = var_Q_circuit
    bias = var_Q_bias
    _circuit_output = circuit(weights, angles=angles)
    return torch.tensor(_circuit_output) + bias


def mse_loss(labels, predictions):
	""" Square loss function

	Args:
		labels (array[float]): 1-d array of labels
		predictions (array[float]): 1-d array of predictions
	Returns:
		float: square loss
	"""
	loss = 0
	for l, p in zip(labels, predictions):
	    loss += (l - p) ** 2
	loss /= len(labels)
	return loss

def huber_loss(labels, predictions):
	""" Square loss function

	Args:
		labels (array[float]): 1-d array of labels
		predictions (array[float]): 1-d array of predictions
	Returns:
		float: square loss
	"""
	loss = nn.SmoothL1Loss()
	return loss(labels, predictions)

def epsilon_greedy(var_Q_circuit, var_Q_bias, epsilon, n_actions, state, train=False):
    if train or np.random.rand() < ((epsilon/n_actions)+(1-epsilon)):
        action = torch.argmax(variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles = decimalToBinaryFixLength(4, state['agent'])))
    else:
        action = torch.tensor(np.random.randint(0, n_actions))
    return action


def train(config):
    env = gym.make('QuantumGridWorld')
    n_actions = env.action_space.n
    obs, _ = env.reset()

    num_qubits = 4
    num_layers = 2
    dtype = torch.float64

    var_init_circuit = Variable(torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3), device='cpu').type(dtype), requires_grad=True)
    var_init_bias = Variable(torch.tensor([0.0, 0.0, 0.0, 0.0], device='cpu').type(dtype), requires_grad=True)

    var_Q_circuit = var_init_circuit
    var_Q_bias = var_init_bias
    var_target_Q_circuit = var_Q_circuit.clone().detach()
    var_target_Q_bias = var_Q_bias.clone().detach()
    opt = torch.optim.RMSprop([var_Q_circuit, var_Q_bias], lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    
    TARGET_UPDATE = 10
    batch_size = config['batch_size']
    OPTIMIZE_STEPS = 5

    target_update_counter = 0
    
    iter_index = []
    iter_reward = []
    iter_total_steps = []
    cost_list = []
    timestep_reward = []
    timestep = []
    loss_l = []

    memory = QReplayMemory(128)

    for episode in range(config['n_episodes']):
        print("Episode: ", episode)
        obs, _ = env.reset()
        a = epsilon_greedy(var_Q_circuit, var_Q_bias, config['epsilon'], n_actions, obs, train=True)
        t = 0
        total_reward = 0
        done = False

        while t < config['max_steps'] and not done:
            target_update_counter += 1
            obs_next, reward, done, truc, info = env.step(a.item())
            total_reward += reward
            memory.push(obs, a, obs_next, reward, done)
            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                Q_target = []
                for transition in transitions:
                    obs, a, obs_next, reward, done = transition
                    if done:
                        Q_target.append(reward)
                    else:
                        Q_target.append(reward + config['gamma'] * torch.max(variational_classifier(var_Q_circuit = var_target_Q_circuit, var_Q_bias = var_target_Q_bias, angles=decimalToBinaryFixLength(4, obs_next['agent']))))
                Q_target = torch.tensor(Q_target)
                def closure():
                    opt.zero_grad() 
                    predictions = [variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=decimalToBinaryFixLength(4, item[0]['agent']))[item[1]] for item in transitions]
                    loss = mse_loss(Q_target, predictions)
                    loss.backward()
                    loss_l.append((t, loss.item()))
                    return loss
                opt.step(closure)

            if target_update_counter % TARGET_UPDATE == 0:
                var_target_Q_circuit = var_Q_circuit.clone().detach()
                var_target_Q_bias = var_Q_bias.clone().detach()

            a = epsilon_greedy(var_Q_circuit, var_Q_bias, config['epsilon'], n_actions, obs, train=True)
            obs = obs_next
            t += 1
            if done:
                timestep_reward.append(total_reward)
                print("Episode: ", episode, "Reward: ", total_reward)
                iter_index.append(episode)
                iter_reward.append(total_reward)
                iter_total_steps.append(t)
                break

    return iter_index, iter_reward, iter_total_steps, timestep_reward, var_Q_circuit, var_Q_bias, loss_l

def plot_rewards(iter_index, iter_reward):
    plt.plot(iter_index, iter_reward)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

def plot_loss(loss_l):
    timesteps, losses = zip(*loss_l)
    plt.plot(timesteps, losses)
    plt.title('Loss over Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # ===== Config =====
    lr = 0.95
    gamma = 0.9
    epsilon = 0.9
    n_episodes = 100
    max_steps = 2500
    n_test = 2
    batch_size = 16
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

    plot_rewards(iter_index, iter_reward)
    plot_loss(loss_l)