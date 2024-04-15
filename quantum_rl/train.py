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
from collections import namedtuple

dev = qml.device("default.qubit", wires=4)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))    

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

def layer(W):
	""" Single layer of the variational classifier.

	Args:
		W (array[float]): 2-d array of variables for one layer
	"""

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
	"""The circuit of the variational classifier."""
	# Can consider different expectation value
	# PauliX , PauliY , PauliZ , Identity  

	statepreparation(angles)
	
	for W in weights:
		layer(W)

	return [qml.expval(qml.PauliZ(ind)) for ind in range(4)]


def variational_classifier(var_Q_circuit, var_Q_bias , angles=None):
	"""The variational classifier."""

	# Change to SoftMax???

	weights = var_Q_circuit
	# bias_1 = var_Q_bias[0]
	# bias_2 = var_Q_bias[1]
	# bias_3 = var_Q_bias[2]
	# bias_4 = var_Q_bias[3]
	# bias_5 = var_Q_bias[4]
	# bias_6 = var_Q_bias[5]

	# raw_output = circuit(weights, angles=angles) + np.array([bias_1,bias_2,bias_3,bias_4,bias_5,bias_6])
	raw_output = circuit(weights, angles=angles) + var_Q_bias
	# We are approximating Q Value
	# Maybe softmax is no need
	# softMaxOutPut = np.exp(raw_output) / np.exp(raw_output).sum()

	return raw_output

def square_loss(labels, predictions):
	""" Square loss function

	Args:
		labels (array[float]): 1-d array of labels
		predictions (array[float]): 1-d array of predictions
	Returns:
		float: square loss
	"""
	loss = 0
	for l, p in zip(labels, predictions):
	    loss = loss + (l - p) ** 2
	loss = loss / len(labels)
	return loss

def cost(var_Q_circuit, var_Q_bias, features, labels):
	"""Cost (error) function to be minimized."""

	# predictions = [variational_classifier(weights, angles=f) for f in features]
	# Torch data type??
	
	predictions = [variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=decimalToBinaryFixLength(4,item.state))[item.action] for item in features]
	# predictions = torch.tensor(predictions,requires_grad=True)
	# labels = torch.tensor(labels)
	# print("PRIDICTIONS:")
	# print(predictions)
	# print("LABELS:")
	# print(labels)

	return square_loss(labels, predictions)


def huber_loss(labels, predictions):
	""" Square loss function

	Args:
		labels (array[float]): 1-d array of labels
		predictions (array[float]): 1-d array of predictions
	Returns:
		float: square loss
	"""
	# In Deep Q Learning
	# labels = target_action_value_Q
	# predictions = action_value_Q

	# loss = 0
	# for l, p in zip(labels, predictions):
	# 	loss = loss + (l - p) ** 2
	# loss = loss / len(labels)

	# loss = nn.MSELoss()
	loss = nn.SmoothL1Loss()
	# output = loss(torch.tensor(predictions), torch.tensor(labels))
	# print("LOSS OUTPUT")
	# print(output)

	return loss(labels, predictions)



def epsilon_greedy(var_Q_circuit, var_Q_bias, epsilon, n_actions, state, train=False):
    
    if train or np.random.rand() < ((epsilon/n_actions)+(1-epsilon)):
            # action = np.argmax(Q[s, :])
            # variational classifier output is torch tensor
            # action = np.argmax(variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles = decimalToBinaryFixLength(9,s)))
            action = torch.argmax(variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles = decimalToBinaryFixLength(4,s)))
    else:
        # need to be torch tensor
        action = torch.tensor(np.random.randint(0, n_actions))
    return action


def train(config):
    env = gym.make('QuantumGridWorld')
    n_actions = env.action_space.n
    obs, _ = env.reset()
    breakpoint()

    num_qubits = 4
    num_layers = 2
    dtype = torch.float64

    var_init_circuit = Variable(torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3), device='cpu').type(dtype), requires_grad=True)
    var_init_bias = Variable(torch.tensor([0.0, 0.0, 0.0, 0.0], device='cpu').type(dtype), requires_grad=True)

    var_Q_circuit = var_init_circuit
    var_Q_bias = var_init_bias
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

    memory = QReplayMemory(80)

    for episode in range(config['n_episodes']):
        print("Episode: ", episode)
        obs, _ = env.reset()
        breakpoint()





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


