import torch
from torch import nn, optim
from collections import deque
import numpy as np
import random

class NaiveDQNNetwork(nn.Module):
    """
    Naive DQN network for the NaiveBitSequenceEnv environment. Assumes the state space and action space are the same (complete graph).

    For the network input (state), we use one-hot encoding of the bit sequence. The output would be a 2^n dimensional real vector,
    where each entry is the set of all possible actions (bit sequences). This parameterizes the Q function.
    """

    def __init__(self, n: int):
        super(NaiveDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(n, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2 ** n)
        self.n = n

    def forward(self, x):
        x = torch.nn.functional.gelu(self.fc1(x))
        x = torch.nn.functional.gelu(self.fc2(x))
        return self.fc3(x)
