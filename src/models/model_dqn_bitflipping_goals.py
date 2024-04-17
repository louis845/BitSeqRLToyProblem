import torch
from torch import nn, optim
from collections import deque
import numpy as np
import random

class BitFlippingDQNNetworkGoals(nn.Module):
    """
    BitFlipping DQN network for the FlippingBitSequenceEnvWithGoals environment. Assumes the state space and action space are the same (complete graph).

    For the network input (state), we use R^n vector encoding for the bit sequence. The output would be a n^2 dimensional real vector,
    where the first index of the vector corresponds to the action of flipping the bit at the index, and the second index corresponds to the goal.
    This parameterizes the goal-dependent Q function Q(s, a, g).
    """

    def __init__(self, n: int):
        super(BitFlippingDQNNetworkGoals, self).__init__()
        self.fc1 = nn.Linear(n, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n * n)
        self.n = n

    def forward(self, x):
        x = torch.nn.functional.gelu(self.fc1(x))
        x = torch.nn.functional.gelu(self.fc2(x))
        return self.fc3(x).view(-1, self.n, self.n)
