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

    def __init__(self, n: int, input_goal: bool=False):
        super(BitFlippingDQNNetworkGoals, self).__init__()
        if input_goal:
            self.fc1 = nn.Linear(2 * n, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, n)
        else:
            self.fc1 = nn.Linear(n, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, n * n)
        self.input_goal = input_goal
        self.n = n

    def forward(self, x):
        x = torch.nn.functional.gelu(self.fc1(x))
        x = torch.nn.functional.gelu(self.fc2(x))
        if self.input_goal:
            return self.fc3(x)
        return self.fc3(x).view(-1, self.n, self.n)


class BitFlippingDQNNetworkGoalsUVFA(nn.Module):
    """
    BitFlipping DQN network for the FlippingBitSequenceEnvWithGoals environment. Assumes the state space and action space are the same (complete graph).

    For the network input (state), we use R^n vector encoding for the bit sequence. The output would be a n^2 dimensional real vector,
    where the first index of the vector corresponds to the action of flipping the bit at the index, and the second index corresponds to the goal.
    This parameterizes the goal-dependent Q function Q(s, a, g).
    """

    def __init__(self, n: int):
        super(BitFlippingDQNNetworkGoalsUVFA, self).__init__()
        self.fc1 = nn.Linear(n, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128 * n)

        self.goal_embedding = nn.Embedding(n, 128)

        self.n = n

    def forward(self, state, goal):
        assert goal.shape[0] == state.shape[0], "Batch size of state and goal must match"
        assert len(goal.shape) == 1, "Goal must be a 1D tensor"

        x = torch.nn.functional.gelu(self.fc1(state))
        x = torch.nn.functional.gelu(self.fc2(x))
        x = self.fc3(x).view(-1, self.n, 128)
        goal = self.goal_embedding(goal)
        return torch.bmm(x, goal.unsqueeze(2)).squeeze(2)
    
    def compute_full_goals(self, state):
        x = torch.nn.functional.gelu(self.fc1(state))
        x = torch.nn.functional.gelu(self.fc2(x))
        x = self.fc3(x).view(-1, self.n, 128)
        return torch.bmm(x, torch.t(self.goal_embedding.weight).unsqueeze(0).expand(state.shape[0], -1, -1))