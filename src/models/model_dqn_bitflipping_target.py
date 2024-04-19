import torch
from torch import nn, optim
from collections import deque
import numpy as np
import random

class BitFlippingDQNNetworkTarget(nn.Module):
    """
    BitFlipping DQN network for the FlippingBitSequenceEnvRNGTarget environment.

    For the network input (state), we use R^n vector encoding for the bit sequence. Since the Q-function is also conditioned
    on the goal, we concatenate the state and goal bit sequence to form the input. The output is a n-dimensional real vector,
    where each index represents the Q-value of flipping the bit at that index (the action).
    """

    def __init__(self, n: int):
        super(BitFlippingDQNNetworkTarget, self).__init__()
        self.fc1 = nn.Linear(2 * n, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n)
        self.n = n

    def forward(self, state: torch.Tensor, goal: torch.Tensor):
        x = torch.cat((state, goal), dim=1)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = torch.nn.functional.gelu(self.fc2(x))
        return self.fc3(x)


class BitFlippingDQNNetworkTargetUVFA(nn.Module):
    """
    BitFlipping DQN network (UVFA) for the FlippingBitSequenceEnvRNGTarget environment.

    Same inputs and outputs, but with UVFA.
    """

    def __init__(self, n: int):
        super(BitFlippingDQNNetworkTargetUVFA, self).__init__()
        self.state_fc1 = nn.Linear(n, 128)
        self.goal_fc1 = nn.Linear(n, 128)
        self.bilinear_form = nn.Bilinear(128, 128, n)
        self.n = n

    def forward(self, state: torch.Tensor, goal: torch.Tensor):
        state = torch.nn.functional.gelu(self.state_fc1(state)) # (A, 128)
        goal = torch.nn.functional.gelu(self.goal_fc1(goal)) # (A, 128)
        result = self.bilinear_form(state, goal) # (A, n)
        return result

class BitFlippingDQNNetworkTargetHandCrafted(nn.Module):
    """
    BitFlipping DQN network (Hand Crafted) for the FlippingBitSequenceEnvRNGTarget environment.

    Same inputs and outputs, but with a handcrafted architecture that has the inductive bias of the bit flipping environment.
    That is, the Q-function should depend on how many bits are different between the state and the goal, and the action at
    the bit should depend only on the values of the bits at that index.
    """

    def __init__(self, n: int):
        super(BitFlippingDQNNetworkTargetHandCrafted, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.n = n

    def forward(self, state: torch.Tensor, goal: torch.Tensor):
        difference = (state - goal) ** 2 # shape (A, n)
        num_difference = torch.sum(difference, dim=1, keepdim=True) # shape (A, 1)
        num_difference = num_difference.expand(-1, self.n) # shape (A, n)

        # This tensor stores, for each agent and bit index, the difference between the state and the goal at that index,
        # and the number of differences between the state and the goal for that agent.
        x = torch.cat((difference.unsqueeze(2), num_difference.unsqueeze(2)), dim=2) # shape (A, n, 2)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = torch.nn.functional.gelu(self.fc2(x))
        return self.fc3(x).squeeze(2)
