import numpy as np
import torch
from torch import nn, optim
from collections import deque

from ..models.buffer_base import BufferBase

class DQNAgent:
    """Simple DQN agent."""
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    learning_rate: float

    model: nn.Module
    optimizer: optim.Optimizer
    memory: BufferBase
    device: torch.device

    action_space_size: int

    def __init__(self, model: nn.Module,
                 buffer: BufferBase,
                 device: torch.device,
                 action_space_size: int,

                 gamma: float=0.95,
                 epsilon: float=1.0,
                 epsilon_min: float=0.01,
                 epsilon_decay: float=0.995,
                 learning_rate: float=0.05):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = device
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-4)
        self.action_space_size = action_space_size

        self.memory = buffer

        print("-------------------- Initializing DQNAgent --------------------")
        print("gamma: {}".format(gamma))
        print("epsilon: {}".format(epsilon))
        print("epsilon_min: {}".format(epsilon_min))
        print("epsilon_decay: {}".format(epsilon_decay))
        print("learning_rate: {}".format(learning_rate))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append_multiple_at_once(state, action, reward, next_state, done)

    def act(self, state: torch.tensor, explore=True) -> torch.tensor:
        """Choose an action based on the current state.
        
        Args:
            state: The current state of the environment. Shape (num_agents, state_size)
            
        Returns: The action to take. Shape (num_agents,), long tensor representing the action index."""
        num_agents = state.shape[0]
        with torch.no_grad():
            state_float = state if state.dtype == torch.float32 else state.float()
            if explore:
                return torch.where(torch.rand(num_agents, device=self.device) <= self.epsilon,
                            torch.randint(0, self.action_space_size, (num_agents,), device=self.device),
                            self.model(state_float).argmax(dim=1))
            else:
                return self.model(state_float).argmax(dim=1)

    def replay(self, batch_size):
        self.optimizer.zero_grad()

        # sample a minibatch from the memory (goal is omitted)
        state, action, reward, next_state, done, _ = self.memory.sample(batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)
        action = torch.tensor(action, dtype=torch.int64, device=self.device) # shape (batch_size,)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device) # shape (batch_size,)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)
        done = torch.tensor(done, dtype=torch.bool, device=self.device) # shape (batch_size,)

        # compute the target Q value
        target = torch.where(done, reward, reward + self.gamma * self.model(next_state).max(dim=1)[0])
        current = self.model(state).gather(1, action.unsqueeze(1)).squeeze()

        loss = nn.functional.mse_loss(current, target)
        loss.backward()
        self.optimizer.step()
        
        # apply epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return loss.item()