import numpy as np
import torch
from torch import nn, optim
from collections import deque

from ..models.buffer_base import BufferBase

class DQNAgent:
    """Simple DQN agent."""
    tau: float
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    learning_rate: float

    model: nn.Module
    running_model: nn.Module
    optimizer: optim.Optimizer
    memory: BufferBase
    device: torch.device

    action_space_size: int

    def __init__(self, model: nn.Module,
                 running_model: nn.Module,
                 buffer: BufferBase,
                 device: torch.device,
                 action_space_size: int,

                 tau: float=0.995,
                 gamma: float=0.95,
                 epsilon: float=1.0,
                 epsilon_min: float=0.05,
                 epsilon_decay: float=0.999,
                 learning_rate: float=0.001):
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = device
        self.model = model
        self.running_model = running_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        self.action_space_size = action_space_size

        self.memory = buffer

        print("-------------------- Initializing DQNAgent --------------------")
        print("gamma: {}".format(gamma))
        print("epsilon: {}".format(epsilon))
        print("epsilon_min: {}".format(epsilon_min))
        print("epsilon_decay: {}".format(epsilon_decay))
        print("learning_rate: {}".format(learning_rate))

        self.running_model.load_state_dict(self.model.state_dict())

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

    def replay(self, batch_size, opt_steps=20):
        cum_loss = 0.0
        for _ in range(opt_steps):
            self.optimizer.zero_grad()

            # sample a minibatch from the memory (goal is omitted)
            state, action, reward, next_state, done, _ = self.memory.sample(batch_size)
            state = torch.tensor(state, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)
            action = torch.tensor(action, dtype=torch.int64, device=self.device) # shape (batch_size,)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device) # shape (batch_size,)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)
            done = torch.tensor(done, dtype=torch.bool, device=self.device) # shape (batch_size,)

            # compute the target Q value
            with torch.no_grad():
                target = torch.where(done, reward, reward + self.gamma * self.running_model(next_state).max(dim=1)[0])
            current = self.model(state).gather(1, action.unsqueeze(1)).squeeze()

            loss = nn.functional.mse_loss(current, target)
            loss.backward()
            self.optimizer.step()
            
            # apply epsilon decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            cum_loss += loss.item()
        
            # do polyak averaging
            with torch.no_grad():
                running_model_state_dict = self.running_model.state_dict()
                model_state_dict = self.model.state_dict()
                for key in running_model_state_dict:
                    running_model_state_dict[key].copy_(self.tau * running_model_state_dict[key] + (1 - self.tau) * model_state_dict[key])

        return cum_loss / opt_steps