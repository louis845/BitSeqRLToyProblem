import numpy as np
import torch
from torch import nn, optim
from collections import deque

from ..models.buffer_base import BufferBase

from typing import Union

class DQNHERAgent:
    """DQN agent with HER."""
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
    goal_space_size: int

    def __init__(self, model: nn.Module,
                 buffer: BufferBase,
                 device: torch.device,
                 action_space_size: int,
                 goal_space_size: int,

                 gamma: float=0.95,
                 epsilon: float=1.0,
                 epsilon_min: float=0.01,
                 epsilon_decay: float=0.995,
                 learning_rate: float=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = device
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.action_space_size = action_space_size
        self.goal_space_size = goal_space_size

        self.memory = buffer

        print("-------------------- Initializing DQNAgent --------------------")
        print("gamma: {}".format(gamma))
        print("epsilon: {}".format(epsilon))
        print("epsilon_min: {}".format(epsilon_min))
        print("epsilon_decay: {}".format(epsilon_decay))
        print("learning_rate: {}".format(learning_rate))

    def remember(self, state, action, reward, next_state, done, goal):
        self.memory.append_multiple(state, action, reward, next_state, done, goal)
    
    def expand_broadcast_goal(self, goal: torch.tensor) -> torch.tensor:
        return goal.unsqueeze(1).unsqueeze(2).expand(-1, self.action_space_size, -1)

    def act(self, state: torch.tensor, explore=True, goal: Union[int, torch.tensor]=0) -> torch.tensor:
        """Choose an action based on the current state.
        
        Args:
            state: The current state of the environment. Shape (num_agents, state_size)
            explore: Whether to randomly explore with epsilon probability. If False, the greedy action is chosen.
            goal: The goal to use for the action. Shape (num_agents,). If a single integer is provided, it is broadcasted to all agents.
            
        Returns: The action to take. Shape (num_agents,), long tensor representing the action index."""

        assert isinstance(goal, int) or (isinstance(goal, torch.Tensor) and len(goal.shape) == 1), "goal must be an integer or a 1D tensor"
        assert isinstance(goal, int) or goal.dtype == torch.int64 or goal.dtype == torch.long, "goal must be of type torch.int64 or torch.long"
        assert isinstance(goal, int) or goal.shape[0] == state.shape[0], "number of agents must be the same for state and goal"

        num_agents = state.shape[0]
        with torch.no_grad():
            state_float = state if state.dtype == torch.float32 else state.float()
            if explore:
                if isinstance(goal, int):
                    return torch.where(torch.rand(num_agents, device=self.device) <= self.epsilon,
                                torch.randint(0, self.action_space_size, (num_agents,), device=self.device),
                                self.model(state_float)[:, :, goal].argmax(dim=1))
                else:
                    return torch.where(torch.rand(num_agents, device=self.device) <= self.epsilon,
                                torch.randint(0, self.action_space_size, (num_agents,), device=self.device),
                                self.model(state_float).gather(2, self.expand_broadcast_goal(goal)).squeeze(2).argmax(dim=1))
            else:
                if isinstance(goal, int):
                    return self.model(state_float)[:, :, goal].argmax(dim=1)
                else:
                    return self.model(state_float).gather(2, self.expand_broadcast_goal(goal)).squeeze(2).argmax(dim=1)

    def replay(self, batch_size) -> float:
        if len(self.memory) < batch_size:
            return
        
        self.optimizer.zero_grad()

        # sample a minibatch from the memory (goal is omitted)
        state, action, reward, next_state, done, goal = self.memory.sample(batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)
        action = torch.tensor(action, dtype=torch.int64, device=self.device) # shape (batch_size,)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device) # shape (batch_size,)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)
        done = torch.tensor(done, dtype=torch.bool, device=self.device) # shape (batch_size,)
        goal = torch.tensor(goal, dtype=torch.int64, device=self.device) # shape (batch_size,)

        # compute the target Q value
        q_values_for_goals_next_state = self.model(next_state).gather(2, self.expand_broadcast_goal(goal)).squeeze(2) # shape (batch_size, action_space_size)
        q_values_for_goals_current_state = self.model(state).gather(2, self.expand_broadcast_goal(goal)).squeeze(2) # shape (batch_size, action_space_size)
        target = torch.where(done, reward, reward + self.gamma * q_values_for_goals_next_state.max(dim=1)[0])
        current = q_values_for_goals_current_state.gather(1, action.unsqueeze(1)).squeeze()

        loss = nn.functional.mse_loss(current, target)
        loss.backward()
        loss_value = loss.item()
        self.optimizer.step()
        
        # apply epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return loss_value