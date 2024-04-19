import numpy as np
import torch
from torch import nn, optim
from collections import deque

from ..models.buffer_base import BufferBase

class DQNAgentTarget:
    """Simple DQN agent with moving target."""
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

                 tau: float=0.99,
                 gamma: float=0.9,
                 epsilon: float=1.0,
                 epsilon_min: float=0.1,
                 epsilon_decay: float=0.999,
                 learning_rate: float=0.001,
                 weight_decay: float=0.0001):
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
    
    def remember_trajectory(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                            next_state: torch.Tensor, done: torch.Tensor, goals: torch.Tensor,
                            step_valid: torch.Tensor):
        """
        For each tensor, assumes the first dimension is the length of the trajectory.
        step_valid is a 2D bool tensor of shape (T, A) = (trajectory length, number of agents) indicating
        whether the step is valid for each agent at each time step.
        """
        tensors = [state, action, reward, next_state, done, goals, step_valid]
        assert len(set([t.shape[0] for t in tensors])) == 1, "all tensors must have the same trajectory length"
        assert len(set([t.shape[1] for t in tensors])) == 1, "all tensors must have the same number of agents"
        assert step_valid.dtype == torch.bool, "step_valid must be of type torch.bool"

        T, A = state.shape[0], state.shape[1]
        # remember the steps in trajectory
        for t in range(T):
            valid_agents = step_valid[t, :]
            if step_valid[t, :].any():
                self.memory.append_multiple_at_once(state[t, valid_agents, :].cpu().numpy(),
                                                    action[t, valid_agents].cpu().numpy(),
                                                    reward[t, valid_agents].cpu().numpy(),
                                                    next_state[t, valid_agents, :].cpu().numpy(),
                                                    done[t, valid_agents].cpu().numpy(),
                                                    goals[t, valid_agents, :].cpu().numpy())

    def remember(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray,
                 next_state: np.ndarray, done: np.ndarray, goals: np.ndarray):
        self.memory.append_multiple_at_once(state, action, reward, next_state, done, goals)

    def act(self, state: torch.Tensor, goals: torch.Tensor, explore=True) -> torch.Tensor:
        """Choose an action based on the current state.
        
        Args:
            state: The current state of the environment. Shape (num_agents, state_size)
            goals: The goal state of the environment. Shape (num_agents, state_size)
            
        Returns: The action to take. Shape (num_agents,), long tensor representing the action index."""
        num_agents = state.shape[0]
        with torch.no_grad():
            state_float = state if state.dtype == torch.float32 else state.to(torch.float32)
            goals_float = goals if goals.dtype == torch.float32 else goals.to(torch.float32)
            if explore:
                return torch.where(torch.rand(num_agents, device=self.device) <= self.epsilon,
                            torch.randint(0, self.action_space_size, (num_agents,), device=self.device),
                            self.model(state_float, goals_float).argmax(dim=1))
            else:
                return self.model(state_float, goals_float).argmax(dim=1)

    def replay(self, batch_size, opt_steps=5):
        cum_loss = 0.0
        for _ in range(opt_steps):
            self.optimizer.zero_grad()

            # sample a minibatch from the memory (goal is omitted)
            state, action, reward, next_state, done, goals = self.memory.sample(batch_size)
            state = torch.tensor(state, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)
            action = torch.tensor(action, dtype=torch.int64, device=self.device) # shape (batch_size,)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device) # shape (batch_size,)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)
            done = torch.tensor(done, dtype=torch.bool, device=self.device) # shape (batch_size,)
            goals = torch.tensor(goals, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)

            # compute the target Q value
            with torch.no_grad():
                target = torch.where(done, reward, reward + self.gamma * self.running_model(next_state, goals).max(dim=1)[0])
                target = target.clip(0.0, 1.0)
            current = self.model(state, goals).gather(1, action.unsqueeze(1)).squeeze()

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