import numpy as np
import torch
from torch import nn, optim
from collections import deque

from ..models.buffer_base import BufferBase
from ..models.buffer_bitflipping_goals_multirew import BufferBitflippingGoalsMultiREW
from ..models.model_dqn_bitflipping_goals import BitFlippingDQNNetworkGoals, BitFlippingDQNNetworkGoalsUVFA

from typing import Union

class DQNHERAgent:
    """DQN agent with HER."""
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
    goal_space_size: int
    input_type: str # either ["state", "state_goal", "state_goal_uvfa"]
    is_multi_reward_buffer: bool

    def __init__(self, model: nn.Module,
                 running_model: nn.Module,
                 buffer: BufferBase,
                 device: torch.device,
                 action_space_size: int,
                 goal_space_size: int,

                 tau: float=0.995,
                 gamma: float=0.95,
                 epsilon: float=1.0,
                 epsilon_min: float=0.05,
                 epsilon_decay: float=0.999,
                 learning_rate: float=0.001):
        assert (not isinstance(buffer, BufferBitflippingGoalsMultiREW)) or\
            isinstance(model, BitFlippingDQNNetworkGoalsUVFA) or not model.input_goal, "BufferBitflippingGoalsMultiREW requires BitFlippingDQNNetworkGoalsUVFA or input_goal=False"

        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = device
        self.model = model
        self.running_model = running_model
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

        if isinstance(model, BitFlippingDQNNetworkGoals):
            if model.input_goal:
                self.input_type = "state_goal"
            else:
                self.input_type = "state"
        elif isinstance(model, BitFlippingDQNNetworkGoalsUVFA):
            self.input_type = "state_goal_uvfa"
        else:
            raise ValueError("model must be an instance of BitFlippingDQNNetworkGoals or BitFlippingDQNNetworkGoalsUVFA")
        
        self.is_multi_reward_buffer = isinstance(buffer, BufferBitflippingGoalsMultiREW)

        # set parameters of running model to be the same as the model
        self.running_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done, goal):
        self.memory.append_multiple_at_once(state, action, reward, next_state, done, goal)
    
    def expand_broadcast_goal(self, goal: torch.tensor) -> torch.tensor:
        return goal.unsqueeze(1).unsqueeze(2).expand(-1, self.action_space_size, -1)
    
    def cat_state_goal(self, state: torch.tensor, goal: torch.tensor) -> torch.tensor:
        return torch.cat([state, torch.nn.functional.one_hot(goal, num_classes=self.goal_space_size).to(torch.float32)], dim=1)

    def act(self, state: torch.tensor, explore=True, explore_mode="none",
            goal: Union[int, torch.tensor]=0) -> torch.tensor:
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
            if self.input_type == "state_goal":
                if isinstance(goal, int):
                    goal = torch.tensor([goal] * num_agents, dtype=torch.int64, device=self.device)
                logits = self.model(self.cat_state_goal(state_float, goal))
            elif self.input_type == "state_goal_uvfa":
                if isinstance(goal, int):
                    goal = torch.tensor([goal] * num_agents, dtype=torch.int64, device=self.device)
                logits = self.model(state_float, goal)
            else:
                if isinstance(goal, int):
                    logits = self.model(state_float)[:, :, goal]
                else:
                    logits = self.model(state_float).gather(2, self.expand_broadcast_goal(goal)).squeeze(2)
            if explore:
                if explore_mode == "logits_temperature":
                    # uniform in [0.01, 0.1)
                    temperature = (torch.rand(num_agents, device=self.device) * 0.09 + 0.01).unsqueeze(1)
                    return torch.where(torch.rand(num_agents, device=self.device) <= self.epsilon,
                                torch.randint(0, self.action_space_size, (num_agents,), device=self.device),
                                torch.distributions.Categorical(logits=logits / temperature).sample())
                else:
                    return torch.where(torch.rand(num_agents, device=self.device) <= self.epsilon,
                                torch.randint(0, self.action_space_size, (num_agents,), device=self.device),
                                logits.argmax(dim=1))

            else:
                if isinstance(goal, int):
                    return logits.argmax(dim=1)
                else:
                    return logits.argmax(dim=1)

    def replay(self, batch_size, opt_steps:int=20) -> float:
        cum_loss = 0.0
        for _ in range(opt_steps):
            self.optimizer.zero_grad()

            # sample a minibatch from the memory (goal is omitted)
            state, action, reward, next_state, done, goal = self.memory.sample(batch_size)
            state = torch.tensor(state, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)
            action = torch.tensor(action, dtype=torch.int64, device=self.device) # shape (batch_size,)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            # shape (batch_size, goal_space_size) if multi-reward buffer, else (batch_size,)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device) # shape (batch_size, state_size)
            done = torch.tensor(done, dtype=torch.bool, device=self.device) # shape (batch_size,)

            if self.is_multi_reward_buffer:
                # compute the target Q value
                if self.input_type == "state_goal_uvfa":
                    with torch.no_grad():
                        q_values_for_goals_next_state = self.running_model.compute_full_goals(next_state)
                    q_values_for_goals_current_state = self.model.compute_full_goals(state)
                else:
                    with torch.no_grad():
                        q_values_for_goals_next_state = self.running_model(next_state) # shape (batch_size, action_space_size, goal_space_size)
                    q_values_for_goals_current_state = self.model(state) # shape (batch_size, action_space_size, goal_space_size)
                target = torch.where(done.unsqueeze(1), reward, reward + self.gamma * q_values_for_goals_next_state.max(dim=1)[0])
                current = q_values_for_goals_current_state.gather(1, action.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.goal_space_size)).squeeze(dim=1)
            else:
                goal = torch.tensor(goal, dtype=torch.int64, device=self.device) # shape (batch_size,)
                # compute the target Q value
                if self.input_type == "state_goal":
                    with torch.no_grad():
                        q_values_for_goals_next_state = self.running_model(self.cat_state_goal(next_state, goal)) # shape (batch_size, action_space_size)
                    q_values_for_goals_current_state = self.model(self.cat_state_goal(state, goal)) # shape (batch_size, action_space_size)
                elif self.input_type == "state_goal_uvfa":
                    with torch.no_grad():
                        q_values_for_goals_next_state = self.running_model(next_state, goal)
                    q_values_for_goals_current_state = self.model(state, goal)
                else:
                    with torch.no_grad():
                        q_values_for_goals_next_state = self.running_model(next_state).gather(2, self.expand_broadcast_goal(goal)).squeeze(2) # shape (batch_size, action_space_size)
                    q_values_for_goals_current_state = self.model(state).gather(2, self.expand_broadcast_goal(goal)).squeeze(2) # shape (batch_size, action_space_size)
                target = torch.where(done, reward, reward + self.gamma * q_values_for_goals_next_state.max(dim=1)[0])
                current = q_values_for_goals_current_state.gather(1, action.unsqueeze(1)).squeeze()

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