import numpy as np
import torch
from torch import nn, optim
from collections import deque

from ..models.buffer_base import BufferBase
from .agent_dqn_target import DQNAgentTarget
from ..environments.env_bit_sequence_flipping_rng_target import FlippingBitSequenceEnvRNGTarget

class DQNAgentTargetHER(DQNAgentTarget):
    def __init__(self, *args, **kwargs):
        super(DQNAgentTargetHER, self).__init__(*args, **kwargs)
    
    def remember_trajectory(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                            next_state: torch.Tensor, done: torch.Tensor, goals: torch.Tensor,
                            step_valid: torch.Tensor):
        """
        For each tensor, assumes the first dimension is the length of the trajectory.
        step_valid is a 2D bool tensor of shape (T, A) = (trajectory length, number of agents) indicating
        whether the step is valid for each agent at each time step.
        """
        super(DQNAgentTargetHER, self).remember_trajectory(state, action, reward, next_state, done, goals, step_valid)

        T = state.shape[0]
        # choose agents for HER with 30% probability
        agents_for_her = step_valid.any(dim=0) & (torch.rand(step_valid.shape[1], device=self.device) < 0.3)
        if agents_for_her.sum() == 0:
            return
        state = state[:, agents_for_her, ...]
        action = action[:, agents_for_her, ...]
        reward = reward[:, agents_for_her, ...]
        next_state = next_state[:, agents_for_her, ...]
        done = done[:, agents_for_her, ...]
        goals = goals[:, agents_for_her, ...]
        step_valid = step_valid[:, agents_for_her, ...]
        A = state.shape[1]

        # choose a random future timestep to replace the goal
        max_future = torch.sum(step_valid, dim=0).cpu().numpy()
        future_index = torch.tensor(np.random.randint(0, max_future, size=A), device=self.device, dtype=torch.long)
        future_goals = torch.zeros([A] + list(goals.shape[2:]), device=self.device, dtype=goals.dtype)
        for a in range(A):
            future_goals[a, ...] = next_state[future_index[a], a, ...]

        # remember the steps in trajectory
        for t in range(T):
            goal_already_achieved = (state[t, :, :] == future_goals).all(dim=1) # if the agent has already achieved the (new) goal
            if t > 0:
                valid_agents = step_valid[t, :] & ~goal_already_achieved & valid_agents
            else:
                valid_agents = step_valid[t, :] & ~goal_already_achieved
            if valid_agents.any():
                with torch.no_grad():
                    new_reward = FlippingBitSequenceEnvRNGTarget.compute_reward_to_target(
                        next_state[t, valid_agents, :], future_goals[valid_agents, :])
                    new_done = (new_reward > 0)
                
                self.memory.append_multiple_at_once(state[t, valid_agents, :].cpu().numpy(),
                                                    action[t, valid_agents].cpu().numpy(),
                                                    new_reward.cpu().numpy(),
                                                    next_state[t, valid_agents, :].cpu().numpy(),
                                                    new_done.cpu().numpy(),
                                                    future_goals[valid_agents, :].cpu().numpy())
        