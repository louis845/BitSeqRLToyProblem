import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
import torch

from typing import Optional

class FlippingBitSequenceEnvRNGTarget(gym.Env):
    """
    Same as FlippingBitSequenceEnv, but with a goal for each agent. So the reward depends on the goal. The goal is represented by an integer from 0 to n-1.
    The reward function R(s, a, g) = 1 if the next state s' has taxicab distance <= goal from the target, 0 otherwise.
    """

    def __init__(self, n: int, device: torch.device):
        super(FlippingBitSequenceEnvRNGTarget, self).__init__()
        self.observation_space = spaces.MultiBinary(n) # state space of all bit sequences of the same length.
        self.action_space = spaces.Discrete(n) # action space of flipping a bit in the bit sequence. this can be an integer from 0 to n-1.

        self.n = n # length of the bit sequence
        self.device = device # device to run the environment on

        self.state = self.observation_space.sample() # random initial state
        self.target = self.observation_space.sample() # random target state

        print("Initialized FlippingBitSequenceEnvRNGTarget with n = {}".format(n))


    def distance_to_target(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculate the taxicab distance from the state to the target.
        """
        assert state.shape[0] == self.target.shape[0], "number of agents must be the same as the number of states"
        assert state.shape[1] == self.n, "state must be of shape (A, n)"
        return torch.sum(state != self.target, dim=1)
    
    def compute_reward_to_target(next_state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the reward based on the distance from the target.
        """
        assert next_state.shape[0] == target.shape[0], "number of agents must be the same as the number of states"
        assert next_state.shape[1] == target.shape[1], "next_state must be of shape (A, n)"
        return (next_state == target).all(dim=1).to(torch.float32)

    def compute_reward(self, next_state: torch.Tensor) -> torch.Tensor:
        """
        Calculate the reward based on the distance from the target.
        """
        return FlippingBitSequenceEnvRNGTarget.compute_reward_to_target(next_state, self.target)

    def step(self, action: torch.Tensor):
        """
        action: torch.Tensor of shape (A,), where A is the number of agents and n is the length of the bit sequence.
        It is expected that action[k] is the index of the bit to flip for agent k.

        Returns: next_state, reward, done, info
        
        next_state: torch.Tensor of shape (A, n), where A is the number of agents and n is the length of the bit sequence.
        reward: torch.Tensor of shape (A,), where A is the number of agents, corresponding to the ultimate reward (1 if the agent reaches the target, 0 otherwise).
        done: torch.Tensor of shape (A,), where A is the number of agents, corresponding to whether the agent has reached the target.
        info: dictionary containing additional information.
        info["previous_done"]: torch.Tensor of shape (A,), where A is the number of agents, corresponding to whether the agent was already done in the previous step.
        info["distance"]: torch.Tensor of shape (A,), where A is the number of agents, corresponding to the taxicab distance from the target.
        """

        assert isinstance(action, torch.Tensor), "action must be a torch.Tensor"
        assert len(action.shape) == 1, "action must be of shape (A,)"
        assert action.dtype == torch.long, "action must be of type torch.long"

        # Update state
        # the initial state is a torch.Tensor, which means the multiple agents are already initialized
        assert action.shape[0] == self.state.shape[0], "number of agents must be the same as the number of states"

        # if the agent is already at the finish point, then the agent is already done, so we don't need to do anything
        prev_done = (self.state == self.target).all(dim=1)
        # flip the bit at the action index, for each agent where the agent is not done
        needs_flip = (~prev_done).unsqueeze(1) & torch.nn.functional.one_hot(action, num_classes=self.n).to(torch.bool)
        self.state = torch.where(needs_flip, 1 - self.state, self.state) # update the state here
        
        # Calculate reward. Reward is 1 if the state is the same as the target, 0 otherwise.
        # Since there are multiple agents, the reward is a tensor of shape (A,).
        done = (self.state == self.target).all(dim=1)
        reward = self.compute_reward(self.state)
        
        # Calculate per-goal per-agent rewards. Reward is 0 if the agent is already done.
        taxicab_distance = self.distance_to_target(self.state) # taxicab distance from the state to the target. Shape: (A,)
        info = {"previous_done": prev_done, "distance": taxicab_distance}
        return self.state, reward, done, info

    def reset(self, num_agents: int=1) -> tuple[torch.tensor, torch.tensor]:
        # randomly initialize the initial state and the target state
        self.state = torch.tensor(
            np.stack([self.observation_space.sample() for _ in range(num_agents)], axis=0),
            dtype=torch.long, device=self.device)
        self.target = torch.tensor(
            np.stack([self.observation_space.sample() for _ in range(num_agents)], axis=0),
            dtype=torch.long, device=self.device
        )
        return self.state, self.target

    def render(self, mode='human'):
        print(f"Current State: {self.state}")

    def close(self):
        pass
    