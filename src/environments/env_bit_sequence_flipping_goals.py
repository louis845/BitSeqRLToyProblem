import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
import torch

from typing import Optional

class FlippingBitSequenceEnvWithGoals(gym.Env):
    """
    Same as FlippingBitSequenceEnv, but with a goal for each agent. So the reward depends on the goal. The goal is represented by an integer from 0 to n-1.
    The reward function R(s, a, g) = 1 if the next state s' has taxicab distance <= goal from the target, 0 otherwise.
    """

    def __init__(self, n: int, device: torch.device, fix_original_state: bool=False):
        super(FlippingBitSequenceEnvWithGoals, self).__init__()
        self.observation_space = spaces.MultiBinary(n) # state space of all bit sequences of the same length.
        self.action_space = spaces.Discrete(n) # action space of flipping a bit in the bit sequence. this can be an integer from 0 to n-1.

        self.n = n # length of the bit sequence
        self.device = device # device to run the environment on

        self.state = self.observation_space.sample() # random initial state
        self.target = self.observation_space.sample() # random target state
        distance = (self.state != self.target).sum()
        while distance == 0:
            self.target = self.observation_space.sample()
            distance = (self.state != self.target).sum()
        self.original_initial_state = self.state
        self.target = torch.tensor(self.target, dtype=torch.long, device=self.device)

        print("Initialized FlippingBitSequenceEnvWithGoals with n = {}. Randomly picked target: {}".format(n, self.target))
        print("Initial distance to target: {}".format(distance))

        self.fix_original_state = fix_original_state
        self.initial_distance = distance

    
    def distance_to_target(self, state: torch.tensor) -> torch.tensor:
        """
        Calculate the taxicab distance from the state to the target.
        """
        return (state != self.target.unsqueeze(0)).sum(dim=1)

    def step(self, action: torch.Tensor, goals: Optional[torch.Tensor]=None):
        """
        action: torch.Tensor of shape (A,), where A is the number of agents and n is the length of the bit sequence.
        It is expected that action[k] is the index of the bit to flip for agent k.

        goals: torch.Tensor of shape (A, G), where A is the number of agents, and G is the number of goals to compute for each agent.
        It is expected that goals[k, g] is the goal g for agent k, which is an integer from 0 to n-1.

        Returns: next_state, reward, done, info
        
        next_state: torch.Tensor of shape (A, n), where A is the number of agents and n is the length of the bit sequence.
        reward: torch.Tensor of shape (A,), where A is the number of agents, corresponding to the ultimate reward (1 if the agent reaches the target, 0 otherwise).
        done: torch.Tensor of shape (A,), where A is the number of agents, corresponding to whether the agent has reached the target.
        info: dictionary containing additional information.
        info["previous_done"]: torch.Tensor of shape (A,), where A is the number of agents, corresponding to whether the agent was already done in the previous step.
        info["goal_rewards"]: torch.Tensor of shape (A, G), where A is the number of agents, and G is the number of goals to compute for each agent. The reward for each goal.
        """

        assert isinstance(action, torch.Tensor), "action must be a torch.Tensor"
        assert len(action.shape) == 1, "action must be of shape (A,)"
        assert action.dtype == torch.long, "action must be of type torch.long"
        if goals is not None:
            assert isinstance(goals, torch.Tensor), "goals must be a torch.Tensor"
            assert len(goals.shape) == 2, "goals must be of shape (A, G)"
            assert action.shape[0] == goals.shape[0], "number of agents must be the same for action and goals"

        # Update state
        # the initial state is a torch.Tensor, which means the multiple agents are already initialized
        assert action.shape[0] == self.state.shape[0], "number of agents must be the same as the number of states"

        # if the agent is already at the finish point, then the agent is already done, so we don't need to do anything
        prev_done = (self.state == self.target.unsqueeze(0)).all(dim=1)
        prev_taxicab_distance = self.distance_to_target(self.state) # Shape: (A,)
        # flip the bit at the action index, for each agent where the agent is not done
        needs_flip = (~prev_done).unsqueeze(1) & torch.nn.functional.one_hot(action, num_classes=self.n).to(torch.bool)
        self.state = torch.where(needs_flip, 1 - self.state, self.state) # update the state here
        
        # Calculate reward. Reward is 1 if the state is the same as the target, 0 otherwise.
        # Since there are multiple agents, the reward is a tensor of shape (A,).
        done = (self.state == self.target.unsqueeze(0)).all(dim=1)
        reward = torch.where(prev_done, torch.zeros_like(prev_done, dtype=torch.float32),
                             done.to(torch.float32))
        
        # Calculate per-goal per-agent rewards. Reward is 0 if the agent is already done.
        taxicab_distance = self.distance_to_target(self.state) # taxicab distance from the state to the target. Shape: (A,)
        if goals is None:
            goal_rewards = None
        else:
            goal_rewards = torch.where(prev_done.unsqueeze(1),
                torch.zeros_like(prev_done, dtype=torch.float32).unsqueeze(1),
                (taxicab_distance.unsqueeze(1) <= goals).to(torch.float32))
        info = {"previous_done": prev_done, "goal_rewards": goal_rewards, "distance": taxicab_distance}
        return self.state, reward, done, info

    def reset(self, num_agents: int=1):
        if self.fix_original_state:
            self.state = torch.tensor(
                np.stack([self.original_initial_state for _ in range(num_agents)], axis=0),
                dtype=torch.long, device=self.device)
        else:
            self.state = torch.tensor(
                np.stack([self.observation_space.sample() for _ in range(num_agents)], axis=0),
                dtype=torch.long, device=self.device)
        return self.state

    def render(self, mode='human'):
        print(f"Current State: {self.state}")

    def close(self):
        pass

    def get_initial_distance(self) -> int:
        return self.initial_distance