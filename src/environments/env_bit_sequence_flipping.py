import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
import torch

class FlippingBitSequenceEnv(gym.Env):
    """
    Bit sequence environment where the state space is all bit sequences of the same length, and the actions are flipping a bit in the bit sequence.
    The target is a randomly chosen bit sequence. The graph is still connected, but there are multiple steps the agent has to take to reach the target.
    """

    def __init__(self, n: int, device: torch.device):
        super(FlippingBitSequenceEnv, self).__init__()
        self.observation_space = spaces.MultiBinary(n) # state space of all bit sequences of the same length.
        self.action_space = spaces.Discrete(n) # action space of flipping a bit in the bit sequence. this can be an integer from 0 to n-1.

        self.n = n # length of the bit sequence
        self.device = device # device to run the environment on

        self.state = self.observation_space.sample() # random initial state
        self.target = self.observation_space.sample() # random target state
        self.target = torch.tensor(self.target, dtype=torch.long, device=self.device)

        print("Initialized FlippingBitSequenceEnv with n = {}. Randomly picked target: {}".format(n, self.target))

    def step(self, action: torch.Tensor):
        """
        action: torch.Tensor of shape (A,), where A is the number of agents and n is the length of the bit sequence.
        It is expected that action[k] is the index of the bit to flip for agent k.
        """

        assert isinstance(action, torch.Tensor), "action must be a torch.Tensor"
        assert len(action.shape) == 1, "action must be of shape (A,)"
        assert action.dtype == torch.long, "action must be of type torch.long"
        if isinstance(self.state, np.ndarray):
            # the initial state is a numpy array, which means the multiple agents are not initialized yet
            self.state = torch.tensor(self.state, dtype=torch.long, device=self.device).unsqueeze(0).expand(action.shape[0], -1) # broadcast the initial state to all agents
            # flip the bit at the action index
            self.state[torch.arange(action.shape[0], device=self.device, dtype=action.dtype), action] =\
                1 - self.state[torch.arange(action.shape[0], device=self.device, dtype=action.dtype), action]
            
            prev_done = torch.zeros(action.shape[0], dtype=torch.bool, device=self.device)
        else:
            # the initial state is a torch.Tensor, which means the multiple agents are already initialized
            assert action.shape[0] == self.state.shape[0], "number of agents must be the same as the number of states"

            # if the agent is already at the finish point, then the agent is already done, so we don't need to do anything
            prev_done = (self.state == self.target.unsqueeze(0)).all(dim=1)
            # flip the bit at the action index, for each agent where the agent is not done
            needs_flip = (~prev_done).unsqueeze(1) & torch.nn.functional.one_hot(action, num_classes=self.n).to(torch.bool)
            self.state = torch.where(needs_flip, 1 - self.state, self.state)
        
        # Calculate reward. Reward is 1 if the state is the same as the target, 0 otherwise.
        # Since there are multiple agents, the reward is a tensor of shape (A,).
        done = (self.state == self.target.unsqueeze(0)).all(dim=1)
        reward = torch.where(prev_done, torch.zeros_like(prev_done, dtype=torch.float32),
                             done.to(torch.float32))
        info = {"previous_done": prev_done}
        return self.state, reward, done, info

    def reset(self, num_agents: int=1):
        self.state = torch.tensor(
            np.stack([self.observation_space.sample() for _ in range(num_agents)], axis=0),
            dtype=torch.long, device=self.device)
        return self.state

    def render(self, mode='human'):
        print(f"Current State: {self.state}")

    def close(self):
        pass