import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
import torch

class NaiveBitSequenceEnv(gym.Env):
    """
    The most naive bit sequence environment where the state space is all bit sequences of the same length, and the actions are the same as the state space.
    This means the environment is a complete graph.
    """

    def __init__(self, n: int, device: torch.device):
        super(NaiveBitSequenceEnv, self).__init__()
        self.observation_space = spaces.MultiBinary(n) # state space of all bit sequences of the same length.
        self.action_space = spaces.MultiBinary(n) # action space of going to which bit sequence. this means that the environment is a complete graph.

        self.n = n # length of the bit sequence
        self.device = device # device to run the environment on

        self.state = self.observation_space.sample() # random initial state
        self.target = self.observation_space.sample() # random target state
        self.target = torch.tensor(self.target, dtype=torch.long, device=self.device)

        self.exp_seq = torch.tensor(2 ** np.arange(n), dtype=torch.long, device=self.device)

        print("Initialized NaiveBitSequenceEnv with n = {}. Randomly picked target: {}".format(n, self.target))
    
    def bit_sequence_to_int(self, seq: torch.tensor) -> torch.tensor:
        """
        Convert a bit sequence to an integer. Assumes seq is of shape (A, n). Returns a tensor of shape (A,).
        """
        return torch.sum(seq * self.exp_seq, dim=1)
    
    def int_to_bit_sequence(self, n: torch.tensor) -> torch.tensor:
        """
        Convert an integer to a bit sequence. Assumes n is of shape (A,). Returns a tensor of shape (A, n).
        """
        return (torch.bitwise_and(n.unsqueeze(1), self.exp_seq.unsqueeze(0)) > 0).long()

    def step(self, action: torch.Tensor):
        """
        action: torch.Tensor of shape (A,), where A is the number of agents and n is the length of the bit sequence.
        It is expected that action[k] is the number representing the bit sequence that agent k wants to go to.
        """

        assert isinstance(action, torch.Tensor), "action must be a torch.Tensor"
        assert len(action.shape) == 1, "action must be of shape (A,)"
        assert action.dtype == torch.long, "action must be of type torch.long"
        if isinstance(self.state, np.ndarray):
            # the initial state is a numpy array, which means the multiple agents are not initialized yet
            self.state = self.int_to_bit_sequence(action).to(self.device)

            prev_done = torch.zeros(action.shape[0], dtype=torch.bool, device=self.device)
        else:
            # the initial state is a torch.Tensor, which means the multiple agents are already initialized
            assert action.shape[0] == self.state.shape[0], "number of agents must be the same as the number of states"

            # if the agent is already at the finish point, then the agent is already done, so we don't need to do anything
            prev_done = (self.state == self.target.unsqueeze(0)).all(dim=1)
            self.state.copy_(torch.where(prev_done.unsqueeze(1), self.state, self.int_to_bit_sequence(action)))
        
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