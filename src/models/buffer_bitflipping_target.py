import gc

import numpy as np
import torch

from .buffer_base import BufferBase
from ..utils import bit_sequence_to_int, int_to_bit_sequence

class BufferBitflippingTarget(BufferBase):
    """Simple replay buffer with target, with amoritized O(1) append access. This replay buffer corresponds to the bitflipping target DQN model."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    goals: np.ndarray
    buffer_size: int

    n: int
    no_repeat: bool
    device: torch.device
    n_exp: np.ndarray
    max_buffer_size: int

    def __init__(self, n: int, device: torch.device = torch.device("cpu"),
                 max_buffer_size: int = None):
        self.n = n
        self.device = device

        self.states = np.zeros((0, n), dtype=np.int32) # state space is stored in 0, 1 format
        self.actions = np.zeros((0,), dtype=np.int32) # action space is stored in integer format
        self.rewards = np.zeros((0,), dtype=np.float32)
        self.next_states = np.zeros((0, n), dtype=np.int32)
        self.dones = np.zeros((0,), dtype=bool)
        self.goals = np.zeros((0, n), dtype=np.int32) # state space is stored in 0, 1 format
        self.buffer_size = 0
        self.max_buffer_size = max_buffer_size
    
    def __len__(self) -> int:
        return self.buffer_size

    def append(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, goal: np.ndarray):
        assert state.shape == (self.n,), "state must have the same length as the buffer. state: {}, buffer: {}".format(state.shape, self.n)
        assert next_state.shape == (self.n,), "next_state must have the same length as the buffer, next_state: {}, buffer: {}".format(next_state.shape, self.n)
        assert goal.shape == (self.n,), "goal must have the same length as the buffer, goal: {}, buffer: {}".format(goal.shape, self.n)

        if self.buffer_size == 0:
            self.states = state.reshape(1, -1).copy()
            self.actions = np.array([action], dtype=np.int32)
            self.rewards = np.array([reward], dtype=np.float32)
            self.next_states = next_state.reshape(1, -1).copy()
            self.dones = np.array([done], dtype=bool)
            self.goals = goal.reshape(1, -1).copy()
        else:
            if self.buffer_size == self.states.shape[0]: # expand the buffer
                new_states = np.zeros((2 * self.buffer_size, self.n), dtype=np.int32)
                new_actions = np.zeros(2 * self.buffer_size, dtype=np.int32)
                new_rewards = np.zeros(2 * self.buffer_size, dtype=np.float32)
                new_next_states = np.zeros((2 * self.buffer_size, self.n), dtype=np.int32)
                new_dones = np.zeros(2 * self.buffer_size, dtype=bool)
                new_goals = np.zeros((2 * self.buffer_size, self.n), dtype=np.int32)
                new_states[:self.buffer_size, :] = self.states
                new_actions[:self.buffer_size] = self.actions
                new_rewards[:self.buffer_size] = self.rewards
                new_next_states[:self.buffer_size, :] = self.next_states
                new_dones[:self.buffer_size] = self.dones
                new_goals[:self.buffer_size, :] = self.goals
                del self.states, self.actions, self.rewards, self.next_states, self.dones, self.goals
                self.states = new_states
                self.actions = new_actions
                self.rewards = new_rewards
                self.next_states = new_next_states
                self.dones = new_dones
                self.goals = new_goals
                gc.collect()
            
            if self.no_repeat:
                self.states[self.buffer_size, :].copy_(torch.tensor(state, dtype=torch.int32))
                self.actions[self.buffer_size].copy_(torch.tensor(action, dtype=torch.int32))
                self.rewards[self.buffer_size].copy_(torch.tensor(reward, dtype=torch.float32))
                self.next_states[self.buffer_size, :].copy_(torch.tensor(next_state, dtype=torch.int32))
                self.dones[self.buffer_size].copy_(torch.tensor(done, dtype=torch.bool))
                self.goals[self.buffer_size, :].copy_(torch.tensor(goal, dtype=torch.int32))
            else:
                self.states[self.buffer_size, :] = state
                self.actions[self.buffer_size] = action
                self.rewards[self.buffer_size] = reward
                self.next_states[self.buffer_size, :] = next_state
                self.dones[self.buffer_size] = done
                self.goals[self.buffer_size, :] = goal
            
        self.buffer_size += 1

        self.shrink_buffer_if_needed()
    
    def append_multiple_at_once(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray, goals: np.ndarray):
        if self.buffer_size == 0:
            # find lowest multiple of 2 that is larger or equal to the number of unique states
            new_length = 2 ** np.ceil(np.log2(states.shape[0])).astype(np.int32)
            self.states = np.zeros((new_length, self.n), dtype=np.int32)
            self.actions = np.zeros(new_length, dtype=np.int32)
            self.rewards = np.zeros(new_length, dtype=np.float32)
            self.next_states = np.zeros((new_length, self.n), dtype=np.int32)
            self.dones = np.zeros(new_length, dtype=bool)
            self.goals = np.zeros((new_length, self.n), dtype=np.int32)
            self.buffer_size = states.shape[0]
            # set the values
            self.states[:self.buffer_size, :] = states
            self.actions[:self.buffer_size] = actions
            self.rewards[:self.buffer_size] = rewards
            self.next_states[:self.buffer_size, :] = next_states
            self.dones[:self.buffer_size] = dones
            self.goals[:self.buffer_size, :] = goals
        else:
            new_length = self.states.shape[0]
            while new_length < self.buffer_size + states.shape[0]:
                new_length *= 2
            if new_length > self.states.shape[0]: # if requires growing, grow
                new_states = np.zeros((new_length, self.n), dtype=np.int32)
                new_actions = np.zeros(new_length, dtype=np.int32)
                new_rewards = np.zeros(new_length, dtype=np.float32)
                new_next_states = np.zeros((new_length, self.n), dtype=np.int32)
                new_dones = np.zeros(new_length, dtype=bool)
                new_goals = np.zeros((new_length, self.n), dtype=np.int32)
                new_states[:self.buffer_size, :] = self.states[:self.buffer_size, :]
                new_actions[:self.buffer_size] = self.actions[:self.buffer_size]
                new_rewards[:self.buffer_size] = self.rewards[:self.buffer_size]
                new_next_states[:self.buffer_size, :] = self.next_states[:self.buffer_size, :]
                new_dones[:self.buffer_size] = self.dones[:self.buffer_size]
                new_goals[:self.buffer_size, :] = self.goals[:self.buffer_size, :]
                del self.states, self.actions, self.rewards, self.next_states, self.dones, self.goals
                self.states = new_states
                self.actions = new_actions
                self.rewards = new_rewards
                self.next_states = new_next_states
                self.dones = new_dones
                self.goals = new_goals
                gc.collect()
            # set the values
            self.states[self.buffer_size:self.buffer_size + states.shape[0], :] = states
            self.actions[self.buffer_size:self.buffer_size + states.shape[0]] = actions
            self.rewards[self.buffer_size:self.buffer_size + states.shape[0]] = rewards
            self.next_states[self.buffer_size:self.buffer_size + states.shape[0], :] = next_states
            self.dones[self.buffer_size:self.buffer_size + states.shape[0]] = dones
            self.goals[self.buffer_size:self.buffer_size + states.shape[0], :] = goals
            # update buffer size
            self.buffer_size += states.shape[0]
        self.shrink_buffer_if_needed()
    
    def shrink_buffer_if_needed(self):
        if self.max_buffer_size is not None and self.buffer_size > self.max_buffer_size:
            # keep the latest max_buffer_size elements
            self.states[:self.max_buffer_size, :] = self.states[self.buffer_size - self.max_buffer_size:self.buffer_size, :]
            self.actions[:self.max_buffer_size] = self.actions[self.buffer_size - self.max_buffer_size:self.buffer_size:]
            self.rewards[:self.max_buffer_size] = self.rewards[self.buffer_size - self.max_buffer_size:self.buffer_size:]
            self.next_states[:self.max_buffer_size, :] = self.next_states[self.buffer_size - self.max_buffer_size:self.buffer_size:, :]
            self.dones[:self.max_buffer_size] = self.dones[self.buffer_size - self.max_buffer_size:self.buffer_size:]
            self.goals[:self.max_buffer_size, :] = self.goals[self.buffer_size - self.max_buffer_size:self.buffer_size:, :]
            self.buffer_size = self.max_buffer_size
    
    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > self.buffer_size:
            remaining = batch_size % self.buffer_size
            reps = int(np.floor(batch_size / self.buffer_size))
            indices = np.tile(np.arange(self.buffer_size), reps)
            if remaining > 0:
                indices = np.concatenate((indices, np.random.choice(self.buffer_size, remaining, replace=False)))
        else:
            indices = np.random.choice(self.buffer_size, batch_size, replace=False)
        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices], self.goals[indices]
    
    def get_avg_reward(self) -> float:
        if self.buffer_size == 0:
            return -100.0
        return self.rewards[:self.buffer_size].mean()