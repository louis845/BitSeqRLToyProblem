import gc

import numpy as np
import torch

from .buffer_base import BufferBase

class BufferNaive(BufferBase):
    """Simple replay buffer, with amoritized O(1) append access. This replay buffer corresponds to the naive DQN model."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    buffer_size: int

    def __init__(self, n: int):
        self.n = n
        self.states = np.zeros((0, n), dtype=np.int32) # state space is stored in 0, 1 format
        self.actions = np.zeros((0,), dtype=np.int32) # action space is stored in integer format
        self.rewards = np.zeros((0,), dtype=np.float32)
        self.next_states = np.zeros((0, n), dtype=np.int32)
        self.dones = np.zeros((0,), dtype=bool)
        self.buffer_size = 0
    
    def __len__(self) -> int:
        return self.buffer_size

    def append(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, goal: int = None):
        assert state.shape == (self.n,), "state must have the same length as the buffer. state: {}, buffer: {}".format(state.shape, self.n)
        assert next_state.shape == (self.n,), "next_state must have the same length as the buffer, next_state: {}, buffer: {}".format(next_state.shape, self.n)

        if self.buffer_size == 0:
            self.states = state.reshape(1, -1).copy()
            self.actions = np.array([action], dtype=np.int32)
            self.rewards = np.array([reward], dtype=np.float32)
            self.next_states = next_state.reshape(1, -1).copy()
            self.dones = np.array([done], dtype=bool)
        else:
            if self.buffer_size == self.states.shape[0]:
                new_states = np.zeros((2 * self.buffer_size, self.n), dtype=np.int32)
                new_actions = np.zeros(2 * self.buffer_size, dtype=np.int32)
                new_rewards = np.zeros(2 * self.buffer_size, dtype=np.float32)
                new_next_states = np.zeros((2 * self.buffer_size, self.n), dtype=np.int32)
                new_dones = np.zeros(2 * self.buffer_size, dtype=bool)
                new_states[:self.buffer_size, :] = self.states
                new_actions[:self.buffer_size] = self.actions
                new_rewards[:self.buffer_size] = self.rewards
                new_next_states[:self.buffer_size, :] = self.next_states
                new_dones[:self.buffer_size] = self.dones
                del self.states, self.actions, self.rewards, self.next_states, self.dones
                self.states = new_states
                self.actions = new_actions
                self.rewards = new_rewards
                self.next_states = new_next_states
                self.dones = new_dones
                gc.collect()

            self.states[self.buffer_size, :] = state
            self.actions[self.buffer_size] = action
            self.rewards[self.buffer_size] = reward
            self.next_states[self.buffer_size, :] = next_state
            self.dones[self.buffer_size] = done
            
        self.buffer_size += 1
    
    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, None]:
        if batch_size > self.buffer_size:
            indices = np.random.choice(self.buffer_size, batch_size, replace=True)
        else:
            indices = np.random.choice(self.buffer_size, batch_size, replace=False)
        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices], None
    