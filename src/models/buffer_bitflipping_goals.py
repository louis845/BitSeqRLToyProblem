import gc

import numpy as np
import torch

from .buffer_base import BufferBase

class BufferBitflippingGoals(BufferBase):
    """Simple replay buffer, with amoritized O(1) append access. This replay buffer corresponds to the bitflipping DQN model."""

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

    def __init__(self, n: int, no_repeat: bool = False, device: torch.device = torch.device("cpu")):
        self.n = n
        self.no_repeat = no_repeat
        self.device = device

        if no_repeat:
            self.states = torch.zeros((0, n), dtype=torch.int32, device=device) # state space is stored in 0, 1 format
            self.actions = torch.zeros((0,), dtype=torch.int32, device=device) # action space is stored in integer format
            self.rewards = torch.zeros((0,), dtype=torch.float32, device=device)
            self.next_states = torch.zeros((0, n), dtype=torch.int32, device=device)
            self.dones = torch.zeros((0,), dtype=torch.bool, device=device)
            self.goals = torch.zeros((0,), dtype=torch.int32, device=device)
        else:
            self.states = np.zeros((0, n), dtype=np.int32) # state space is stored in 0, 1 format
            self.actions = np.zeros((0,), dtype=np.int32) # action space is stored in integer format
            self.rewards = np.zeros((0,), dtype=np.float32)
            self.next_states = np.zeros((0, n), dtype=np.int32)
            self.dones = np.zeros((0,), dtype=bool)
            self.goals = np.zeros((0,), dtype=np.int32)
        self.buffer_size = 0
    
    def __len__(self) -> int:
        return self.buffer_size

    def append(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, goal: int = None):
        assert state.shape == (self.n,), "state must have the same length as the buffer. state: {}, buffer: {}".format(state.shape, self.n)
        assert next_state.shape == (self.n,), "next_state must have the same length as the buffer, next_state: {}, buffer: {}".format(next_state.shape, self.n)

        if self.buffer_size == 0:
            if self.no_repeat:
                self.states = torch.tensor(state.reshape(1, -1).copy(), dtype=torch.int32, device=self.device)
                self.actions = torch.tensor([action], dtype=torch.int32, device=self.device)
                self.rewards = torch.tensor([reward], dtype=torch.float32, device=self.device)
                self.next_states = torch.tensor(next_state.reshape(1, -1), dtype=torch.int32, device=self.device)
                self.dones = torch.tensor([done], dtype=torch.bool, device=self.device)
                self.goals = torch.tensor([goal], dtype=torch.int32, device=self.device)
            else:
                self.states = state.reshape(1, -1).copy()
                self.actions = np.array([action], dtype=np.int32)
                self.rewards = np.array([reward], dtype=np.float32)
                self.next_states = next_state.reshape(1, -1).copy()
                self.dones = np.array([done], dtype=bool)
                self.goals = np.array([goal], dtype=np.int32)
        else:
            if self.no_repeat:
                # check whether the state is already in the buffer
                if (torch.all(self.states == torch.tensor(state, dtype=torch.int32, device=self.device).unsqueeze(0), dim=1)
                    & (self.actions == action)
                    & torch.all(self.next_states == torch.tensor(next_state, dtype=torch.int32, device=self.device).unsqueeze(0), dim=1)
                    & (self.dones == done)
                    & (self.goals == goal)).any():
                    # if the state is already in the buffer, return
                    return
            if self.buffer_size == self.states.shape[0]: # expand the buffer
                if self.no_repeat:
                    new_states = torch.zeros((2 * self.buffer_size, self.n), dtype=torch.int32, device=self.device)
                    new_actions = torch.zeros(2 * self.buffer_size, dtype=torch.int32, device=self.device)
                    new_rewards = torch.zeros(2 * self.buffer_size, dtype=torch.float32, device=self.device)
                    new_next_states = torch.zeros((2 * self.buffer_size, self.n), dtype=torch.int32, device=self.device)
                    new_dones = torch.zeros(2 * self.buffer_size, dtype=torch.bool, device=self.device)
                    new_goals = torch.zeros(2 * self.buffer_size, dtype=torch.int32, device=self.device)
                    new_states[:self.buffer_size, :].copy_(self.states)
                    new_actions[:self.buffer_size].copy_(self.actions)
                    new_rewards[:self.buffer_size].copy_(self.rewards)
                    new_next_states[:self.buffer_size, :].copy_(self.next_states)
                    new_dones[:self.buffer_size].copy_(self.dones)
                    new_goals[:self.buffer_size].copy_(self.goals)
                    del self.states, self.actions, self.rewards, self.next_states, self.dones, self.goals
                    self.states = new_states
                    self.actions = new_actions
                    self.rewards = new_rewards
                    self.next_states = new_next_states
                    self.dones = new_dones
                    self.goals = new_goals

                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    new_states = np.zeros((2 * self.buffer_size, self.n), dtype=np.int32)
                    new_actions = np.zeros(2 * self.buffer_size, dtype=np.int32)
                    new_rewards = np.zeros(2 * self.buffer_size, dtype=np.float32)
                    new_next_states = np.zeros((2 * self.buffer_size, self.n), dtype=np.int32)
                    new_dones = np.zeros(2 * self.buffer_size, dtype=bool)
                    new_goals = np.zeros(2 * self.buffer_size, dtype=np.int32)
                    new_states[:self.buffer_size, :] = self.states
                    new_actions[:self.buffer_size] = self.actions
                    new_rewards[:self.buffer_size] = self.rewards
                    new_next_states[:self.buffer_size, :] = self.next_states
                    new_dones[:self.buffer_size] = self.dones
                    new_goals[:self.buffer_size] = self.goals
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
                self.goals[self.buffer_size].copy_(torch.tensor(goal, dtype=torch.int32))
            else:
                self.states[self.buffer_size, :] = state
                self.actions[self.buffer_size] = action
                self.rewards[self.buffer_size] = reward
                self.next_states[self.buffer_size, :] = next_state
                self.dones[self.buffer_size] = done
                self.goals[self.buffer_size] = goal
            
        self.buffer_size += 1
    
    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > self.buffer_size:
            indices = np.random.choice(self.buffer_size, batch_size, replace=True)
        else:
            indices = np.random.choice(self.buffer_size, batch_size, replace=False)
        if self.no_repeat:
            return self.states[indices].cpu().numpy(), self.actions[indices].cpu().numpy(), self.rewards[indices].cpu().numpy(),\
                self.next_states[indices].cpu().numpy(), self.dones[indices].cpu().numpy(), self.goals[indices].cpu().numpy()
        else:
            return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices], self.goals[indices]
    