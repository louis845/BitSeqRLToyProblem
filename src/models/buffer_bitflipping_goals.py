import gc

import numpy as np
import torch

from .buffer_base import BufferBase
from ..utils import bit_sequence_to_int, int_to_bit_sequence

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
    
    def append_multiple_at_once(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray, goals: np.ndarray):
        if self.no_repeat:
            # convert states and next_states to int format
            states = bit_sequence_to_int(states, self.n_exp)
            next_states = bit_sequence_to_int(next_states, self.n_exp)

            # sort first by states, then by actions, then by next_states, then by dones, then by goals
            sort_indices = np.lexsort((goals, dones, next_states, actions, states))
            states = states[sort_indices]
            actions = actions[sort_indices]
            rewards = rewards[sort_indices]
            next_states = next_states[sort_indices]
            dones = dones[sort_indices]
            goals = goals[sort_indices]

            # ok, now remove duplicates using np.diff
            all_same = (np.diff(states) == 0) & (np.diff(actions) == 0) & (np.diff(rewards) == 0) & (np.diff(next_states) == 0) & (np.diff(dones) == 0) & (np.diff(goals) == 0)
            unique_indices = np.concatenate(([True], ~all_same))
            states = states[unique_indices]
            actions = actions[unique_indices]
            rewards = rewards[unique_indices]
            next_states = next_states[unique_indices]
            dones = dones[unique_indices]
            goals = goals[unique_indices]

            # convert back to bit format
            states = int_to_bit_sequence(states, self.n_exp)
            next_states = int_to_bit_sequence(next_states, self.n_exp)

            # now add
            if self.buffer_size == 0:
                # find lowest multiple of 2 that is larger or equal to the number of unique states
                new_length = 2 ** np.ceil(np.log2(states.shape[0])).astype(np.int32)
                self.states = torch.zeros((new_length, self.n), dtype=torch.int32, device=self.device)
                self.actions = torch.zeros(new_length, dtype=torch.int32, device=self.device)
                self.rewards = torch.zeros(new_length, dtype=torch.float32, device=self.device)
                self.next_states = torch.zeros((new_length, self.n), dtype=torch.int32, device=self.device)
                self.dones = torch.zeros(new_length, dtype=torch.bool, device=self.device)
                self.goals = torch.zeros(new_length, dtype=torch.int32, device=self.device)
                self.buffer_size = states.shape[0]
                # set the values
                self.states[:self.buffer_size, :].copy_(torch.tensor(states, dtype=torch.int32))
                self.actions[:self.buffer_size].copy_(torch.tensor(actions, dtype=torch.int32))
                self.rewards[:self.buffer_size].copy_(torch.tensor(rewards, dtype=torch.float32))
                self.next_states[:self.buffer_size, :].copy_(torch.tensor(next_states, dtype=torch.int32))
                self.dones[:self.buffer_size].copy_(torch.tensor(dones, dtype=torch.bool))
                self.goals[:self.buffer_size].copy_(torch.tensor(goals, dtype=torch.int32))
            else:
                # convert to torch tensor
                states = torch.tensor(states, dtype=torch.int32, device=self.device)
                actions = torch.tensor(actions, dtype=torch.int32, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                next_states = torch.tensor(next_states, dtype=torch.int32, device=self.device)
                dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
                goals = torch.tensor(goals, dtype=torch.int32, device=self.device)
                # restrict to the states not existing in the buffer
                states_repeated = (states.unsqueeze(1) == self.states[:self.buffer_size, :].unsqueeze(0)).all(dim=2).any(dim=1)
                actions_repeated = (actions.unsqueeze(1) == self.actions[:self.buffer_size].unsqueeze(0)).any(dim=1)
                next_states_repeated = (next_states.unsqueeze(1) == self.next_states[:self.buffer_size, :].unsqueeze(0)).all(dim=2).any(dim=1)
                dones_repeated = (dones.unsqueeze(1) == self.dones[:self.buffer_size].unsqueeze(0)).any(dim=1)
                goals_repeated = (goals.unsqueeze(1) == self.goals[:self.buffer_size].unsqueeze(0)).any(dim=1)
                repeated = states_repeated & actions_repeated & next_states_repeated & dones_repeated & goals_repeated
                if repeated.all():
                    return
                states = states[~repeated]
                actions = actions[~repeated]
                rewards = rewards[~repeated]
                next_states = next_states[~repeated]
                dones = dones[~repeated]
                goals = goals[~repeated]

                # now add
                new_length = self.states.shape[0]
                while new_length < self.buffer_size + states.shape[0]:
                    new_length *= 2
                if new_length > self.states.shape[0]:
                    # if requires growing, grow
                    new_states = torch.zeros((new_length, self.n), dtype=torch.int32, device=self.device)
                    new_actions = torch.zeros(new_length, dtype=torch.int32, device=self.device)
                    new_rewards = torch.zeros(new_length, dtype=torch.float32, device=self.device)
                    new_next_states = torch.zeros((new_length, self.n), dtype=torch.int32, device=self.device)
                    new_dones = torch.zeros(new_length, dtype=torch.bool, device=self.device)
                    new_goals = torch.zeros(new_length, dtype=torch.int32, device=self.device)
                    new_states[:self.buffer_size, :].copy_(self.states[:self.buffer_size, :])
                    new_actions[:self.buffer_size].copy_(self.actions[:self.buffer_size])
                    new_rewards[:self.buffer_size].copy_(self.rewards[:self.buffer_size])
                    new_next_states[:self.buffer_size, :].copy_(self.next_states[:self.buffer_size, :])
                    new_dones[:self.buffer_size].copy_(self.dones[:self.buffer_size])
                    new_goals[:self.buffer_size].copy_(self.goals[:self.buffer_size])
                    del self.states, self.actions, self.rewards, self.next_states, self.dones, self.goals
                    self.states = new_states
                    self.actions = new_actions
                    self.rewards = new_rewards
                    self.next_states = new_next_states
                    self.dones = new_dones
                    self.goals = new_goals
                    gc.collect()
                    torch.cuda.empty_cache()
                # set the values
                self.states[self.buffer_size:self.buffer_size + states.shape[0], :].copy_(states)
                self.actions[self.buffer_size:self.buffer_size + states.shape[0]].copy_(actions)
                self.rewards[self.buffer_size:self.buffer_size + states.shape[0]].copy_(rewards)
                self.next_states[self.buffer_size:self.buffer_size + states.shape[0], :].copy_(next_states)
                self.dones[self.buffer_size:self.buffer_size + states.shape[0]].copy_(dones)
                self.goals[self.buffer_size:self.buffer_size + states.shape[0]].copy_(goals)
        else:
            # directly add, we allow repeats anyway
            if self.buffer_size == 0:
                # find lowest multiple of 2 that is larger or equal to the number of unique states
                new_length = 2 ** np.ceil(np.log2(states.shape[0])).astype(np.int32)
                self.states = np.zeros((new_length, self.n), dtype=np.int32)
                self.actions = np.zeros(new_length, dtype=np.int32)
                self.rewards = np.zeros(new_length, dtype=np.float32)
                self.next_states = np.zeros((new_length, self.n), dtype=np.int32)
                self.dones = np.zeros(new_length, dtype=np.bool)
                self.goals = np.zeros(new_length, dtype=np.int32)
                self.buffer_size = states.shape[0]
                # set the values
                self.states[:self.buffer_size, :] = states
                self.actions[:self.buffer_size] = actions
                self.rewards[:self.buffer_size] = rewards
                self.next_states[:self.buffer_size, :] = next_states
                self.dones[:self.buffer_size] = dones
                self.goals[:self.buffer_size] = goals
            else:
                new_length = self.states.shape[0]
                while new_length < self.buffer_size + states.shape[0]:
                    new_length *= 2
                if new_length > self.states.shape[0]: # if requires growing, grow
                    new_states = np.zeros((new_length, self.n), dtype=np.int32)
                    new_actions = np.zeros(new_length, dtype=np.int32)
                    new_rewards = np.zeros(new_length, dtype=np.float32)
                    new_next_states = np.zeros((new_length, self.n), dtype=np.int32)
                    new_dones = np.zeros(new_length, dtype=np.bool)
                    new_goals = np.zeros(new_length, dtype=np.int32)
                    new_states[:self.buffer_size, :] = self.states[:self.buffer_size, :]
                    new_actions[:self.buffer_size] = self.actions[:self.buffer_size]
                    new_rewards[:self.buffer_size] = self.rewards[:self.buffer_size]
                    new_next_states[:self.buffer_size, :] = self.next_states[:self.buffer_size, :]
                    new_dones[:self.buffer_size] = self.dones[:self.buffer_size]
                    new_goals[:self.buffer_size] = self.goals[:self.buffer_size]
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
                self.goals[self.buffer_size:self.buffer_size + states.shape[0]] = goals
    
    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > self.buffer_size:
            remaining = batch_size % self.buffer_size
            reps = int(np.floor(batch_size / self.buffer_size))
            indices = np.tile(np.arange(self.buffer_size), reps)
            if remaining > 0:
                indices = np.concatenate((indices, np.random.choice(self.buffer_size, remaining, replace=False)))
        else:
            indices = np.random.choice(self.buffer_size, batch_size, replace=False)
        if self.no_repeat:
            return self.states[indices].cpu().numpy(), self.actions[indices].cpu().numpy(), self.rewards[indices].cpu().numpy(),\
                self.next_states[indices].cpu().numpy(), self.dones[indices].cpu().numpy(), self.goals[indices].cpu().numpy()
        else:
            return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.dones[indices], self.goals[indices]
    