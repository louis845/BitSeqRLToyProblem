from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Iterable

class BufferBase(ABC):
    """Base class for buffer implementations."""

    @abstractmethod
    def append(self, state: Any, action: Any, reward: Any, next_state: Any, done: Any, goal: Any = None):
        """Append a new experience to the buffer."""
        raise NotImplementedError
    
    def append_multiple(self, states: Iterable[Any], actions: Iterable[Any], rewards: Iterable[Any], next_states: Iterable[Any], dones: Iterable[Any], goals: Iterable[Any] = None):
        """Append multiple experiences to the buffer."""
        if goals is not None:
            for state, action, reward, next_state, done, goal in zip(states, actions, rewards, next_states, dones, goals):
                self.append(state, action, reward, next_state, done, goal)
        else:
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                self.append(state, action, reward, next_state, done)

    @abstractmethod
    def sample(self, batch_size: int) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Sample a batch of experiences from the buffer."""
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of experiences stored in the buffer."""
        raise NotImplementedError