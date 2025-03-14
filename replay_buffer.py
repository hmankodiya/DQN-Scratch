from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces


@dataclass
class TorchReplayBufferSamples:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminateds: torch.Tensor


@dataclass
class ReplayBufferSamples:
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminateds: np.ndarray

    def to_torch(self, device: str = "cuda:0") -> TorchReplayBufferSamples:
        return TorchReplayBufferSamples(
            torch.as_tensor(self.observations, device=device),
            torch.as_tensor(self.next_observations, device=device),
            torch.as_tensor(self.actions, device=device),
            torch.as_tensor(self.rewards, device=device),
            torch.as_tensor(self.terminateds, device=device),
        )


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
    ):
        self.current_index = 0
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.is_full = False

        _h, _w, _c = observation_space.shape
        self.observations = np.zeros(
            (self.buffer_size, _c, _h, _w), dtype=np.float32
        )
        self.next_observations = np.zeros(
            (self.buffer_size, _c, _h, _w), dtype=np.float32
        )

        action_dim = 1
        self.actions = np.zeros(
            (self.buffer_size, action_dim), dtype=action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.terminateds = np.zeros((self.buffer_size,), dtype=bool)

    def store_transistion(
        self,
        observation: np.ndarray,
        next_observation: np.ndarray,
        action,
        reward: float,
        terminated: bool,
    ):
        self.observations[self.current_index] = observation
        self.next_observations[self.current_index] = next_observation
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.terminateds[self.current_index] = terminated

        self.current_index += 1

        if self.current_index == self.buffer_size:
            self.is_full = True
            self.current_index = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.is_full else self.current_index
        batch_indices = np.random.randint(low=0, high=upper_bound, size=batch_size)
        
        return ReplayBufferSamples(
            observations = self.observations[batch_indices],
            next_observations = self.next_observations[batch_indices],
            actions = self.actions[batch_indices],
            rewards = self.rewards[batch_indices],
            terminateds = self.terminateds[batch_indices],
        )
