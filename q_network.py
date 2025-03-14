from typing import Type

import torch
import torch.nn as nn
from gymnasium import spaces


class QNetworkCNN(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
    ):
        super(QNetworkCNN, self).__init__()
        self.observation_space = observation_space
        self.n_actions = int(action_space.n)

        in_channels = self.observation_space.shape[-1]
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=(8, 8),
                stride=(4, 4),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
        )

        self.feature_size = self._derive_activation_size()
        self.q_net = nn.Sequential(
            nn.Linear(in_features=self.feature_size, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.n_actions),
        )

    def _derive_activation_size(self):
        observation_shape = self.observation_space.shape  # extracting observation shape
        height, width, in_channnels = observation_shape
        random_observation_tensor = torch.zeros((1, in_channnels, height, width), dtype=torch.float32)

        feature_size = self.feature_extractor(random_observation_tensor).shape[-1]
        return feature_size

    def forward(self, observation):
        features = self.feature_extractor(observation)
        return self.q_net(features)
    

    def is_trainable(self, requires_grad=True):
        for params in self.parameters():
            params.requires_grad = requires_grad