from typing import Type
from tqdm import tqdm

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from q_network import QNetworkCNN
from replay_buffer import ReplayBuffer


def greedy_action_selection(q_net: QNetworkCNN, observation: torch.Tensor):
    with torch.no_grad():
        action = q_net(observation).argmax().item()

    return action


def epsilon_greedy_action_selection(
    q_net: QNetworkCNN,
    observation: torch.Tensor,
    action_space: spaces.Discrete,
    epsilon: float,
):
    if np.random.rand() <= epsilon:  # exploration
        action = action_space.sample()
    else:  # exploitation
        action = greedy_action_selection(q_net, observation)

    return action


def exp_epsilon_scheduler(
    min_epsilon: float, max_epsilon: float, decay_rate: float, current_step: int
):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(
        -decay_rate * current_step
    )


def linear_epsilon_schedule(
    initial_value: float,
    final_value: float,
    max_steps: int,
    current_step: int,
) -> float:
    # Compute current progress (in [0, 1], 0 being the start)
    progress = current_step / max_steps
    # Clip the progress so the schedule is constant after reaching the final value
    progress = min(progress, 1.0)

    return initial_value + progress * (final_value - initial_value)


def preprocess_observation(observation):
    return np.transpose((observation / 255.0).astype(dtype=np.float32), axes=[2, 0, 1])


CONFIG = dict(
    # device
    device="cuda:0",
    # Training parameters
    n_training_episodes=100,  # Total training episodes
    learning_rate=0.01,  # Learning rate
    # Evaluation parameters
    n_eval_episodes=100,  # Total number of test episodes
    # Environment parameters
    env_id="SpaceInvaders",  # Name of the environment
    max_steps=99,  # Max steps per episode
    gamma=0.95,  # Discounting rate
    eval_seed=[],  # The evaluation seed of the environment
    # Exploration parameters
    exploration_max_epsilon=1.0,  # Exploration probability at start
    exploration_min_epsilon=0.05,  # Minimum exploration probability
    exploration_decay_rate=0.0005,  # Exponential decay rate for exploration prob
    exploration_fraction=0.1,
    # buffer
    replay_buffer_size=10_000,
    batch_size=64,
)


class DQN:
    def __init__(self, env, config=CONFIG, _init_setup_model=True):
        self.env = env
        self.config = config

        if _init_setup_model:
            self.q_net = self.init_model()

    def configure_optimizer(self):
        return torch.optim.Adam(
            self.q_net.parameters(), lr=self.config["learning_rate"]
        )

    def init_model(self):
        q_net = QNetworkCNN(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        ).to(device=self.config["device"])
        q_net.is_trainable(requires_grad=True)

        return q_net

    def load_qnet(self, path):
        state_dict = torch.load(path)
        self.q_net.load_state_dict(state_dict)
        self.q_net.is_trainable(requires_grad=False)

    def update_epsilon(self, current_step, scheduler="exp"):
        if scheduler == "exp":
            return exp_epsilon_scheduler(
                min_epsilon=self.config["exploration_min_epsilon"],
                max_epsilon=self.config["exploration_max_epsilon"],
                decay_rate=self.config["exploration_decay_rate"],
                current_step=current_step,
            )

        return linear_epsilon_schedule(
            initial_value=self.config["exploration_initial_epsilon"],
            final_value=self.config["exploration_initial_epsilon"],
            max_steps=self.config["exploration_fraction"] * self.config["n_timesteps"],
            current_step=current_step,
        )

    def update_dqn(
        self,
        q_net_target: QNetworkCNN,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        batch_size=32,
    ):
        replay_data = replay_buffer.sample(batch_size=batch_size).to_torch(
            device=self.config["device"]
        )
        with torch.no_grad():
            next_q_values = q_net_target(replay_data.next_observations)
            next_q_values, _ = next_q_values.max(dim=1)

            should_bootstrap = torch.logical_not(replay_data.terminateds)
            td_target = (
                replay_data.rewards
                + self.config["gamma"] * next_q_values * should_bootstrap
            )

        q_values = self.q_net(replay_data.observations)  # b x action_space.n
        current_q_values = torch.gather(
            q_values, dim=1, index=replay_data.actions
        ).squeeze(
            dim=1
        )  # (b, 1) ->  (b, )

        assert current_q_values.shape == (
            batch_size,
        ), f"{current_q_values.shape} != ({batch_size}, )"
        assert (
            current_q_values.shape == td_target.shape
        ), f"{current_q_values.shape} != {td_target.shape}"

        loss = ((current_q_values - td_target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _step_rollout(
        self, observation, usage, epsilon=None, replay_buffer: ReplayBuffer = None
    ):

        observation = preprocess_observation(observation)
        observation = (
            torch.as_tensor(observation)
            .unsqueeze(dim=0)
            .to(device=self.config["device"])
        )
        if usage == "exploit":
            action = greedy_action_selection(self.q_net, observation)
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )

        else:
            assert (
                replay_buffer is not None
            ), "ReplayBuffer cannot be None while exploring."

            action = epsilon_greedy_action_selection(
                q_net=self.q_net,
                observation=observation,
                action_space=self.env.action_space,
                epsilon=epsilon,
            )
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            replay_buffer.store_transistion(
                observation=observation.squeeze(dim=0).detach().cpu().numpy(),
                next_observation=preprocess_observation(next_observation),
                action=action,
                reward=float(reward),
                terminated=terminated,
            )

        return action, next_observation, reward, terminated, truncated, info

    def rollout(
        self,
        max_steps,
        seed: int = 0,
        usage: str = "exploit",
        epsilon: int = None,
        replay_buffer: ReplayBuffer = None,
    ):
        terminated, truncated = False, False
        observation, info = self.env.reset(seed=seed)
        logs = dict(
            total_steps=0,
            action_step=[],
            status="",
            total_rewards_ep=0,
        )

        for step in range(max_steps):
            action, next_observation, reward, terminated, truncated, info = (
                self._step_rollout(
                    observation=observation,
                    usage=usage,
                    epsilon=epsilon,
                    replay_buffer=replay_buffer,
                )
            )

            logs["action_step"].append(action)
            logs["total_rewards_ep"] += reward

            if terminated or truncated:
                logs["total_steps"] = step
                logs["status"] = "terminated" if terminated else "truncated"
                break

            observation = next_observation

        if step + 1 == max_steps:
            logs["status"] = "max_steps"
            logs["total_steps"] = step

        return logs

    def _step_train(self, q_net_target, epsilon, optimizer, replay_buffer, episode):
        episode_logs = self.rollout(
            self.config["max_steps"],
            epsilon=epsilon,
            usage="explore",
            replay_buffer=replay_buffer,
        )

        q_net_target.load_state_dict(self.q_net.state_dict())

        self.update_dqn(
            q_net_target=q_net_target,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
        )

        return episode_logs

    def train(self, n_training_episodes=1000):
        logs = []

        if n_training_episodes is None or n_training_episodes == 0:
            n_training_episodes = self.config["n_training_episodes"]

        replay_buffer = ReplayBuffer(
            buffer_size=self.config["replay_buffer_size"],
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )
        optimizer = self.configure_optimizer()
        q_net_target = self.init_model()
        q_net_target.load_state_dict(self.q_net.state_dict())
        q_net_target.is_trainable(requires_grad=False)

        tqdm_iterator = tqdm(range(n_training_episodes))
        for episode in tqdm_iterator:
            epsilon = self.update_epsilon(episode, scheduler="exp")

            episode_logs = self._step_train(
                q_net_target, epsilon, optimizer, replay_buffer, episode
            )

            logs.append(episode_logs)

        return logs
