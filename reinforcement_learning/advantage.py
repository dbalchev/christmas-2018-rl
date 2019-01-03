from gym.spaces import Box
import numpy as np
import torch

from reinforcement_learning.common import discount_rewards
from reinforcement_learning.reinforce import (
    ReinforceTrainer, BasicPPO)


class MLPValueModule(torch.nn.Module):
    def __init__(self, env, hidden_units):
        if not isinstance(env.observation_space, Box):
            raise ValueError(
                'Unsupported observation space {}'.format(
                    env.observation_space))
        if len(env.observation_space.shape) != 1:
            raise ValueError(
                'Unsupported observation space rank != 1 {}'.format(
                    env.observation_space.shape))
        super().__init__()
        self.hidden_units = hidden_units
        last_layer = torch.nn.Linear(self.hidden_units, 1, bias=False)
        last_layer.weight.data.zero_()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], self.hidden_units,),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_units, self.hidden_units,),
            torch.nn.LeakyReLU(),
            last_layer,
        )
    
    def forward(self, inputs):
        return self.model(inputs)[:, 0]


class AdvantageMixin:
    def __init__(
            self, env, agent, value_module, 
            value_num_trainings_per_batch, **kwargs):
        super().__init__(env=env, agent=agent, **kwargs)
        self.value_module = value_module
        self.value_optimizer = torch.optim.Adam(
            self.value_module.parameters(), lr=1e-3)
        self.value_num_trainings_per_batch = value_num_trainings_per_batch

    def _extra_buffer_kwargs(
            self, observations, actions, rewards, discounted_rewards):
        current_observations = observations[:-1]
        future_observations = observations[1:]
        with torch.no_grad():
            current_state_values = self.value_module(
                torch.tensor(current_observations, dtype=torch.float32)).numpy()
            next_state_values = self.value_module(
                torch.tensor(
                    future_observations, dtype=torch.float32)).numpy()
        advantage = (
            rewards + 
            self.reward_discount * next_state_values -
            current_state_values
        ).tolist()
        discounted_advantage = discount_rewards(
            advantage, self.reward_discount * 0.97)
        return {
            **super()._extra_buffer_kwargs(
                observations, actions, rewards, discounted_rewards),
            'advantage': advantage,
            'discounted_advantage': discounted_advantage,
        }

    def _train_on_replay_buffer(self, replay_buffer_sample):
        observations = replay_buffer_sample['current_state']
        discounted_rewards = replay_buffer_sample['discounted_reward']
        first_l2_loss = None
        for _ in range(self.value_num_trainings_per_batch):
            expected_rewards = self.value_module(
                torch.tensor(observations, dtype=torch.float32))
            l2_loss = torch.nn.functional.mse_loss(
                expected_rewards, torch.tensor(discounted_rewards))
            if first_l2_loss is None:
                first_l2_loss = l2_loss.item()
            self.value_optimizer.zero_grad()
            l2_loss.backward()
            self.value_optimizer.step()
        advantage = replay_buffer_sample['discounted_advantage']
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-6
        replay_buffer_sample = {
            **replay_buffer_sample,
            'discounted_reward': advantage}
        print('l2_loss', first_l2_loss, l2_loss.item())
        return super()._train_on_replay_buffer(replay_buffer_sample)
