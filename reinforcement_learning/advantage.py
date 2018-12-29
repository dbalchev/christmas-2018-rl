from gym.spaces import Box
import torch

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
            self, env, agent, value_module, **kwargs):
        super().__init__(env=env, agent=agent, **kwargs)
        self.value_module = value_module
        self.value_optimizer = torch.optim.Adam(
            self.value_module.parameters(), lr=1e-3)

    def _train_on_episode(self, observations, actions, rewards):
        expected_rewards = self.value_module(
            torch.tensor(observations[:-1], dtype=torch.float32))
        advantage = torch.tensor(rewards) - expected_rewards
        detached_advantage = advantage.detach().numpy()
        l2_loss = torch.nn.functional.mse_loss(
            advantage, torch.tensor(0.0))
        self.value_optimizer.zero_grad()
        l2_loss.backward()
        self.value_optimizer.step()
        return super()._train_on_episode(
            observations, actions, detached_advantage)


class ReinforceWithAdvantageTrainer(AdvantageMixin, ReinforceTrainer):
    pass


class PPOTrainer(AdvantageMixin, BasicPPO):
    pass