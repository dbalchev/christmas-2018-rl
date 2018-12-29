from gym.spaces import Discrete, Box
import numpy as np
import torch

from reinforcement_learning.common import SimpleTrainer, Scaler


class MLPReinforceModel(torch.nn.Module):
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
        self.model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], self.hidden_units,),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.hidden_units, self.hidden_units,),
            torch.nn.LeakyReLU(),

            *self._choose_last_layers(env),
        )
    
    def forward(self, inputs):
        result = self.model(inputs)
        return result
    
    def _choose_last_layers(self, env):
        if isinstance(env.action_space, Discrete):
            last_layer = torch.nn.Linear(
                self.hidden_units, env.action_space.n, bias=False)
            last_layer.weight.data.zero_()
            return [
                last_layer,
                torch.nn.Softmax(dim=-1),
            ]
        if isinstance(env.action_space, Box):
            if len(env.action_space.shape) != 1:
                raise ValueError(
                    'Unsupported action space rank != 1 {}'.format(
                        env.action_space.shape))
            last_layer = torch.nn.Linear(
                self.hidden_units, env.action_space.shape[0], bias=False)
            last_layer.weight.data.zero_()
            return [
                last_layer,
                torch.nn.Sigmoid(),
                Scaler(env.action_space.low, env.action_space.high),
            ]
        raise ValueError(
                'Unsupported action space {}'.format(env.action_space))


class PolicyOptimizationTrainer(SimpleTrainer):

    def __init__(self, env, agent, **kwargs):
        super().__init__(env, **kwargs)
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-3)

    def _choose_action(self, observation):
        with torch.no_grad():
            action_probs = self.agent(
                torch.tensor([observation], dtype=torch.float32)).numpy()[0]
        return self._sample_action(action_probs)

    def _chosen_action_log_probabilities(self, action_logits, actions):
        action_space = self.env.action_space
        if isinstance(action_space, Discrete):
            assert action_logits.size()[-1] == action_space.n
            return action_logits[np.arange(len(actions)), actions].log()
        if isinstance(action_space, Box):
            assert action_space.shape == action_space.shape
            actions = torch.tensor(actions, dtype=torch.float32)
            per_action_log_prob = self.box_action_distribution.log_prob(
                action_logits - actions)
            return per_action_log_prob.sum(dim=-1)
        raise ValueError('Unsupported action space {}'.format(action_space)) 

    def _postprocess_loss(self, normal_loss):
        return normal_loss


class ReinforceTrainer(PolicyOptimizationTrainer):
    def _train_on_episode(self, observations, actions, rewards):
        # print(rewards)
        action_logits = self.agent(
            torch.tensor(observations[:-1], dtype=torch.float32))
        chosen_action_log_probabilities = self._chosen_action_log_probabilities(
            action_logits, actions)
        loss = -chosen_action_log_probabilities * torch.tensor(
            rewards, dtype=torch.float32)
        # action_logits_regularization = torch.nn.functional.mse_loss(
        #     action_logits, torch.tensor(0.0))
        loss = loss.mean() # + 1e-4 * action_logits_regularization
        loss = self._postprocess_loss(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_value = loss.detach().item()
        # self.log('action_logits {}', action_logits)
        return loss_value # chosen_action_log_probabilities.detach().numpy()


class BasicPPO(PolicyOptimizationTrainer):
    def __init__(self, ppo_clip_ratio, **kwargs):
        super().__init__(**kwargs)
        self.ppo_clip_ratio = ppo_clip_ratio

    def _train_on_episode(self, observations, actions, rewards):
        action_logits = self.agent(
            torch.tensor(observations[:-1], dtype=torch.float32))
        chosen_action_log_probabilities = self._chosen_action_log_probabilities(
            action_logits, actions)
        relative_prob = (
            chosen_action_log_probabilities - 
            chosen_action_log_probabilities.detach()).exp()
        clipped_relative_prob = torch.clamp(
            relative_prob, 
            1 - self.ppo_clip_ratio,
            1 + self.ppo_clip_ratio)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        objective = torch.min(
            relative_prob * rewards_t,
            clipped_relative_prob * rewards_t).mean()
        self.optimizer.zero_grad()
        (-objective).backward()
        return objective.detach().item()
        