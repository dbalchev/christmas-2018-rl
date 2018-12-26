from gym.spaces import Discrete, Box
import numpy as np
import torch

from reinforcement_learning.common import SimpleTrainer

class MLPReinforceModel(torch.nn.Module):
    def __init__(self, env, hidden_units):
        if not isinstance(env.action_space, Discrete):
            raise ValueError(
                'Unsupported action space {}'.format(env.action_space))
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
        last_layer = torch.nn.Linear(
            self.hidden_units, env.action_space.n, bias=False)
        last_layer.weight.data.zero_()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], self.hidden_units,),
            torch.nn.LeakyReLU(),
            last_layer,
        )
    
    def forward(self, inputs, apply_softmax=True):
        result = self.model(inputs)
        if apply_softmax:
            result = torch.nn.functional.softmax(result, dim=-1)
        return result


class ReinforceTrainer(SimpleTrainer):

    def __init__(self, env, agent, **kwargs):
        super().__init__(env, **kwargs)
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-2)

    def _choose_action(self, observation):
        with torch.no_grad():
            action_probs = self.agent(
                torch.tensor([observation], dtype=torch.float32)).numpy()[0]
        return self._sample_action(action_probs)

    def _train_on_episode(self, observations, actions, rewards):
        # print(rewards)
        action_logits = self.agent(
            torch.tensor(observations[:-1], dtype=torch.float32))
        chosen_action_probabilities = self._chosen_action_probabilities(
            action_logits, actions)
        loss = -chosen_action_probabilities.log() * torch.tensor(rewards, dtype=torch.float32)
        # action_logits_regularization = torch.nn.functional.mse_loss(
        #     action_logits, torch.tensor(0.0))
        loss = loss.mean() # + 1e-4 * action_logits_regularization
        loss = self._postprocess_loss(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_value = loss.detach().item()
        # self.log('action_logits {}', action_logits)
        return loss_value # chosen_action_probabilities.detach().numpy()

    def _chosen_action_probabilities(self, action_logits, actions):
        action_space = self.env.action_space
        if isinstance(action_space, Discrete):
            assert action_logits.size()[-1] == action_space.n
            return action_logits[np.arange(len(actions)), actions]
        raise ValueError('Unsupported action space {}'.format(action_space)) 

    def _postprocess_loss(self, normal_loss):
        return normal_loss