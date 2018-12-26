from collections import namedtuple

from gym.spaces import Discrete, Box
import numpy as np
import torch

from reinforcement_learning.common import one_hot
from reinforcement_learning.reinforce import ReinforceTrainer

CuriousModelFamily = namedtuple('CuriousModelFamily',
    ['forward_model', 'inverse_model', 'embedder'])


def lunar_embedder(env, embedding_units):
    if not isinstance(env.observation_space, Box):
        raise ValueError(
            'Unsupported observation space {}'.format(
                env.observation_space))
    if len(env.observation_space.shape) != 1:
        raise ValueError(
            'Unsupported observation space rank != 1 {}'.format(
                env.observation_space.shape))

    return torch.nn.Sequential(
        torch.nn.Linear(
            env.observation_space.shape[0], embedding_units, bias=False)
    )


class LunarForwardModel(torch.nn.Module):

    def __init__(self, env, embedding_units):
        if not isinstance(env.action_space, Discrete):
            raise ValueError(
                'Unsupported action space {}'.format(env.action_space))

        super().__init__()
        self.num_actions = env.action_space.n
        hidden_state_size = 48
        self.model = torch.nn.Sequential(
            torch.nn.Linear(
                embedding_units + self.num_actions, hidden_state_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_state_size, embedding_units, bias=False))
    
    def forward(self, state_embedding, action):
        one_hot_action = one_hot(action, self.num_actions)
        return self.model(
            torch.cat([state_embedding, one_hot_action], dim=-1))


class LunarInverseModel(torch.nn.Module):
    
    def __init__(self, env, embedding_units):
        if not isinstance(env.action_space, Discrete):
            raise ValueError(
                'Unsupported action space {}'.format(env.action_space))
        super().__init__()
        self.num_actions = env.action_space.n

        hidden_state_size = 48
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2 * embedding_units, hidden_state_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_state_size, self.num_actions, bias=False),
            torch.nn.Softmax(dim=-1))

    def forward(self, current_state_embedding, next_state_embedding):
        return self.model(
            torch.cat([current_state_embedding, next_state_embedding], 
            dim=-1))


class CuriousLearnerMixin:

    def __init__(self, model_family, **kwargs):
        super().__init__(**kwargs)
        self.model_family = model_family
        self.optimizer = torch.optim.Adam(
            list(self.agent.parameters()) +
            list(model_family.forward_model.parameters()) + 
            list(model_family.inverse_model.parameters()) + 
            list(model_family.embedder.parameters()), 
            # [
            #     {'params': list(self.agent.parameters())},
            #     {
            #         'params': (
            #             list(model_family.forward_model.parameters()) + 
            #             list(model_family.inverse_model.parameters()) + 
            #             list(model_family.embedder.parameters())),
            #         'lr': 1e-3,
            #     },
            # ],
            lr=1e-2)
        self.curiosity_loss = None

    def _run_episode(self):
        observations, actions, rewards = super()._run_episode()
        rewards = np.array(rewards)
        reward_updates, curiosity_loss = self._rewards_updates(
            observations, actions)
        return observations, actions, rewards + reward_updates
    
    def _rewards_updates(self, observations, actions):
        reward_updates = []

        state_embeddings = [
            self.model_family.embedder(
                torch.tensor([observation], dtype=torch.float32))
            for observation in observations]
        curiosity_loss = torch.tensor(0.0)
        for i in range(len(observations) - 1):
            action_t = torch.tensor([actions[i]], dtype=torch.long)
            predicted_next_state_embeddings = self.model_family.forward_model(
                state_embeddings[i], action_t)
            predicted_action_probs = self.model_family.inverse_model(
                state_embeddings[i], state_embeddings[i + 1])
            surprise = torch.nn.functional.mse_loss(
                predicted_next_state_embeddings, state_embeddings[i + 1])
            reward_updates.append(surprise.detach().item())
            action_loss = torch.nn.functional.cross_entropy(
                predicted_action_probs, action_t)
            curiosity_loss += action_loss + surprise
        reward_updates = np.array(reward_updates)
        # self.log('curiosity_loss {}', curiosity_loss)
        # self.log('reward_updates {}', reward_updates)
        self.curiosity_loss = curiosity_loss
        return reward_updates, curiosity_loss

class ReinforceCuriousLearner(CuriousLearnerMixin, ReinforceTrainer):
    def _postprocess_loss(self, normal_loss):
        loss = normal_loss + self.curiosity_loss
        self.curiosity_loss = None
        return loss
