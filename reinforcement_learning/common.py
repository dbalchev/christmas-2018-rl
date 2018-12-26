from abc import ABCMeta, abstractclassmethod

from gym.spaces import Discrete
import numpy as np
import torch


def one_hot(labels, depth):
    """
    >>> one_hot(torch.tensor([0, 1, 0]), 2)
    tensor([[1., 0.],
            [0., 1.],
            [1., 0.]])
    """
    out = torch.zeros(*labels.size(), depth)
    out.scatter_(1, labels[:, None], 1)
    return out


def _discount_rewards(rewards, discount):
    """
    >>> _discount_rewards([0., 0., 1., 0.], 0.5)
    [0.25, 0.5, 1.0, 0.0]
    """
    current_reward = rewards[-1]
    discounted_rewards = [current_reward]
    for reward in rewards[-2::-1]:
        current_reward = reward + discount * current_reward
        discounted_rewards.append(current_reward)
    discounted_rewards.reverse()
    return discounted_rewards

class SimpleTrainer(metaclass=ABCMeta):

    def __init__(
            self, env, *, reward_discount, should_render=False, 
            exploration_prob=0.0):
        self.env = env
        self.should_render = should_render
        self.reward_discount = reward_discount
        self.rng = np.random.RandomState()
        self.exploration_prob = exploration_prob
        
    def train_one_episode(self):
        observations, actions, rewards = self._run_episode()
        total_reward = np.sum(rewards)
        rewards = _discount_rewards(rewards, self.reward_discount)
        loss = self._train_on_episode(observations, actions, rewards)
        return total_reward, loss

    def _maybe_render(self):
        if self.should_render:
            self.env.render()
    
    @abstractclassmethod
    def _choose_action(self, observation):
        pass
    
    @abstractclassmethod
    def _train_on_episode(self, observations, actions, rewards):
        pass

    def log(self, format, *args, **kwargs):
        if not self.should_render:
            return
        print(format.format(*args, **kwargs))
    
    def _run_episode(self):
        rewards = []
        done = False
        observation = self.env.reset()
        observations = [observation]
        actions = []
        self._maybe_render()
        while not done:
            chosen_action = self._choose_action(observation)
            actions.append(chosen_action)
            observation, reward, done, _ = self.env.step(chosen_action)
            if len(observations) > 750:
                done = True
                reward += -20
            self._maybe_render()
            observations.append(observation)
            rewards.append(reward)
        return observations, actions, rewards
    
    def _sample_model_action(self, model_result):
        action_space = self.env.action_space
        if isinstance(action_space, Discrete):
            assert model_result.shape == (action_space.n, )
            return self.rng.choice(action_space.n, p=model_result)
        raise ValueError('Unsupported action space {}'.format(action_space))

    def _sample_action(self, model_result):
        if self.rng.uniform() < self.exploration_prob:
            return self.env.action_space.sample()
        return self._sample_model_action(model_result)
