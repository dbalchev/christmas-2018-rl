import numpy as np
import torch
from abc import ABCMeta, abstractclassmethod


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

    def __init__(self, env, *, reward_discount, should_render=False):
        self.env = env
        self.should_render = should_render
        self.reward_discount = reward_discount
        self.rng = np.random.RandomState()

        
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
            if len(observations) > 250:
                done = True
                reward += -20
            self._maybe_render()
            if not done:
                observations.append(observation)
            rewards.append(reward)
        return observations, actions, rewards
    