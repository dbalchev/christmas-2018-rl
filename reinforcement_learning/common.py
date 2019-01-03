from abc import ABCMeta, abstractclassmethod
from collections import UserList
from warnings import warn

from gym.spaces import Box, Discrete
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


def discount_rewards(rewards, discount):
    """
    >>> discount_rewards([0., 0., 1., 0.], 0.5)
    [0.25, 0.5, 1.0, 0.0]
    """
    current_reward = rewards[-1]
    discounted_rewards = [current_reward]
    for reward in rewards[-2::-1]:
        current_reward = reward + discount * current_reward
        discounted_rewards.append(current_reward)
    discounted_rewards.reverse()
    return discounted_rewards


def zip_dictionaries(dicts):
    keys = dicts[0].keys()
    return {
        key: [d[key] for d in dicts]
        for key in keys
    }


class SimpleTrainer(metaclass=ABCMeta):

    def __init__(
            self, env, *, reward_discount, should_render=False, 
            exploration_prob=0.0, max_episode_length):
        self.env = env
        self.should_render = should_render
        self.reward_discount = reward_discount
        self.rng = np.random.RandomState()
        self.exploration_prob = exploration_prob
        self.max_episode_length = max_episode_length
    
    @property
    def box_action_std(self):
        return 0.25 * (
            self.env.action_space.high - self.env.action_space.low)
    
    @property
    def box_action_distribution(self):
        box_action_std_t = torch.tensor(
            self.box_action_std, dtype=torch.float32)
        return torch.distributions.Normal(
            torch.zeros_like(box_action_std_t),
            box_action_std_t)

    def train_one_episode(self):
        observations, actions, rewards = self._run_episode()
        total_reward = np.sum(rewards)
        discounted_rewards = discount_rewards(
            rewards, self.reward_discount)
        loss = self._train_on_episode(
            observations, actions, rewards, discounted_rewards)
        return total_reward, loss

    def _maybe_render(self):
        if self.should_render:
            self.env.render()

    def _extra_buffer_kwargs(
            self, observations, actions, rewards, discounted_rewards):
        return {}
    
    @abstractclassmethod
    def _choose_action(self, observation):
        pass
    
    def _train_on_episode(
            self, observations, actions, rewards, discounted_rewards):
        warn('Training without replay buffer')
        buffer = ReplayBuffer()
        extra_kwags = self._extra_buffer_kwargs(
            observations, actions, rewards, discounted_rewards)
        buffer.append_train_one_episode_result(
            observations, actions, rewards, discounted_rewards, 
            **extra_kwags)
        dict_of_lists = zip_dictionaries(buffer)
        return self._train_on_replay_buffer(dict_of_lists)
    
    def _train_on_replay_buffer(self, replay_buffer_sample):
        raise NotImplemented(
            'Implement either _train_on_episode or _train_on_replay_buffer')

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
            if len(observations) > self.max_episode_length:
                done = True
            self._maybe_render()
            observations.append(observation)
            rewards.append(reward)
        return observations, actions, rewards
    
    def _sample_model_action(self, model_result):
        action_space = self.env.action_space
        if isinstance(action_space, Discrete):
            assert model_result.shape == (action_space.n, )
            return self.rng.choice(action_space.n, p=model_result)
        if isinstance(action_space, Box):
            assert model_result.shape == action_space.shape
            model_result += self.rng.normal(
                0, self.box_action_std, size=model_result.shape)
            model_result = np.clip(
                model_result, action_space.low, action_space.high)
            return model_result
            
        raise ValueError('Unsupported action space {}'.format(action_space))

    def _sample_action(self, model_result):
        if self.rng.uniform() < self.exploration_prob:
            return self.env.action_space.sample()
        return self._sample_model_action(model_result)


class Scaler(torch.nn.Module):

    def __init__(self, low, high):
        super().__init__()
        self.low = torch.tensor(low, dtype=torch.float32)
        self.delta = torch.tensor(
            high - low, dtype=torch.float32)
    
    def forward(self, inputs):
        return self.low + self.delta * inputs


class ReplayBuffer(UserList):
    def __init__(self, *args, rng=None, max_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size
        self.rng = rng
        if max_size and not rng:
            raise ValueError('If there is a max_size pass rng')
    
    def append_train_one_episode_result(
            self, observations, actions, rewards, discounted_rewards, **kwargs):
        self._clip_if_more_elements(len(observations) - 1)
        for i in range(len(observations) - 1):
            self.append({
                'current_state': observations[i],
                'future_state': observations[i + 1],
                'action': actions[i],
                'reward': rewards[i],
                'discounted_reward': rewards[i],
                'done': i + 2 == len(observations),
                **{
                    key: value[i]
                    for key, value in kwargs.items()
                },
            })
    
    def _clip_if_more_elements(self, num_new_entries):
        if not self.max_size:
            return
        if len(self) + num_new_entries <= self.max_size:
            return
        if num_new_entries >= self.max_size:
            print('Too small replay buffer', num_new_entries)
            self.data.clear()
            return
        self.data = (
            self.rng.choice(
                self.data, self.max_size - num_new_entries, 
                replace=False).
            tolist())


def copy_model(dest, src):
    dest.load_state_dict(src.state_dict())


def polyak_update(dest, src, rate):
    dest_dict = dest.state_dict()
    src_dict = src.state_dict()
    assert dest_dict.keys() == src_dict.keys()
    new_dest_dict = {
        key: rate * dest_dict[key] + (1 - rate) * src_dict[key]
        for key in dest_dict
    }
    dest.load_state_dict(new_dest_dict)