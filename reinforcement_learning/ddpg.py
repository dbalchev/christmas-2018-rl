import numpy as np
import torch

from gym.spaces import Box

from reinforcement_learning.common import (
    SimpleTrainer, polyak_update, zip_dictionaries)


def _t(v):
    return torch.tensor(v, dtype=torch.float32)

class BoxQModule(torch.nn.Module):

    def __init__(self, env, hidden_state_size):
        if not isinstance(env.action_space, Box):
            raise ValueError(
                'Unsupported action space {}'.format(env.action_space))
        if len(env.action_space.shape) != 1:
            raise ValueError(
                'Unsupported observation space rank != 1 {}'.format(
                    env.observation_space.shape))
        if not isinstance(env.observation_space, Box):
            raise ValueError(
                'Unsupported observation space {}'.format(
                    env.observation_space))
        if len(env.observation_space.shape) != 1:
            raise ValueError(
                'Unsupported observation space rank != 1 {}'.format(
                    env.observation_space.shape))
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(
                env.observation_space.shape[0] + env.action_space.shape[0], 
                hidden_state_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_state_size, hidden_state_size,),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_state_size, 1, bias=False))
        
    def forward(self, observation, action):
        return self.model(
            torch.cat([observation, action], dim=-1))[..., 0]


class DDPGTrainer(SimpleTrainer):
    def __init__(
            self, policy, q_func, target_policy, target_q, 
            ddpg_reward_discount, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy
        self.q_func = q_func
        self.target_policy = target_policy
        self.target_q = target_q
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=1e-3)
        self.q_optimizer = torch.optim.Adam(
            self.q_func.parameters(), lr=1e-3)
        self.replay_buffer = []
        self.replay_buffer_max_size = 10000
        self.batch_size = 100
        self.ddpg_reward_discount = ddpg_reward_discount
    
    # @property
    # def target_policy(self):
    #     return self.policy

    # @property
    # def target_q(self):
    #     return self.q_func

    def _choose_action(self, observation):
        with torch.no_grad():
            mean = self.policy(_t([observation]))[0].numpy()
        return self._sample_action(mean)
    
    def _train_on_episode(self, observations, actions, rewards):
        for i in range(len(observations) - 1):
            self.replay_buffer.append({
                'current_state': observations[i],
                'future_state': observations[i + 1],
                'action': actions[i],
                'reward': rewards[i],
                'done': float(i + 2 == len(observations)),
            })
        self._actual_training()
        if len(self.replay_buffer) > self.replay_buffer_max_size:
            self.replay_buffer = self.rng.choice(
                self.replay_buffer, self.replay_buffer_max_size, 
                replace=False).tolist()
        return 0
    
    def _actual_training(self):
        replay_buffer_sample = self.rng.choice(
            self.replay_buffer, self.batch_size)
        replay_buffer = zip_dictionaries(replay_buffer_sample)
        current_state_t = _t(replay_buffer['current_state'])
        future_state_t = _t(replay_buffer['future_state'])
        action_t = _t(replay_buffer['action'])
        reward_t = _t(replay_buffer['reward'])
        done = _t(replay_buffer['done'])

        with torch.no_grad():
            estimated_future_state_q = self.target_q(
                future_state_t, 
                self.target_policy(future_state_t)) 
            target_q_value = (
                reward_t + 
                self.ddpg_reward_discount * (1 - done) *
                estimated_future_state_q)
        actual_q_value = self.q_func(current_state_t, action_t)
        q_func_loss = torch.nn.functional.mse_loss(
            actual_q_value, target_q_value)
        self.q_optimizer.zero_grad()
        q_func_loss.backward()
        self.q_optimizer.step()
        # printed_indices = self.rng.choice(len(replay_buffer), 8)
        # print('target_q_value', target_q_value.numpy()[printed_indices])
        # print(
        #     'actual_q_value', actual_q_value.detach().numpy()[printed_indices])
        actual_q_value = self.q_func(
            current_state_t, self.policy(current_state_t)).mean()
        self.policy_optimizer.zero_grad()
        (-actual_q_value).backward()
        self.policy_optimizer.step()
        polyak_update(self.target_policy, self.policy, 0.995)
        polyak_update(self.target_q, self.q_func, 0.995)
        # print('q_func_loss {}', q_func_loss.item())
        # print('actual_q_value {}', actual_q_value.item())
