from warnings import warn

from reinforcement_learning.common import (
    ReplayBuffer, zip_dictionaries)

class ReplayBufferedTrainerMixin:

    def __init__(
            self, replay_buffer_max_size, batch_size, 
            **kwargs):
        super().__init__(**kwargs)
        self.replay_buffer = ReplayBuffer(
            rng=self.rng, max_size=replay_buffer_max_size)
        self.batch_size = batch_size
    
    def _train_on_episode(
            self, observations, actions, rewards, discounted_rewards):
        warn('Training with replay buffer')
        self.replay_buffer.append_train_one_episode_result(
            observations, actions, rewards, discounted_rewards)
        replay_buffer_sample = self.rng.choice(
            self.replay_buffer, self.batch_size)
        replay_buffer_sample = zip_dictionaries(
            replay_buffer_sample)
        return self._train_on_replay_buffer(
            replay_buffer_sample)


class BatchedTrainerMixin:
    def __init__(
            self, num_steps_per_batch,
            **kwargs):
        super().__init__(**kwargs)
        self.num_steps_per_batch = num_steps_per_batch
        self.replay_buffer = ReplayBuffer()
    
    def _train_on_episode(
            self, observations, actions, rewards, discounted_rewards):
        warn('Batched training')
        extra_kwags = self._extra_buffer_kwargs(
            observations, actions, rewards, discounted_rewards)
        self.replay_buffer.append_train_one_episode_result(
            observations, actions, rewards, discounted_rewards,
            **extra_kwags)
        if len(self.replay_buffer) < self.num_steps_per_batch:
            return
        # print('training!')
        replay_buffer_dict = zip_dictionaries(self.replay_buffer)
        result = self._train_on_replay_buffer(replay_buffer_dict)
        self.replay_buffer.clear()
        return result
