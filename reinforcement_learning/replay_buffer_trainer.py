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
    
    def _train_on_episode(self, observations, actions, rewards):
        warn('Training with replay buffer')
        self.replay_buffer.append_train_one_episode_result(
            observations, actions, rewards)
        replay_buffer_sample = self.rng.choice(
            self.replay_buffer, self.batch_size)
        replay_buffer_sample = zip_dictionaries(
            replay_buffer_sample)
        return self._train_on_replay_buffer(
            replay_buffer_sample)


class BatchedTrainerMixin:
    def __init__(
            self, num_steps_per_batch, num_trainings_per_batch,
            **kwargs):
        super().__init__(**kwargs)
        self.num_trainings_per_batch = num_trainings_per_batch
        self.num_steps_per_batch = num_steps_per_batch
        self.num_episodes_in_buffer = 0
        self.replay_buffer = ReplayBuffer()
    
    def _train_on_episode(self, observations, actions, rewards):
        warn('Batched training')
        self.replay_buffer.append_train_one_episode_result(
            observations, actions, rewards)
        self.num_episodes_in_buffer += 1
        if len(self.replay_buffer) < self.num_steps_per_batch:
            return
        print('training!')
        replay_buffer_dict = zip_dictionaries(self.replay_buffer)
        for _ in range(self.num_trainings_per_batch):
            result = self._train_on_replay_buffer(replay_buffer_dict)
        self.replay_buffer.clear()
        self.num_episodes_in_buffer = 0
        return result
