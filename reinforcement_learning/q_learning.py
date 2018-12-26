from gym.spaces import Discrete, Box
import numpy as np
import torch

from reinforcement_learning.common import SimpleTrainer, one_hot

class QModule(torch.nn.Module):
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
        if not isinstance(env.action_space, Discrete):
            raise ValueError('Unsupported action space {}'.format(action_space))

        super().__init__()
        self.num_actions = env.action_space.n
        self.hidden_units = hidden_units
        last_layer = torch.nn.Linear(self.hidden_units, 1, bias=False)
        last_layer.weight.data.zero_()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(
                env.observation_space.shape[0] + self.num_actions, 
                self.hidden_units,),
            torch.nn.LeakyReLU(),
            last_layer,
        )
    
    def forward(self, observation, action):
        one_hot_action = one_hot(action, self.num_actions)
        return self.model(
            torch.cat([observation, one_hot_action], dim=1))[..., 0]


def _copy_model(dest, src):
    dest.load_state_dict(src.state_dict())

class QLearningTrainer(SimpleTrainer):

    def __init__(self, env, agent, target_agent=None, **kwargs):
        if not isinstance(env.action_space, Discrete):
            raise ValueError('Unsupported action space {}'.format(action_space))
        super().__init__(env, **kwargs)
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-2)
        self.copy_to_target_prob = 0.1
        if target_agent is None:
            target_agent = type(agent)()
        self.target_agent = target_agent
        _copy_model(self.target_agent, self.agent)
    
    def _choose_action(self, observation):
        with torch.no_grad():
            observation_t = torch.tensor(
                [observation], dtype=torch.float32)
            action_values = [
                self.agent(
                    observation_t, torch.tensor([action])).numpy()[0]
                for action in range(self.env.action_space.n)]
        if self.rng.uniform() < self.exploration_prob:
            chosen_action = self.env.action_space.sample()
        else:
            chosen_action = np.argmax(action_values)
        return chosen_action
    
    def _train_on_episode(self, observations, actions, rewards):
        next_step_estimated_value = 0
        t = len(rewards)
        losses = []
        for i in range(t - 1, -1, -1):
            target = rewards[i] - self.reward_discount *  next_step_estimated_value
            current_step_estimated_value = self.agent(
                torch.tensor([observations[i]], dtype=torch.float32), 
                torch.tensor([actions[i]]))
            losses.append(torch.nn.functional.l1_loss(
                current_step_estimated_value, 
                torch.tensor(target)))
            with torch.no_grad():
                next_step_estimated_value = self.target_agent(
                    torch.tensor([observations[i]], dtype=torch.float32), 
                    torch.tensor([actions[i]])).item()
        loss = torch.mean(torch.stack(losses))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.rng.uniform() < self.copy_to_target_prob:
            _copy_model(self.target_agent, self.agent)
        return next_step_estimated_value
