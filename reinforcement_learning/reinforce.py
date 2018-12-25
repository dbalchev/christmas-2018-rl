import torch
import numpy as np

class LunarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_units = 48
        last_layer = torch.nn.Linear(self.hidden_units, 4, bias=False)
        last_layer.weight.data.zero_()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, self.hidden_units,),
            torch.nn.LeakyReLU(),
            last_layer,
            torch.nn.Softmax(),
        )
    
    def forward(self, inputs):
        return self.model(inputs)


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


class ReinforceTrainer:

    def __init__(self, env, agent, should_render=False):
        self.env = env
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-2)
        self.should_render = should_render
        self.rng = np.random.RandomState()
    
    def train_one_episode(self, exploration_prob, reward_discount):
        observations, actions, rewards = self._run_episode(exploration_prob)
        total_reward = np.sum(rewards)
        rewards = _discount_rewards(rewards, reward_discount)
        loss = self._train_on_episode(observations, actions, rewards)
        return total_reward, loss

    def _maybe_render(self):
        if self.should_render:
            self.env.render()

    def _run_episode(self, exploration_prob):
        rewards = []
        done = False
        observation = self.env.reset()
        observations = [observation]
        actions = []
        self._maybe_render()
        while not done:
            with torch.no_grad():
                action_probs = self.agent(
                    torch.tensor([observation], dtype=torch.float32)).numpy()[0]
            chosen_action = self.rng.choice(np.arange(4), p=action_probs)
            if self.rng.uniform() < exploration_prob:
                chosen_action = self.rng.randint(4)
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

    def _train_on_episode(self, observations, actions, rewards):
        # print(rewards)
        action_probabilities = self.agent(
            torch.tensor(observations, dtype=torch.float32))
        chosen_action_probabilities = action_probabilities[np.arange(len(actions)), actions]
        loss = -chosen_action_probabilities.log() * torch.tensor(rewards, dtype=torch.float32)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_value = loss.detach().item()
        return loss_value # chosen_action_probabilities.detach().numpy()
