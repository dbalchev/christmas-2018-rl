import torch
import numpy as np

from reinforcement_learning.common import SimpleTrainer

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


class ReinforceTrainer(SimpleTrainer):

    def __init__(self, env, agent, exploration_prob, should_render=False):
        super().__init__(env, should_render)
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-2)
        self.exploration_prob = exploration_prob

    def _choose_action(self, observation):
        with torch.no_grad():
                action_probs = self.agent(
                    torch.tensor([observation], dtype=torch.float32)).numpy()[0]
        chosen_action = self.rng.choice(np.arange(4), p=action_probs)
        if self.rng.uniform() < self.exploration_prob:
            chosen_action = self.rng.randint(4)
        return chosen_action

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
