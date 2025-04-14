from collections import namedtuple, deque
import torch.nn as nn
import torch.optim as optim
import math
import torch
import torch.nn.functional as F
import random

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward"])

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_action, n_neurons, candidate_pool_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_neurons)
        self.layer2 = nn.Linear(n_neurons, n_neurons)
        self.layer3 = nn.Linear(n_neurons, n_action * candidate_pool_size)
        self.n_action = n_action
        self.candidate_pool_size = candidate_pool_size

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        # Reshape the output into [batch_size, n_action, candidate_pool_size]
        return x.view(-1, self.n_action, self.candidate_pool_size)

class GreedyDQNAgent:
    def __init__(
        self,
        env,
        batch_size=128,
        gamma=0.8,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=500,
        lr=0.001,
        n_neurons=128,
    ):

        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr
        self.n_neurons = n_neurons

        self.n_observations = self.env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]

        state = env.reset()

        self.policy_net = DQN(self.n_observations, self.n_actions, self.n_neurons, len(self.env.df)).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # Retrieve candidate pool size and slate size.
        candidate_pool_size = len(self.env.df)  # total number of candidate papers
        slate_size = self.env.action_space.shape[0]  # number of recommendations to output (e.g., 10)
        
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state)
                mask = torch.zeros(1, candidate_pool_size, device=device, dtype=torch.bool)
                if self.env.recommended_papers:
                    invalid_indices = list(self.env.recommended_papers)
                    mask[0, invalid_indices] = True
                mask = mask.unsqueeze(1).expand(1, slate_size, candidate_pool_size)
                q_values = q_values.masked_fill(mask, float('-inf'))
                actions = torch.argmax(q_values, dim=2)
                return actions.squeeze(0).tolist()
        else:
            valid_candidates = list(set(range(candidate_pool_size)) - self.env.recommended_papers)
            # If there are fewer valid candidates than needed, fallback to the full candidate set.
            if len(valid_candidates) < slate_size:
                valid_candidates = list(range(candidate_pool_size))
            random_action_list = random.sample(valid_candidates, slate_size)
            return random_action_list

    def train(self, num_episodes=1000):
        rewards = []
        for i_episode in range(num_episodes):
            terminated = False
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            episode_reward = 0

            while not terminated:
                action = self.select_action(state)
                observation, reward, terminated, _, _ = self.env.step(action)
                episode_reward += reward

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)

                state = next_state

            rewards.append(episode_reward)
            self.optimize_model()

        return rewards

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.stack([torch.tensor(x, dtype=torch.long, device=device) for x in batch.action])
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1, 1)

        q_values = self.policy_net(state_batch)
        action_batch_exp = action_batch.unsqueeze(2)
        chosen_q = q_values.gather(2, action_batch_exp).squeeze(2)
        state_action_values = chosen_q.sum(dim=1, keepdim=True)
        next_state_values = torch.zeros(self.batch_size, device=device)

        if non_final_next_states.shape[0] > 0:
            next_q_values = self.policy_net(non_final_next_states)
            next_max_per_slot = next_q_values.max(dim=2)[0]
            next_state_values[non_final_mask] = next_max_per_slot.sum(dim=1)

        expected_state_action_values = reward_batch + (self.gamma * next_state_values.unsqueeze(1))

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()