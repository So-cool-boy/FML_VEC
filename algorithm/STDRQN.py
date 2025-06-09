# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# ST—DRQN
class DRQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DRQN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        # x: [batch, seq_len, input_dim]
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        q_values = self.fc(out[:, -1, :])  # 只取最后一帧的输出
        return q_values, hidden

# SumTree
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

# 优先经验回放
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5
        self.capacity = capacity
        self.max_priority = 1.0

    def push(self, sequence):
        priority = self.max_priority
        self.tree.add(priority, sequence)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        probs = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.capacity * probs, -beta)
        is_weights /= is_weights.max()

        state_seqs, action_seqs, reward_seqs, next_state_seqs, done_seqs = zip(*batch)

        return (
            torch.tensor(state_seqs, dtype=torch.float32),
            torch.tensor(action_seqs, dtype=torch.int64),
            torch.tensor(reward_seqs, dtype=torch.float32),
            torch.tensor(next_state_seqs, dtype=torch.float32),
            torch.tensor(done_seqs, dtype=torch.float32),
            torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1),
            idxs
        )

    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error.item()) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len([d for d in self.tree.data if d is not None])

# STDRQNAgent
class STDRQNAgent:
    def __init__(self, input_dim, output_dim, hidden_dim=128, seq_len=10,
                 buffer_size=10000, batch_size=32, gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500,
                 beta_start=0.4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.beta_start = beta_start

        self.q_net = DRQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net = DRQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state_seq, hidden_state=None):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() < epsilon:
            return random.randint(0, self.output_dim - 1), hidden_state
        else:
            state_seq_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values, hidden_state = self.q_net(state_seq_tensor, hidden_state)
            return q_values.argmax().item(), hidden_state

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        beta = min(1.0, self.beta_start + self.steps_done * 1e-5)

        states, actions, rewards, next_states, dones, is_weights, idxs = self.replay_buffer.sample(self.batch_size, beta)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        is_weights = is_weights.to(self.device)

        q_values, _ = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        target_q_values = rewards.unsqueeze(1) + self.gamma * max_next_q_values * (1 - dones.unsqueeze(1))

        td_errors = target_q_values - q_values
        loss = (is_weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(idxs, td_errors.detach())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
