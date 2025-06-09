# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# DRQN网络（含LSTM）
class DRQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DRQN, self).__init__()
        self.hidden_dim = hidden_dim

        # 状态向量 → LSTM → 全连接层输出Q值
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state=None):
        # x: (batch, seq_len, input_dim)
        if hidden_state is None:
            out, hidden_state = self.lstm(x)
        else:
            out, hidden_state = self.lstm(x, hidden_state)
        q_values = self.fc(out[:, -1, :])  # 取最后时间步的输出
        return q_values, hidden_state

# 经验池，存储序列
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, sequence):
        self.buffer.append(sequence)  # 每个序列是一个列表：[state_seq, action_seq, reward_seq, next_state_seq, done_seq]

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_seqs, action_seqs, reward_seqs, next_state_seqs, done_seqs = zip(*batch)
        return (
            torch.tensor(state_seqs, dtype=torch.float32),
            torch.tensor(action_seqs, dtype=torch.int64),
            torch.tensor(reward_seqs, dtype=torch.float32),
            torch.tensor(next_state_seqs, dtype=torch.float32),
            torch.tensor(done_seqs, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# DRQN Agent
class DRQNAgent:
    def __init__(self, input_dim, output_dim, hidden_dim=128, seq_len=10, buffer_size=1000, batch_size=32,
                 gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.q_net = DRQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net = DRQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
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

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)  # (batch, seq_len, input_dim)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)  # (batch,)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        q_values, _ = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        target_q_values = rewards.unsqueeze(1) + self.gamma * max_next_q_values * (1 - dones.unsqueeze(1))

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

