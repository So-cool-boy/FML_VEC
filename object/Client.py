# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from algorithm.STDRQN import STDRQNAgent

class VehicleClient:
    def __init__(self, client_id, model_fn, buffer, lr=1e-3):
        self.id = client_id
        self.model = model_fn()
        self.buffer = buffer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def local_update(self, epochs=1, batch_size=32):
        for _ in range(epochs):
            sampled = self.buffer.sample(batch_size)
            if sampled is None:
                continue
            batch, idxs, weights = sampled
            loss, td_errors = self.compute_loss(batch, weights)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.buffer.update_priorities(idxs, td_errors.detach().numpy())

    def compute_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        q_values, _ = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            q_next, _ = self.model(next_states)
            target = rewards + (1 - dones) * 0.99 * q_next.max(1)[0]

        td_error = (q_value - target).abs()
        return (weights * (q_value - target).pow(2)).mean(), td_error

    def get_weights(self):
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
