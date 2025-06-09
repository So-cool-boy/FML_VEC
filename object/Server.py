# -*- coding: UTF-8 -*-
from object.Client import VehicleClient

class FederatedServer:
    def __init__(self, model_fn, client_ids, buffer_fn):
        self.global_model = model_fn()
        self.clients = [VehicleClient(cid, model_fn, buffer_fn()) for cid in client_ids]

    def distribute(self):
        weights = self.global_model.state_dict()
        for client in self.clients:
            client.set_weights(weights)

    def aggregate(self):
        total = len(self.clients)
        new_state = self.global_model.state_dict()
        for k in new_state:
            new_state[k] = sum([client.get_weights()[k] for client in self.clients]) / total
        self.global_model.load_state_dict(new_state)

    def federated_round(self, local_epochs=1):
        self.distribute()
        for client in self.clients:
            client.local_update(epochs=local_epochs)
        self.aggregate()