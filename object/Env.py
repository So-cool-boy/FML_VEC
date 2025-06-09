# -*- coding: UTF-8 -*-

import numpy as np
import traci  # SUMO接口
import socket  # OMNeT++通信速率接口，可按需替换为ZeroMQ/REST等


class VANETEnv:
    def __init__(self, num_vehicles, sumo_port=8813):
        self.num_vehicles = num_vehicles
        self.sumo_port = sumo_port
        self.step_count = 0
        self.omnet_host = '127.0.0.1'
        self.omnet_port = 9000

        self._connect_to_sumo()

    def _connect_to_sumo(self):
        if not traci.isLoaded():
            traci.init(self.sumo_port)

    def get_state(self, vehicle_id):
        # SUMO位置与速度
        try:
            pos = traci.vehicle.getPosition(vehicle_id)  # (x, y)
            speed = traci.vehicle.getSpeed(vehicle_id)
        except traci.exceptions.TraCIException:
            pos = (0.0, 0.0)
            speed = 0.0

        # OMNeT++通信速率
        comm_rate = self.get_comm_rate_from_omnet(vehicle_id)

        state = np.array([pos[0], pos[1], speed, comm_rate], dtype=np.float32)
        return state

    def get_comm_rate_from_omnet(self, vehicle_id):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.omnet_host, self.omnet_port))
                s.sendall(vehicle_id.encode())
                data = s.recv(1024).decode()
                return float(data)
        except Exception as e:
            print(f"[OMNeT++] Error for {vehicle_id}: {e}")
            return 0.0

    def step(self, action):
        reward = -np.abs(action - 0.5)
        done = self.step_count > 50
        self.step_count += 1
        return None, reward, done

