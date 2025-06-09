## 1. Dependency package

- torch 1.13.1
- numpy 1.23.5
- traci 1.18
- matplotlib 3.7.1
- SUMO 1.8.0
- OMNeT++ 5.6.2 
- INET Framework 3.7.0

## 2. Public data sets

```text
@article{Liu2016LargescaleVR,
  title={Large-scale vehicle re-identification in urban surveillance videos},
  author={Xinchen Liu and Wu Liu and Huadong Ma and Huiyuan Fu},
  journal={2016 IEEE International Conference on Multimedia and Expo (ICME)},
  year={2016},
  pages={1-6}
}
```

## 3. Code Description

- algorithm/DQN.py: Traditional DQN algorithm
- algorithm/DRQN.py: DQN integrated with LSTM
- algorithm/STDRQN.py: DQN that combines LSTM and priority sampling
- object/Client.py: Key Code of Federated Learning Client
- object/Client.py: Key Code of Federated Learning Server
- object/Env.py: Key code for interactive environment, where data acquisition is coordinated with SUMO and OMNet++.
- tools/....py: Some other code tools
