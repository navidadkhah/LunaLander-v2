# Lunar Lander version 2

This project implements the **Lunar Lander v2** problem using Deep Reinforcement Learning (DRL) techniques. The primary focus is on evaluating and improving the performance of the **Deep Q-Network (DQN)** by utilizing **Dueling DQN (D3QN)** architectures. This work was completed as part of a reinforcement learning course assignment.

https://github.com/user-attachments/assets/4be15cb7-94a9-4df5-8371-6adba5845043

## Algorithm Summaries

### Deep Q-Network (DQN)
DQN is a popular reinforcement learning algorithm that combines Q-learning with deep neural networks. In the **Lunar Lander** environment, DQN uses a deep network to approximate the Q-value function, which tells the agent how good or bad it is to take certain actions in specific states.

However, DQN suffers from:
- **Overestimation Bias**: It tends to overestimate the action values.
- **Instability**: Training can become unstable due to correlated updates.

<p align="center">
  <img alt="Rewards" src="https://github.com/user-attachments/assets/f3bdba55-0cde-4761-9508-fbf264f2c867" width="330">
  <img alt="Loss" src="https://github.com/user-attachments/assets/500d70c7-3acb-4add-83dc-8e3b8159c6ea" width="330">
  <img alt="Epsilon" src="https://github.com/user-attachments/assets/dbc26ca8-4ecc-4178-95ea-787418fa7a27" width="330">
</p>

### Dueling Double DQN (D3QN)
D3QN further enhances performance by incorporating the dueling architecture, where the Q-value function is split into two streams:
1. **State Value Function (V(s))**: How good it is to be in a state, regardless of action.
2. **Advantage Function (A(s, a))**: The benefit of taking a specific action compared to others.

This helps the agent learn more efficiently in the **Lunar Lander** by better differentiating between valuable states and actions, improving both stability and performance.

### D3QN Summary for Lunar Lander
By combining **Double DQN** and **Dueling Networks**, **D3QN** offers significant improvements in solving the Lunar Lander problem. It results in:
- **Reduced overestimation** of Q-values.
- **Improved state value approximation**, making the agent more adept at landing in difficult scenarios.
- **Faster convergence** and better reward maximization than standard DQN.

<p align="center">
  <img alt="Rewards" src="https://github.com/user-attachments/assets/0386ff64-ff8a-44cb-96f3-ae8bb9bb2b49" width="330">
  <img alt="Loss" src="https://github.com/user-attachments/assets/6e08e2e0-0bc0-4cb5-9bc7-e939b11d0b07" width="330">
  <img alt="Epsilon" src="https://github.com/user-attachments/assets/195400e4-cc1b-47c0-8c94-4bc21d81d754" width="330">
</p>

## Implementation Details

### Algorithms
- **DQN**: The baseline algorithm, implemented with a simple feedforward network and epsilon-greedy exploration.
- **D3QN**: An advanced version incorporating Double Q-learning and Dueling Network architectures, which leads to better stability and faster convergence.

### Key Hyperparameters
- **Network architecture**: 
  - 3 fully connected layers with 64 neurons each
  - Activation: ReLU
- **Loss Function**: Mean Squared Error (MSE)
- **Exploration Strategy**: Epsilon-greedy with early stopping
- **Optimizer**: Adam
- **Discount Factor**: Varying γ over time as described in the report

(For a more comprehensive list of hyperparameters, refer to the report.)

## Results

The **D3QN** model significantly outperformed the DQN in terms of stability and reward maximization. By introducing a dynamic gamma strategy and optimizing the training process, we achieved consistent improvements in solving the Lunar Lander environment.

### GIFs
## DQN:
Episode reward: 180.14

https://github.com/user-attachments/assets/15a7ffc9-c1b4-4bd7-81fb-6367cc13871b

## D3QN:
Episode reward: 249.5

https://github.com/user-attachments/assets/1ad163dc-e36f-46ee-83d5-6fe57db30b87





## Conclusion

This project demonstrated the effectiveness of advanced DRL techniques in solving the Lunar Lander problem. The D3QN model, in particular, offered substantial improvements over the baseline DQN, and future work may explore further enhancements such as prioritized experience replay or multi-agent setups.

---

