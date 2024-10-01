import os
import gc
import torch
import pygame
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 2024


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, action, next_state, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size):
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=device) for i in indices]).to(
            device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=device)
        next_states = torch.stack(
            [torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)
        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.dones)


class Dueling_DQN_Network(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(Dueling_DQN_Network, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        adv = self.advantage(x)
        val = self.value(x)
        return val + adv - adv.mean()


class DQN_Agent:
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay, clip_grad_norm, learning_rate, discount,
                 memory_capacity, temperature, min_temperature, decay_rate):
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.action_space = env.action_space
        self.action_space.seed(seed)
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity)
        self.temperature = float(temperature)
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate
        self.main_network = Dueling_DQN_Network(num_actions=self.action_space.n,
                                                input_dim=self.observation_space.shape[0]).to(device)
        self.target_network = Dueling_DQN_Network(num_actions=self.action_space.n,
                                                  input_dim=self.observation_space.shape[0]).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.clip_grad_norm = clip_grad_norm
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()
        with torch.no_grad():
            Q_values = self.main_network(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(Q_values).item()
            return action

    def learn(self, batch_size, done):
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        predicted_q = self.main_network(states).gather(dim=1, index=actions)
        with torch.no_grad():
            next_main_q = self.main_network(next_states).max(dim=1, keepdim=True)[1]
            next_target_q = self.target_network(next_states).gather(dim=1, index=next_main_q)
        next_target_q[dones] = 0
        y_js = rewards + (self.discount * next_target_q)
        loss = self.critertion(predicted_q, y_js)
        self.running_loss += loss.item()
        self.learned_counts += 1
        if done:
            episode_loss = self.running_loss / self.learned_counts
            self.loss_history.append(episode_loss)
            self.running_loss = 0
            self.learned_counts = 0
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step()

    def hard_update(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_epsilon(self):
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def decay_temperature(self):
        self.temperature = max(self.min_temperature, self.temperature * self.decay_rate)

    def save(self, path):
        torch.save(self.main_network.state_dict(), path)


class Model_TrainTest:
    def __init__(self, hyperparams):
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]
        self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        self.render = hyperparams["render"]
        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.memory_capacity = hyperparams["memory_capacity"]
        self.render_fps = hyperparams["render_fps"]
        self.temperature = hyperparams["temperature"]
        self.min_temperature = hyperparams["min_temperature"]
        self.decay_rate = hyperparams["decay_rate"]
        self.env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0,
                            turbulence_power=1.5, render_mode="human" if self.render else None)
        self.env.metadata['render_fps'] = self.render_fps
        self.agent = DQN_Agent(env=self.env, epsilon_max=self.epsilon_max, epsilon_min=self.epsilon_min,
                               epsilon_decay=self.epsilon_decay, clip_grad_norm=self.clip_grad_norm,
                               learning_rate=self.learning_rate, discount=self.discount_factor,
                               memory_capacity=self.memory_capacity, temperature=self.temperature,
                               min_temperature=self.min_temperature, decay_rate=self.decay_rate)

    def state_preprocess(self, state: int, num_states: int):
        onehot_vector = torch.zeros(num_states, dtype=torch.float32, device=device)
        onehot_vector[state] = 1
        return onehot_vector

    def train(self):
        total_steps = 0
        self.reward_history = []
        self.e_greedy_history = []
        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)
                self.agent.replay_memory.store(state, action, next_state, reward, done)
                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (done or truncation))
                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()
                state = next_state
                episode_reward += reward
                step_size += 1

                # Appends for tracking history
            self.reward_history.append(episode_reward)  # episode reward
            self.e_greedy_history.append(self.agent.epsilon_max)
            total_steps += step_size

                # Decay epsilon at the end of each episode
            self.agent.update_epsilon()

                # -- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')

            result = (f"Episode: {episode}, "
                      f"Total Steps: {total_steps}, "
                      f"Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon_max:.2f}")
            print(result)
        self.plot_training(episode)

    def test(self, max_episodes):
        """
        Reinforcement learning policy evaluation.
        """

        # Load the weights of the test_network
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()

        # Testing loop over episodes
        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                # state = self.state_preprocess(state, num_states=self.num_states)
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                state = next_state
                episode_reward += reward
                step_size += 1

            # Print log
            result = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Reward: {episode_reward:.2f}, ")
            print(result)

        pygame.quit()  # close the rendering window





    def plot_training(self, episode):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(self.reward_history, np.ones(50) / 50, mode='valid')

        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./lunarLander/D3QN/reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_history, label='Loss', color='#CB291A', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./lunarLander/D3QN/Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.title("Epsilon")
        plt.plot(self.e_greedy_history, label='Epsilon', color='#0000ff', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./lunarLander/D3QN/Temperature.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # Parameters:
    train_mode = False
    render = not train_mode
    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": f'./lunarLander/D3QN/final_weights' + '_' + '1500' + '.pth',
        "save_path": f'./lunarLander/D3QN/final_weights',
        "save_interval": 500,
        "clip_grad_norm": 3,
        "learning_rate": 15e-5,
        "discount_factor": 0.99,
        "batch_size": 32,
        "update_frequency": 4,
        "max_episodes": 3000 if train_mode else 5,
        "max_steps": 500,
        "render": render,

        "epsilon_max": 0.999 if train_mode else -1,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.999,

        "memory_capacity": 1_000_000 if train_mode else 0,

        "temperature": 6.0,
        "min_temperature": 0.1,
        "decay_rate": 0.99,
        "render_fps": 0,
    }

    # Run
    DRL = Model_TrainTest(RL_hyperparams)  # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes=RL_hyperparams['max_episodes'])

