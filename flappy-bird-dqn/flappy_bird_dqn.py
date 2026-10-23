import numpy as np
import gymnasium as gym
import pygame
import torch
import torch.nn as nn
import flappy_bird_gymnasium

class NormalizeObservation(gym.ObservationWrapper):
    def observation(self, obs):
        return obs.astype(np.float32)

def apply_wrappers(env):
    env = NormalizeObservation(env)
    return env


class Agent(nn.Module):
    def __init__(self, state_dim=12, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        # Returns Q-values for each action
        return self.net(x)

    def act(self, observation, epsilon=0.0):
        # Epsilon-greedy selection
        if np.random.random() < epsilon:
            return np.random.randint(2)
        
        observation = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(observation)
        return torch.argmax(q_values).item()
  
def init_model(train_env):
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    agent = Agent(state_dim, action_dim)
    return agent

def train_model(agent, env, episodes=500):
    env = apply_wrappers(env)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Hyperparameters
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    for episode in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Simple One-Step Q-Learning Update
            target = reward
            if not terminated:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + gamma * torch.max(agent(next_state_tensor)).item()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            current_q = agent(state_tensor)[0][action]
            
            loss = criterion(current_q, torch.tensor(target, dtype=torch.float32))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")
            
    return agent