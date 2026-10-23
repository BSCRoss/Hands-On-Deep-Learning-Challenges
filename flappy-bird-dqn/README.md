# Flappy Bird AI - Neural-Network Q-Learning

## Overview

Implemented a reinforcement learning agent that learns to play Flappy Bird by interacting with the game environment and updating action-value estimates using a neural network.

## Implementation

- **Neural network:** Fully connected network with two hidden layers of 64 neurons, using ReLU activations.
- **State representation:** Uses the 12-dimensional environment observation as input.
- **Action selection:** Epsilon-greedy policy balancing exploration and exploitation.
- **Learning algorithm:** One-step Q-learning with a discounted estimate of future rewards.
- **Optimization:** Adam optimizer with a mean squared error loss.
- **Exploration schedule:** Epsilon decays from 1.0 toward a minimum of 0.01.

## Technologies

Python · PyTorch · Gymnasium · Flappy Bird Gymnasium

## Training

The training function runs for up to 500 episodes by default and logs the episode reward and exploration rate every 10 episodes.

The original implementation was designed to run within the ETH CodeExpert challenge environment, which supplies the game environment and invokes the model initialization and training functions.

## Results

See [Results](results.md) for training metrics,
evaluation scores, and observations.