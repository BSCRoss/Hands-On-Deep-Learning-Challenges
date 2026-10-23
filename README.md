# Hands-On-Deep-Learning-Challenges

Hands-On Deep Learning Challenges — ETH Zürich

A collection of selected implementations from the Hands-On Deep Learning course at ETH Zürich. This repository highlights practical work with deep learning, reinforcement learning, and graph neural networks.
Coursework note: These projects are based on ETH Zürich course challenges. 

Projects
1. Flappy Bird Agent — Deep Reinforcement Learning
Goal: Train an agent to play Flappy Bird using reinforcement learning.

Approach
- Implemented a Deep Q-Network (DQN) agent.
- Trained the agent through interaction with the game and reward-based feedback.
- Evaluated behavior and learning performance during training.
Technologies: Python, PyTorch


2. Maze Solving with Graph Neural Networks
Goal: Predict which nodes belong to the unique path between two marked endpoints in a maze represented as a graph.

Approach
- Implemented a graph neural network (GNN) for node-level binary classification.
- Used node features to identify the two endpoints and predicted whether each node belongs to the connecting path.
- Trained only on 4×4 maze graphs, with the challenge evaluating generalization to larger 8×8, 16×16, and 32×32 graphs.
Technologies: Python, PyTorch, PyTorch Geometric