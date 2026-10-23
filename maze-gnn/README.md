# Maze Solving with Graph Neural Networks

A Graph Neural Network (GNN) for predicting which nodes belong to the solution path of a maze, implemented using PyTorch and PyTorch Geometric.

## Overview

The maze is represented as a graph, with nodes corresponding to maze cells and edges representing valid connections. The input features identify the maze's endpoints. The model predicts a binary label for every node, indicating whether it belongs to the path between the endpoints.

The challenge is to learn from small mazes and generalize to larger graph structures.

## Model Architecture

The model consists of three main components:

- **Node encoder:** Maps the two-dimensional input features into a 96-dimensional hidden representation.
- **Message-passing layers:** A custom `MazeConv` layer constructs messages from the representations of neighboring nodes. Each message uses the source representation, destination representation, and their difference.
- **Node decoder:** Maps the final node representations to two class logits, predicting whether each node belongs to the maze solution.

### Message passing and updates

The custom message-passing layer uses:
- A multilayer perceptron (MLP) to compute messages.
- Additive aggregation to combine incoming messages.
- A GRU cell to update node representations.
- Layer normalization to stabilize representations.

The original encoded node features are concatenated with the current representations before each message-passing round, helping preserve information about the endpoints.

### Adaptive message-passing depth

The number of message-passing rounds depends on the number of nodes:

`max(8, int(2.5 * sqrt(num_nodes)) + 4)`

This increases the model's propagation depth for larger mazes.

## Training

- **Frameworks:** PyTorch, PyTorch Geometric
- **Hidden dimension:** 96
- **Optimizer:** Adam
- **Loss:** Negative log-likelihood
- **Training schedule:** 100 epochs, with cosine learning-rate decay
- **Regularization:** Weight decay, dropout, and gradient clipping
- **Model selection:** Selects the checkpoint with the best validation metric, combining node-level F1 and the proportion of perfectly classified validation graphs.

The training code prioritizes 4×4 maze graphs when constructing the training and validation sets.

## What I Learned

- Implementing custom message-passing layers for graph-based prediction.
- Combining MLPs, GRU updates, and normalization in a GNN.
- Designing training loops with validation, learning-rate scheduling, and checkpoint selection.
- Exploring how message-passing depth affects generalization across graph sizes.
