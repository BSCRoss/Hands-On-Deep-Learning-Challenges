# Results — Maze Solving with GNNs

## 1. Challenge Score

| Metric | Result |
|---|---:|
| CodeExpert challenge score | 0.3166 |
| Score thresholds passed | 3 |

## 2. Validation Performance

| Epoch | Loss | F1 Score | Node Accuracy | Graph Accuracy |
|---:|---:|---:|---:|---:|
| 10 | 0.1377 | 0.934 | 96.3% | 90.0% |
| 20 | 0.0729 | 0.972 | 98.3% | 93.3% |
| 30 | 0.0535 | 0.938 | 96.5% | 90.0% |
| 40 | 0.0210 | 0.986 | 99.2% | 90.0% |
| 50 | 0.0159 | 0.982 | 99.0% | 90.0% |
| 60 | 0.0137 | 0.990 | 99.4% | 93.3% |

*Validation metrics are measured on the 4×4 maze validation set. They should not be interpreted as performance on larger mazes.*

## 3. Observations

- Validation loss decreased substantially during training.
- Node-level F1 and accuracy were high on the validation set.
- The challenge score was considerably lower than the validation metrics, suggesting that further investigation of generalization is worthwhile.
- The current log does not provide separate results for 8×8, 16×16 and 32×32 mazes.

## 4. Future Improvements

- Evaluate performance separately for each maze size.
- Investigate generalization to larger graphs.
- Compare different hidden dimensions and message-passing depths.
- Experiment with training strategies and validation splits.
