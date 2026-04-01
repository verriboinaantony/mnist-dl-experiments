## Experiment 01 — Basic Network

### Architecture

- 3 layer fully connected network
- No regularization
- No normalization

### Hyperparameters

- Optimizer : SGD
- Learning rate : 0.01
- Batch size : 64
- Epochs : 10

### Results

| Metric         | Value  |
| -------------- | ------ |
| Train Accuracy | 93.59% |
| Val Accuracy   | 93.87% |
| Test Accuracy  | 93.90% |
| Train Loss     | 0.2252 |
| Val Loss       | 0.2123 |

### Diagnosis

Good fit — small gap between train and val accuracy.
Model is not overfitting or underfitting.
But accuracy is stuck at 93-94% — SGD is slowing down.

### What to improve next

Switch SGD to Adam optimizer.
Adam adapts learning rate automatically — should converge faster and reach higher accuracy.
