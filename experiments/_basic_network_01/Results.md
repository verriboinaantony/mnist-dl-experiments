# Experiment 02 — Single Layer Network

## Goal

Test how model performs with reduced complexity.
Compare against 3 layer baseline to understand
the effect of depth on accuracy.

## Architecture

```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)   # directly input to output
)
```

## Hyperparameters

| Parameter     | Value |
| ------------- | ----- |
| Optimizer     | SGD   |
| Learning rate | 0.01  |
| Batch size    | 64    |
| Epochs        | 10    |

## Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| ----- | ---------- | --------- | -------- | ------- |
| 1     | 0.9820     | 79.89%    | 0.6001   | 86.68%  |
| 2     | 0.5519     | 86.70%    | 0.4762   | 88.59%  |
| 3     | 0.4722     | 87.90%    | 0.4260   | 89.29%  |
| 4     | 0.4334     | 88.64%    | 0.3975   | 89.63%  |
| 5     | 0.4092     | 89.09%    | 0.3786   | 90.08%  |
| 6     | 0.3924     | 89.44%    | 0.3649   | 90.24%  |
| 7     | 0.3797     | 89.69%    | 0.3544   | 90.49%  |
| 8     | 0.3698     | 89.90%    | 0.3466   | 90.61%  |
| 9     | 0.3618     | 90.08%    | 0.3402   | 90.77%  |
| 10    | 0.3550     | 90.21%    | 0.3345   | 90.84%  |

## Comparison with Experiment 01

| Metric         | 3 Layer Network | 1 Layer Network | Difference |
| -------------- | --------------- | --------------- | ---------- |
| Train Accuracy | 93.59%          | 90.21%          | -3.38%     |
| Val Accuracy   | 93.87%          | 90.84%          | -3.03%     |
| Train Loss     | 0.2252          | 0.3550          | +0.1298    |
| Val Loss       | 0.2123          | 0.3345          | +0.1222    |

## Diagnosis

High Bias — model is too simple to learn complex patterns.

```
Train Accuracy → 90.21%
Val Accuracy   → 90.84%
Gap            → 0.63%  ← small gap but both are low
```

Small gap means no overfitting. But both numbers are
lower than the 3 layer network — classic underfitting.

## What I learned

- More layers = more complexity = better pattern learning
- Even 1 layer reaches 90% because MNIST is a simple dataset
- On harder datasets the gap would be much larger
- Depth matters — removing layers hurts performance

## Next Experiment

Switch SGD to Adam optimizer on the 3 layer network.
Target: push accuracy above 96%.
