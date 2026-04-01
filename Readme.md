# MNIST Deep Learning Experiments

A structured learning project where I build and improve
a neural network on MNIST step by step — tracking every
experiment, result, and observation.

---

## Goal

Learn deep learning concepts practically by implementing
each improvement from Andrew Ng's course and measuring
its effect on model performance.

---

## Results Summary

| Experiment         | Key Change                   | Train Acc | Val Acc | Improvement |
| ------------------ | ---------------------------- | --------- | ------- | ----------- |
| 01 Basic Network   | Baseline 3 layers            | 93.59%    | 93.87%  |
| 02 Single Layer    | Reduced complexity           | 90.21%    | 90.84%  |
| 03 Kaiming He Init | Better weight initialization | 95.73%    | 95.65%  | +1.78%      |
| 03 Adam Optimizer  | Coming soon                  | -         | -       |

---

## Project Structure

```
MNIST/
├── experiments/    ← one folder per experiment
├── models/         ← saved model weights
├── data/           ← MNIST dataset
├── test.py         ← shared evaluation script
└── config.py       ← shared hyperparameters
```

---

## Experiments

### 01 — Basic Network

A simple 3 layer fully connected network.
No regularization, no normalization.
Optimizer: SGD | LR: 0.01 | Epochs: 5
Result: **93.90% accuracy**

### 02 — Better Optimizer

Coming soon.

### 03 — Batch Normalization

Coming soon.

### 04 — Dropout

Coming soon.

---

## How to run

Train a model:

```
python experiments/01_basic_network/train.py
```

Test a model:

```
python test.py
```

---

## Learning Resources

- Andrew Ng Deep Learning Specialization
- PyTorch official documentation
- learnpytorch.io

```

---

## Now move your current files

Move your current `model.py` and `train.py` into `experiments/01_basic_network/`. Your structure should look like:
```

MNIST/
├── experiments/
│ └── 01_basic_network/
│ ├── model.py
│ ├── train.py
│ └── results.md
├── models/
│ └── basic_net.pth
├── data/
├── test.py
├── config.py
└── README.md
