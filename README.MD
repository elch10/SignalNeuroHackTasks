# The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

The problem is that neural networks have many parameters (weights), that hard to learn. The *Jonathan Frankle* and *Michael Carbin* find the pruning technique that can find smaller part of the neural networks that are responsible for resolving current task. The article can found [here](https://openreview.net/forum?id=rJl-b3RcF7)

This repository contains implementation of that idea on `PyTorch` and examples with `Lenet-300-100` and `Conv-Lenet-5`.

We can prune 75% of `Lenet-300-100` on the `MNIST` dataset and 25% of `Conv-Lenet-5` on `SVHN` dataset. But it is not end, we didn't search over all different *pruning* percents for different models due limited resources.