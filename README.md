# 2048 Deep Recurrent Reinforcement Learning

**Current Status**: *Unsuccessful*

## Introduction and Motivation

While many AI solvers already exist for the game 2048, a lot of them depend on *expextimax search* with a hand crafted features for the evaluation function. This work was initially motivated by the desire to create a solver that learns how to understand and play the game without any human engineered features, which makes Reinforcement Learning with a learned function approximation the suitable approach to tackle such problem.

Upon searching for other attempts, many were already made at solving 2048 with RL. The earliest successful attempts [[3](#ref3), [4](#ref4)] used *n-tuple networks* for as a function approximation method with TD-learning. However, the size of the network is immensely large (1M and 24M weights). A inferior attempt using Q-learning with deep neural networks without feature engineering used relatively smaller yet considerably large set of weights [[5](#ref5)], but resulted in superior results when trained under the supervision of an hand-crafted expextimax-based AI.

After the release of Google DeepMind's DQN architecture [[1](#ref1)], many attempts were made at adapting it to 2048 [[2](#ref2), [3](#ref3), [4](#ref4)]. However, all of these approaches were unsuccessful, probably because of the small size of the neural networks used.

After this survey, this work was motivated by the desire to explore the idea that using DQN with a recurrent layer to account for the temporal dependencies between the states way allow for the usage of a smaller network in solving 2048. The work is based on the modification of DQN to include recurrent layers introduced in [[2](#ref2)].

## Overview

DRQN works just like DQN, with the exception that the reply memory in DRQN store the whole episode in its temporal order instead of individual transitions. At each training step, a random episode is selected from the reply memory and starting from a random transition in that episode, a specified number (16 or 32) of consecutive episodes is chosen and passed to a Q network with the architecture depicted in the following figure.

![](http://i.imgur.com/9VT8u7n.png)

The loss is function is the same as the one in DQN but in addition to taking the mean across the time steps:

![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cmathcal%7BL%7D%28%5Ctheta%29%3D%5Cmathbb%7BE%7D_%7Be%27%20%5Csim%20%5Cmathcal%7BD%7D%7D%5Cleft%5B%20%5Cmathbb%7BE%7D_%7B%28s%2Ca%2Cr%2Cs%27%29%20%5Cin%20e%27%7D%5Cleft%5B%5Cleft%28r%20&plus;%20%5Cgamma%5Cmax_%7Ba%27%7DQ%28s%27%2Ca%27%3B%5Coverline%7B%5Ctheta%7D%29%20-%20Q%28s%2C%20a%3B%5Ctheta%29%20%5Cright%20%29%5E2%20%5Cright%20%5D%5Cright%5D)

---
## References

<a id="ref1">[1]</a> Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).

<a id="ref2">[2]</a> Hausknecht, Matthew, and Peter Stone. "Deep recurrent q-learning for partially observable mdps." arXiv preprint arXiv:1507.06527 (2015).

<a id="ref3">[3]</a> Szubert, Marcin, and Wojciech Jaśkowski. "Temporal difference learning of N-tuple networks for the game 2048." 2014 IEEE Conference on Computational Intelligence and Games. IEEE, 2014.
APA

<a id="ref4">[4]</a> Wu, I-Chen, et al. "Multi-Stage Temporal Difference Learning for 2048." Technologies and Applications of Artificial Intelligence. Springer International Publishing, 2014. 366-378.

<a id="ref5">[5]</a> Tjwei. "2048-NN" Github repository, https://github.com/tjwei/2048-NN (accessed August 12, 2016)

<a id="ref2">[6]</a> Weston, Travis. "2048-Deep-Learning" Github repository, https://github.com/anubisthejackle/2048-Deep-Learning (accessed August 12, 2016)

<a id="ref3">[7]</a> Matiisen, Tambet. "matlab2048" Github repository, https://github.com/tambetm/matlab2048 (accessed August 12, 2016)

<a id="ref4">[8]</a> Yoon, Wonjun. "2048_deepql_torch" Github repository, https://github.com/wonjunyoon/2048_deepql_torch (accessed August 12, 2016)
