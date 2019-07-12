# Confronting multitask and meta-reinforcement learning approaches in continuous action spaces with explicit sparse reward function

Being able to build an artificial intelligence (AI) agent that is able to perform well on a bunch of different tasks and to generalize to new ones has been a challenge in the research area. A key assumption in this scheme is that tasks need to share a specific structure. It is this structure that is learnt by the agent and that is transferred to new assignments. <br><br>
In this repository, we investigate both multitask learning and meta-learning approaches. The former being a special case of the latter, our goal is to see whether meta-learning techniques can generalize better than classical multitask learning. Our settings impose the use of Goal Conditioned Policies to tackle the undecidability of the problem.<br><br>
We work on an existing 2d-Hypercube (rectangle) environment and personnalize it for the sake of our goals. We base our confrontation on Deep Deterministic Policy Gradients algorithms alongside with Hindsight Experience Replay for multitask learning and Model Agnostic Meta-Learning for meta-learning. We build an evaluation protocol to capture the generalization property of each algorithm we use. This protocol consists in a spatial decomposition of the rectangle in a training zone and a testing zone: during training, the goals are generated in the training zone, we then test the generalization property on previously unseen goals in previously unexplored testing zone.<br>
<p  align="center">
<img src="https://www.zupimages.net/up/19/28/pzh0.png" alt="drawing" width="300"/><br>
Evaluation protocol
 </p>
<p  align="center">
<kbd><img src="https://media.giphy.com/media/lSguOE2YDghUctOkbX/giphy.gif" alt="drawing" width="300"/></p></kbd><br>
<p  align="center">MAML with GCP agent performing on testing zone D</p><br>

#### Set up

Open a terminal and

1. git clone this repository
  ```
  $ git clone https://github.com/akakzia/meta_vs_multi.git
  ```
  
2. install the requirements needed in order to reproduce the experiments
 ```
  $ cd meta_vs_multi
  $ pip install requirements.txt
  ```

#### Train the agents
1. Train MAML agent and specify parameters or use default ones
```
  $ python meta_rl/main.py 
```
2. Train DDPG+HER agent and specify parameters or use default ones
```
  $ python her/main.py --HER=True
```
To toggle on or off the use of Goal Conditioned Policies, go to code and change GCP value in the registration of the hypercube environment


#### Help
Go to [wiki](https://github.com/akakzia/meta_vs_multi/wiki) for more help or contact me personally on ahmed.akakzia@gmail.com




