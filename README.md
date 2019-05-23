# Comparing multi-task and meta-reinforcement learning approaches in continuous action spaces with explicit sparse reward function

<kbd>![DDPG](https://media.giphy.com/media/TJhAQ0LZAmiLVhphA9/giphy.gif)</kbd><kbd>![MAML](https://media.giphy.com/media/UWOtisD6E0XfTuW3bm/giphy.gif)</kbd><br>
<p align="center"> Left:DDPG+HER agent, Right: MAML with GCP agent, both performing in zone ABC </p><br>

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




