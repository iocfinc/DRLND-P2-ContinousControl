# DRLND-P2-ContinuousControl

## Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Different Training Versions

There are two possible versions of the environment to solve in this project. One is to use a single agent approach. The second option would be to use multiple identical agents (20 in all) each with its separate copy of the environment and solve the problem in a distributed manner.

The second version is there to allow us to make use of algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb). These algorithms require multiple non-interacting and parallel copies of the same agent for the experience gathering.

## Solving the Environment

For this project, we only need to solve one version of the environment. It is up to us if we want to go the single-agent route or the multiple-agent route.

For option 1: **Single-agent Environment*

>The task is episodic, and to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

For option 2: **Multiple-agent Environment**

>The barrier to solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and overall agents). Specifically,
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is overall 20 agents).

## Criteria for Solved Environment

For both environments, there is a minimum score required for the submission to be valid. For the Single-agent it needs to be a score of +30 for a rolling 100 consecutive episodes. For multiple-agents, it is a +20 rolling average for all agents in the latest 100 episodes. So in terms of timeframe considered, only the latest 100 scores are counted. An additional rule for the multiple-agent environment is that the average score for 100 episodes should be the average of all the 20 individual agents.

The idea is to check the convergence of the different algorithms and on different scenarios and improve the total time of convergence.

## Installing Dependencies

The instructions on how to download the dependencies for this project can be found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). Following the instructions on the `README.md` of the main folder should install all the necessary versions of the dependencies to run the projects in the nanodegree. This would include PyTorch, ML-Agents toolkit, and additional Python packages.

>Note: From the main repository, there is a note regarding the compatibility of the ML-Agents toolkit on Windows. Windows 10 is supported by ML-Agents. Other versions of Windows might be able to run the ML-Agents toolkit but it has not been tested so there is no guarantee.

## Cloning the Repository

To get the unsolved version of this project notebook, you can clone the [Deep Reinforcement Learning repository](https://github.com/udacity/deep-reinforcement-learning) of Udacity. This project would be under the p2_continuous-control folder.

```
cd [your/desired/folder/desitnation]
git clone https://github.com/udacity/deep-reinforcement-learning
```

To clone this repository and run the submitted version:

```
cd [your/desired/folder/desitnation]
git clone https://github.com/iocfinc/DRLND-P2-ContinousControl.git
```

## Installing the Reacher Environmnet

This is for the Unity-ML agent environment installation of Reacher. Udacity has already built the environment for us so we just have to download the correct environment depending on our OS.

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

## Solution

For this project submission, the details for the solution would be noted in the `REPORT.md` file. If you would want to do the unsolved version of this same project, you can refer to the notes in the notebook provided in the DRLND repository.

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"