# moving-obstacle-trajectory-prediction

Code for my dissertation on moving object trajectory prediction using PySINDy

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

<!--
Provide a brief introduction to your project here. Explain what the project does and its main purpose.
--> 
For autonomous robots or vehicles, safety and performance are critical factors for motion planning – which can be
loosely defined as how robots decide what motions to perform in order to safely travel from point A to B. The problem of two-dimensional motion 
planning with an arbitrary number of obstacles moving with bounded velocity is NP-hard (7). There are practical solutions that are effective when the robot and
environmental factors are exactly known, but in real-world settings where those assumptions are no longer valid,
safety is not guaranteed. 

When environments are dynamic and not known in advance, the robot would need to reason about uncertainty. 
This project forms part of the predictive approach. In the predictive approach, there are two steps:
1. Forecasting the dynamic object’s future motion
2. Incorporating it into a MPC scheme that implements an optimal avoidance strategy.

This project will focus on implementing a new approach to the first step, and evaluating its performance. The
main idea of the project is to use an extension of the sparse identification of the nonlinear dynamics (SINDy) algo-
rithm first developed in discovering governing equations from data by sparse identification of nonlinear dynamical
systems developed https://doi.org/10.48550/arXiv.1509.03580

The code forecasts the trajectory of a dynamic moving obstacle without any pre-existing knowledge of the
obstacle and its path. To my knowledge, prior to this project, there have been no existing solutions that utilise SINDy or an extension of
SINDy to predict trajectories based on high fidelity simulation data (such as those generated by GAZEBO). 

I use the PySINDy package that implements different extensions of SINDy available here: https://github.com/dynamicslab/pysindy

The contributions of this code will be to:
• Apply the weak formulation of SINDy to predict the future motion of unknown objects modelled using a high
fidelity trajectory simulator Gazebo, and evaluate its performance.
• Improve accuracy and robustness to sudden changes in motion by implementing an elastic moving window
that specifies how much of the trajectory data is relevant for fitting.

Here is an example of the code predicting the trajectory of a UAV that is taking off and moving in a circular motion: 

https://github.com/ah19190/moving-obstacle-trajectory-prediction/assets/100033072/4138299b-0ad4-4fd4-9f82-af27af57abaf




## Features

There are three key features implemented in my model: 
1. Model Selection: Choosing sparsification parameter that balances accuracy and model complexity
2. Elastic moving window: Sliding window that moves along with the data, so that model fitting uses the most
recent data. The window size decreases if prediction accuracy drops or increases if accuracy goes up.
3. Ensemble prediction: Using ensembling, multiple dynamical models are fitted to the window of data. All
of the ensemble models used to predict multiple possible trajectories for the object. By default, only the
median of the models is used as the prediction, but I have made available an option to show all the ensemble
predictions.



## Installation


Explain how to install and set up the project. Include any prerequisites and dependencies that need to be installed.

To install and run the program, first clone the repository to your local computer.
Install the required packages from the relevant sites using your preferred installation method.
1. PySINDy
2. Dill
3. h5py
4. numpy
5. ipython
6. scikit-learn
7. matplotlib
Ensure that you are using python 3.8 and above. 


## Running the code  

Within VSCode, go to src directory and run main file: 

```
python3 main.py
```

Or to run the projectile motion dataset instead:  
```
python3 main_projectile_motion.py
```



