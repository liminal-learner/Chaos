# Chaos (Python)
A short demo and simulator of chaotic systems, including the chaotic love triangle game.

## Dependencies: 
pandas, matplotlib, scipy, seaborn, numpy

## Files:
The jupyternotebook is [here](https://nbviewer.jupyter.org/github/liminal-learner/Chaos/blob/master/Chaos.ipynb). 

## 4'D' Data Science Framework: 
This is not a data science project YET (but I will use the simulator in the future to generate data for modeling), but here is a simple outline nonetheless:

# 1. Define the goal: 
A chaotic system is one in which very small changes in initial conditions lead to large differences in how the system evolves. The goal of this project is to demonstrate this behaviour using a simple game.
Each person/agent selects two other people in the room to track throughout the game. Without saying who, they attempt to create an equilateral triangle with these two people. 
Everyone starts at random positions in the room and the game progresses as people move to achieve their goal.

Ultimately, I want to know what kinds of features influence convergence (number of agents, shape of room, initial proximity and obstacles between targets and agent, constraints on speed, knowledge of target's trajectories, etc). Convergence is declared once the mean step size of all agents has dropped below a given threshold for 100 iterations.

# 2. Discover:
* There is no exploratory data analysis for this project yet, only a simulator. The outputs of several runs of this simulator with different settings will be used as input for an EDA.


# 3. Develop:
## Version 1: "Blind" leaps
* "Blind" because the agent jumps directly to the point that would create an equilateral triangle with her targets, without knowing where the targets will jump in the next iteration
* 

## Version 2: "Blind" baby steps
* "Blind" because the agent takes a step toward the point that would create an equilateral triangle with his targets, without considering anything about the trajectory of the targets 
* 

## Results:
* 



# 4. Deploy:


