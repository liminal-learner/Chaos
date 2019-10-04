# Chaos (Python)
A short demo and simulator of chaotic systems, including the chaotic love triangle game.

## Dependencies: 
pandas, matplotlib, scipy, seaborn, numpy, random

## Files:
The jupyternotebook is [here](https://nbviewer.jupyter.org/github/liminal-learner/Chaos/blob/master/notebooks/Chaos.ipynb). 

## Example Usage:

~~~ 
from Chaos.Simulator import Simulator; 
sim = Simulator(num_agents = 10, max_iterations = 500);
sim.run(plot_trajectories = True, plot_convergence = True); 
~~~

* With the defaults plot_trajectories = True, plot_convergence = True, you will see the agents jump (or step) around the room in order to attempt to make their triangles. The red crosses are the targets for one particular agent (also in red) to help you visualize their experience. At the end of the simulation, a convergence graph will appear that shows how the mean step size of all agents changed over the simulation. If step_size = None, the default, this will look like random noise (see 3.1 below). However, if step_size < 1, this should settle down over the course of the simulation.  

* Note: there appears to be a minor bug with plotting the red agent when step_size = None, the jumping version, though her targets still appear in the red crosses.

## 4'D' Data Science Framework: 
This is not a data science project YET (in the future I will use the Simulator to generate data for modelling), but here is a simple outline nonetheless:

# 1. Define the goal: 
A chaotic system is one in which very small changes in initial conditions lead to large differences in how the system evolves. The goal of this project is to demonstrate this behaviour using a simple game.
Each person/agent selects two other people in the room to track throughout the game. Without saying who, they attempt to create an equilateral triangle with these two people. 
Everyone starts at random positions in the room and the game progresses as people move to achieve their goal.

Ultimately, I want to know what kinds of features influence convergence (number of agents, shape of room, initial proximity and obstacles between targets and agent, constraints on speed, knowledge of target's trajectories, etc). Convergence is declared once the mean step size of all agents has dropped below a given threshold for 100 iterations.

I am also interested in whether I can get a machine learning system to learn who everyone is following. The simulator can generate data for training.

# 2. Discover:
* There is no exploratory data analysis for this project yet, only a simulator. The outputs of several runs of this simulator with different settings will be used as input for an EDA.


# 3. Develop:
## Version 1: "Blind" leaps
* step_size = None, the default
* "Blind" because the agent jumps directly to the point that would create an equilateral triangle with her targets, without knowing where the targets will jump in the next iteration
* 

## Version 2: "Blind" baby steps
* Set step_size = 0.2, the agent moves 20% of the distance to the desired vertex
* "Blind" because the agent takes a step toward the point that would create an equilateral triangle with his targets, without knowing anything about the trajectory of the targets 
* 

## Results:
* 



# 4. Deploy:


