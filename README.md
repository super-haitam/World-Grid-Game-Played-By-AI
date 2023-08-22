# Description

	This project is a demonstration of Reinforcement Learning using Q learning on a relatively very simple grid games with paths and walls.
	The agent's objective is to find the optimal path to get to the goal.

# Mechanics

	Each time the agent makes a move towards a wall or outside the grid, the agent receives a reward of -1, as a punishment, which might seem low, but it encourages the agent not to abondon explaining and just go back and forth in the path cell of reward 0.

	Now, to favourise the behaviour of attaining the goal, which generates a reward of 200.
	
	Finally, in the training process, the most important behaviour is exploration, so I set up epsilon to be 0.7 as a minimum, so that it finds the optimal path quickly once it locate the goals position.

# Results

	The agent seems to be pretty good at finding the path to the goal ,like 99% of the time (to not say 100%).

	But the challenge resides in finding the goal since its position is not given, but always the bottom right cell.
	
	With the hyperparameters that vote for exploration, I'd say it suceeds about 70~80% of the time.