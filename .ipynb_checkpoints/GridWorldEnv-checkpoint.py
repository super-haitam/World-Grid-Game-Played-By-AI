from Agent import Agent

import numpy as np
import random

# import pygame
# pygame.init()

# Variables
gridX, gridY = 4, 5
# scale = 100

# Screen Constants
# WIDTH, HEIGHT = gridY * scale, gridX * scale

# Screen
# screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Rewards
INVALID_ACTION_REWARD   = -5
WINNING_REWARD          = 10

def position_to_int(pos: tuple[int, int]) -> int:
    x, y = pos
    return x * gridY + y
    
class GridWorldEnv:
    def __init__(self):
        # self.clock = pygame.time.Clock()
        self.nb_states = gridX * gridY
        self.nb_actions = 4

        self.init_grid = np.array(
            [
                [0, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0]
            ]
        )
        self.grid = self.init_grid

        self.agent = Agent(gridX, gridY)

        self.actions_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        self.reset()

    def reset(self) -> tuple[int, int]:
        self.grid = self.init_grid
        self.agent.position = (0, 0)

        return position_to_int( self.agent.position )

    def step(self, action) -> tuple[int, int, bool]:
        # print(f"Before Step | X: {self.agent.position[0]}, y: {self.agent.position[1]}")
        # Handle Invalid actions
        if not self.is_action_valid(action):
            next_state = position_to_int( self.agent.position )
            reward = INVALID_ACTION_REWARD
            done = True

            return next_state, reward, done

        self.agent.position = self.agent.get_pos_after_step(self.actions_dict[action])
        
        # Here, the agent's action makes him go to a path cell
        # Let's calculate the reward and done according to the current state
        next_state = position_to_int( self.agent.position )
        reward = self.get_reward()
        done = self.get_is_done()

        # print(f"After Step | X: {self.agent.position[0]}, y: {self.agent.position[1]}")

        return next_state, reward, done

    def render(self):
        # TODO: For Rendering
        pass

    def get_reward(self) -> int:
        x, y = self.agent.position

        if x == gridX-1 and y == gridY-1:
            return WINNING_REWARD

        if self.grid[x][y] == 0:
            return 0

    def get_is_done(self) -> bool:
        x, y = self.agent.position

        if self.grid[x][y] == 1:
            return True

        if x == gridX-1 and y == gridY-1:
            return True

        return False

    def is_action_valid(self, action) -> bool:
        actionStr = self.actions_dict[action]

        # print(f"Inside is_action_valid | X: {self.agent.position[0]}, y: {self.agent.position[1]}")

        # Handling Invalid Cases
        if actionStr == "UP" and self.agent.position[0] == 0:
            return False
        if actionStr == "DOWN" and self.agent.position[0] == gridX - 1:
            return False
        if actionStr == "LEFT" and self.agent.position[1] == 0:
            return False
        if actionStr == "RIGHT" and self.agent.position[1] == gridY - 1:
            return False

        # Handle Trying to move into a wall
        x, y = self.agent.get_pos_after_step(self.actions_dict[action])

        return self.grid[x][y] == 0

    def generate_rand_grid(self):
        grid = np.zeros((gridX, gridY))

        # 0: Rows, 1: Columns
        direction = random.choice([0, 1])

        # Generate a list of succession of 0 and 1, of length random either gridX or gridY
        binary_list = [ 0 if i%2==0 else 1 for i in range(gridX if direction==0 else gridY) ]

        if direction == 0:
            max_num_holes = max(1, gridX // 2)

            for index, value in enumerate(binary_list):
                # If value is 1, fill the row with ones but a random number of holes in the row
                if value == 1:
                    num_holes = random.randint(1, max_num_holes)
                    row = np.ones((gridY,))
                    available_indices = np.arange(gridY)

                    for i in range(num_holes):
                        rand_index = random.choice(available_indices)
                        row[rand_index] = 0
                        np.delete(available_indices, np.where(available_indices == rand_index)[0])

                    grid[index] = row

        elif direction == 1:
            max_num_holes = max(1, gridY // 2)

            for index, value in enumerate(binary_list):
                # If value is 1, fill the column with ones but a random number of holes in the column
                if value == 1:
                    num_holes = random.randint(1, max_num_holes)
                    column = np.ones((gridX,))
                    available_indices = np.arange(gridX)

                    for i in range(num_holes):
                        rand_index = random.choice(available_indices)
                        column[rand_index] = 0
                        np.delete(available_indices, np.where(available_indices == rand_index)[0])

                    grid[:, index] = column

        # Starting and Ending Cell are always 0
        grid[0][0] = 0
        grid[gridY-1][gridX-1] = 0

        return grid

# env = GridWorldEnv()

# g = env.grid.copy()
# g[env.agent.position] = 8
# print(g, "\n\nRight")
# print(env.step(3)) # RIGHT

# g = env.grid.copy()
# g[env.agent.position] = 8
# print(g, "\n\nRight")
# print(env.step(3)) # RIGHT

# g = env.grid.copy()
# g[env.agent.position] = 8
# print(g, "\n\nDown")
# print(env.step(1)) # DOWN

# g = env.grid.copy()
# g[env.agent.position] = 8
# print(g, "\n\nDown")
# print(env.step(1)) # DOWN

# g = env.grid.copy()
# g[env.agent.position] = 8
# print(g, "\n\nRight")
# print(env.step(3)) # RIGHT

# g = env.grid.copy()
# g[env.agent.position] = 8
# print(g, "\n\nRight")
# print(env.step(3)) # RIGHT

# g = env.grid.copy()
# g[env.agent.position] = 8
# print(g, "\n\nDown")
# print(env.step(1)) # DOWN

# # g = env.grid.copy()
# # g[env.agent.position] = 8
# # print(g, "\n\nRight")
# # print(env.step(3)) # RIGHT

# g = env.grid.copy()
# g[env.agent.position] = 8
# print(g)

