from agent import Agent
from drawing_manager import DrawingManager

import pygame
pygame.init()

import numpy as np
import random


# Variables4
gridX, gridY = 20, 24
WAS_SCREEN_INIT = False

# Ratio WIDTH:HEIGHT should be approximately near gridX:gridY

# Screen Constants
WIDTH, HEIGHT = 600, 500
scaleX = HEIGHT // gridX
scaleY = WIDTH // gridY


# Rewards
INVALID_ACTION_REWARD   = -1
WINNING_REWARD          = 200

def position_to_int(pos: tuple[int, int]) -> int:
    x, y = pos
    return x * gridY + y
    
class GridWorldEnv:
    def __init__(self):
        # self.clock = pygame.time.Clock()
        self.nb_states = gridX * gridY
        self.nb_actions = 4

        # self.init_grid = np.array(
        #     [
        #         [0, 0, 0, 0, 1],
        #         [0, 1, 0, 1, 0],
        #         [0, 1, 0, 0, 0],
        #         [0, 0, 0, 1, 0]
        #     ]
        # )
        self.init_grid = self.generate_rand_grid()
       #  self.init_grid = np.array([[0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
       # [0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
       # [0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.],
       # [0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.],
       # [0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1.],
       # [0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
       # [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.]])
        self.grid = self.init_grid

        self.agent = Agent(gridX, gridY)

        self.actions_dict = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        self.drawing_manager = DrawingManager(cell_width=scaleY, cell_height=scaleX)

        self.reset()

    def reset(self) -> tuple[int, int]:
        self.grid = self.init_grid
        self.agent.position = (0, 0)
        self.agent.pos_history = []

        return position_to_int( self.agent.position )

    def step(self, action) -> tuple[int, int, bool]:
        # print(f"Before Step | X: {self.agent.position[0]}, y: {self.agent.position[1]}")
        # Handle Invalid actions
        if not self.is_action_valid(action):
            next_state = position_to_int( self.agent.position )
            reward = INVALID_ACTION_REWARD
            done = self.get_is_done()

            return next_state, reward, done

        self.agent.position = self.agent.get_pos_after_step(self.actions_dict[action])
        
        # Here, the agent's action makes him go to a path cell
        # Let's calculate the reward and done according to the current state
        next_state = position_to_int( self.agent.position )
        reward = self.get_reward()
        done = self.get_is_done()

        # print(f"After Step | X: {self.agent.position[0]}, y: {self.agent.position[1]}")

        # Add the current agent position to its position history 
        self.agent.add_curr_pos_to_history()

        return next_state, reward, done

    def render(self):
        ''' Render the self.grid into the screen, with the Goal, and the agent in its position '''

        # Screen
        global WAS_SCREEN_INIT
        if not WAS_SCREEN_INIT:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Grid World Game Played By AI")
            WAS_SCREEN_INIT = True

        # Event Loop
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.close()

        # Draw the self.grid paths and walls
        for x in range(gridX):
            for y in range(gridY):
                if (x, y) in self.agent.pos_history:
                    self.drawing_manager.draw(self.screen, "Walked Path" , (x, y))
                else:
                    img_str = "Path" if self.grid[x][y] == 0 else "Wall"
                    self.drawing_manager.draw(self.screen, img_str , (x, y))


        # Draw the Goal
        self.drawing_manager.draw(self.screen, "Goal" , (gridX-1, gridY-1))
        
        # Draw the agent
        self.drawing_manager.draw(self.screen, "Agent" , self.agent.position)

        pygame.display.flip()

            
    def close(self):
        global WAS_SCREEN_INIT
        if WAS_SCREEN_INIT:
            WAS_SCREEN_INIT = False
            pygame.quit()

    def get_reward(self) -> int:
        x, y = self.agent.position

        if x == gridX-1 and y == gridY-1:
            return WINNING_REWARD

        if self.grid[x][y] == 0:
            return 0

    def get_is_done(self) -> bool:
        x, y = self.agent.position

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
        grid[gridX-1][gridY-1] = 0

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

