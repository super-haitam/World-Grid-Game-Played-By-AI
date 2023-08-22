import pygame

class DrawingManager:
    def __init__(self, cell_width, cell_height):
        cell_dim = cell_width, cell_height
        self.cell_width = cell_width
        self.cell_height = cell_height

        # Load the images
        path_img    = pygame.transform.scale( pygame.image.load("./Assets/path.png"), cell_dim)
        wall_img    = pygame.transform.scale( pygame.image.load("./Assets/wall.png"), cell_dim)
        agent_img   = pygame.transform.scale( pygame.image.load("./Assets/agent.png"), cell_dim)
        goal_img    = pygame.transform.scale( pygame.image.load("./Assets/goal.png"), cell_dim)

        self.images_dict = {
            "Path": path_img,
            "Wall": wall_img,
            "Agent": agent_img,
            "Goal": goal_img
        }

    def draw(self, screen, img_name, pos):
        pos = pos[0] * self.cell_height, pos[1] * self.cell_width
        inverted_pos = pos[1], pos[0]
        screen.blit(self.images_dict[img_name], inverted_pos)