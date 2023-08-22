class Agent:
    def __init__(self, gridX, gridY):
        self.position = (0, 0)

        self.gridX = gridX
        self.gridY = gridY

        self.pos_history = []

        # Actions: {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}

    def get_pos_after_step(self, actionStr: str):
        if actionStr == "UP":
            return self.position[0]-1, self.position[1]
        if actionStr == "DOWN":
            return self.position[0]+1, self.position[1]
        if actionStr == "LEFT":
            return self.position[0], self.position[1]-1
        if actionStr == "RIGHT":
            return self.position[0], self.position[1]+1

    def add_curr_pos_to_history(self):
        self.pos_history.append(self.position)
