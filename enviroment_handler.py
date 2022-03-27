# Setting Up GUI
# Helper doc - https://betterprogramming.pub/making-grids-in-python-7cf62c95f413

# Created 27/02/2022
# Author: Phillip Garrad


import sys
import pygame
from pygame.locals import KEYDOWN, K_q


class simGUI:
    def __init__(self):
        # Screen Metrics:
        self.map = [[0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0],
               [1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0]]
        self.width = 600
        self.height = 400
        self.screensize = self.width, self.height
        self.black = (0, 0, 0)
        self.grey = (160, 160, 160)
        # GLOBAL VARS, Using a Dictionary.
        self.vars = {'surf': False}


        # This is the main game loop, it constantly runs until you press the Q KEY
        # or close the window.
        # CAUTION: THis will run as fast as you computer allows,
        # if you need to set a specific FPS look at tick methods.


    def main(self):
        pygame.init()  # Initial Setup
        self.width = 100 * len(self.map[0])
        self.height = 100 * len(self.map)
        self.screensize = self.width, self.height

        self.vars['surf'] = pygame.display.set_mode(self.screensize)
        # The loop proper, things inside this loop will
        # be called over and over until you exit the window

        while True:
            self.checkEvents()
            self.vars['surf'].fill(self.grey)
            # self.drawLine()
            self.drawGrid(9, self.map)
            self.drawRect()
            pygame.display.update()


    def drawLine(self):
        # draw a diagonal line from top left coordinates 0,0
        # to bottom right with coordinates 600 (Width), 400 (Height)
        pygame.draw.line(self.vars['surf'], self.black, (0, 0), (self.width, self.height), 2)


    # Draw filled rectangle at coordinates x,y 18,18 with size width,height 20,20
    def drawRect(self):
        pygame.draw.rect(self.vars['surf'], self.black, (18, 18, 20, 20))

    def drawGrid(self, divisions, map):

        # Printing Map Dimensions
        # print(map)
        # print("Lenght {}".format(len(map[0])))
        # print("Heigth {}".format(len(map)))

        CONTAINER_WIDTH_HEIGHT = 300  # Not to be confused with SCREENSIZE
        cont_x, cont_y = 10, 10  # TOP LEFT OF CONTAINER

        # for row in map:
        #
        #     for index in row:

        # DRAW Grid Border:
        # TOP lEFT TO RIGHT
        pygame.draw.line(
            self.vars['surf'], self.black,
            (cont_x, cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y), 2)
        # # BOTTOM lEFT TO RIGHT
        pygame.draw.line(
            self.vars['surf'], self.black,
            (cont_x, CONTAINER_WIDTH_HEIGHT + cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x, CONTAINER_WIDTH_HEIGHT + cont_y), 2)
        # # LEFT TOP TO BOTTOM
        pygame.draw.line(
            self.vars['surf'], self.black,
            (cont_x, cont_y),
            (cont_x, cont_y + CONTAINER_WIDTH_HEIGHT), 2)
        # # RIGHT TOP TO BOTTOM
        pygame.draw.line(
            self.vars['surf'], self.black,
            (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x, CONTAINER_WIDTH_HEIGHT + cont_y), 2)

        # Get cell size, just one since its a square grid.
        cellSize = CONTAINER_WIDTH_HEIGHT / divisions

        # VERTICAL DIVISIONS: (0,1,2) for grid(3) for example
        for x in range(divisions):
            pygame.draw.line(
                self.vars['surf'], self.black,
                (cont_x + (cellSize * x), cont_y),
                (cont_x + (cellSize * x), CONTAINER_WIDTH_HEIGHT + cont_y), 2)
            # # HORIZONTAl DIVISIONS
            pygame.draw.line(
                self.vars['surf'], self.black,
                (cont_x, cont_y + (cellSize * x)),
                (cont_x + CONTAINER_WIDTH_HEIGHT, cont_y + (cellSize * x)), 2)

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_q:
                pygame.quit()
                sys.exit()


if __name__ == '__main__':
    gui1 = simGUI()
    gui1.main()