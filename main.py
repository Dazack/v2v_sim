# Master Script
from pandas.io.formats import style

from logger import Logger
import time
from vehicle import Vehicle
import json
import gym
from gym import spaces
import numpy as np





class Scenario:

    def __init__(self, data):
        logger = Logger()
        self.time_cnt = 0
        self.logger = logger.get_logger('test.log', self.time_cnt)
        self.end_sim_time = data['simulation']['sim_time']
        self.map = data['simulation']['map']
        self.vehicle1 = Vehicle(self.logger, data['vehicle1'], data['simulation'], data['routes'], self.map)
        self.vehicle2 = Vehicle(self.logger, data['vehicle2'], data['simulation'], data['routes'], self.map)
        self.colourmap = {
            "BLACK": '\033[30m',
            "RED": '\033[31m',
            "GREEN": '\033[32m',
            "YELLOW": '\033[33m',
            "BLUE": '\033[34m',
            "MAGENTA": '\033[35m',
            "CYAN": '\033[36m',
            "WHITE": '\033[37m',
            "UNDERLINE": '\033[4m',
            "RESET": '\033[0m'
        }
        HEIGHT = len(self.map)
        WIDTH = len(self.map[0])
        N_CHANNELS = [self.vehicle1, self.vehicle2]

        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=
        (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

    def step(self, action=0):

        start_time = time.time()
        vehicles = [self.vehicle1]

        if action == 0:
            status = 0 # okay
            crash_point = [0, 0]
            crashed_route = 0
            action = "Stay on course"
        elif action == 1:
            status = 1 # Not okay
            crash_point = [0, 0]
            crashed_route = "route1"
            action = "Divert"
        elif action == 2:
            status = 1 # Not okay
            crash_point = [0, 0]
            crashed_route = "route2"
            action = "Divert"



        self.vehicle1.main()
        self.vehicle2.main()

        for vehicle in vehicles:
            snd_msg = vehicle.send_msg(status, crash_point, crashed_route, action)
            self.vehicle2.receive_msg(snd_msg)


        remainder = 1 - (time.time() - start_time)
        if remainder > 0:
            self.logger.info("Sleeping for {}".format(remainder))
            time.sleep(remainder)


    def reset(self, data):
        """ In case simulation space is called to reintialise state"""
        self.logger.info("Reseting Vehicles and simulation map")
        self.end_sim_time = data['simulation']['sim_time']
        self.map = data['simulation']['map']
        self.vehicle1 = Vehicle(self.logger, data['vehicle1'], data['simulation'], data['routes'], self.map)
        self.vehicle2 = Vehicle(self.logger, data['vehicle2'], data['simulation'], data['routes'], self.map)

    def render(self, mode='human', close=False):
        #     # Render the environment to the screen
        out = self.map
        print("Vehicle1 point {}".format(self.vehicle1.position))
        print("Map:")
        for row in range(len(out)): #ud
            for column in range(len(out[row])): #lr
                # print([row, column])
                if [row, column] == self.vehicle1.position:
                    if self.vehicle1.status == "Done":
                        print("{}{}{}".format(self.colourmap["GREEN"], out[row][column], self.colourmap["RESET"]),
                              end=" ")
                    else:
                        print("{}{}{}".format(self.colourmap["RED"], out[row][column], self.colourmap["RESET"]),
                              end=" ")
                elif [row, column] == self.vehicle2.position:
                    if self.vehicle2.status == "Done":
                        print("{}{}{}".format(self.colourmap["YELLOW"], out[row][column], self.colourmap["RESET"]),
                              end=" ")
                    else:
                        print("{}{}{}".format(self.colourmap["BLUE"], out[row][column], self.colourmap["RESET"]),
                              end=" ")
                else:
                    print(out[row][column], end=" ")
            print("")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with open('./scenarios/scenario1.json') as json_file:
        data = json.load(json_file)
    test = Scenario(data)
    x = 0
    while x <= data['simulation']['sim_time']:
        test.step(action=1)
        test.render()
        x += 1





## FRom https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
# import gym
# from gym import spaces

# class CustomEnv(gym.Env):
#   """Custom Environment that follows gym interface"""
#   metadata = {'render.modes': ['human']}
#
#   def __init__(self, arg1, arg2, ...):
#     super(CustomEnv, self).__init__()
#     # Define action and observation space
#     # They must be gym.spaces objects
#     # Example when using discrete actions:
#     self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
#     # Example for using image as input:
#     self.observation_space = spaces.Box(low=0, high=255, shape=
#                     (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
#
#   def step(self, action):
#     # Execute one time step within the environment
#     ...
#   def reset(self):
#     # Reset the state of the environment to an initial state
#     ...
#   def render(self, mode='human', close=False):
#     # Render the environment to the screen
#     ...
