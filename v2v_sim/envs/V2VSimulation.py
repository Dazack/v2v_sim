# Master Script
from pandas.io.formats import style

from logger import Logger
import time
from vehicle import Vehicle
import json
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class V2VSimulationEnv(gym.Env):
    """
    ### Observations
    There are 2352 discrete states since there are 28 vehicle1 positions, 28 possible
    vehicle2 positions, and 3 possible routes.
    """

    def __init__(self, data):
        self.data = data
        logger = Logger()
        self.time_cnt = 0
        self.logger = logger.get_logger('test.log', self.time_cnt)
        self.end_sim_time = data['simulation']['sim_time']
        self.map = data['simulation']['map']
        self.reset_state = data['simulation']['reset_state']
        self.runtime = data['simulation']['runtime']
        self.routes = data['routes']
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
        self.num_states = data['simulation']['num_states'] # this number isn't to important for my simulation
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0]), np.array(self.num_states), dtype=np.float32)

    def _take_action(self, action):
        """ Define available actions"""

        if action == 0:
            status = 0  # okay
            crash_point = [0, 0]
            crashed_route = 0
            action_msg = "Stay on course"
        elif action == 1:
            status = 1  # Not okay
            crash_point = [0, 0]
            crashed_route = "route1"
            action_msg = "Divert"
        elif action == 2:
            status = 1  # Not okay
            crash_point = [0, 0]
            crashed_route = "route2"
            action_msg = "Divert"

        return action_msg, status, crash_point, crashed_route

    def _statemap(self, function="get", state=None):
        """Take a given input position of each vehicle postion and give a state"""
        if state is None:
            state = self.state
        list_valid_coordinates = []
        for row in range(len(self.map)):
            for column in range(len(self.map[0])):
                if self.map[row][column] == 1:
                    self.logger.debug("Valid position")
                    list_valid_coordinates.append([row, column])
        self.logger.debug("Valid Co-ordiate map: {}".format(list_valid_coordinates))

        if function == "get":
            index1 = list_valid_coordinates.index(self.vehicle1.position)
            index2 = list_valid_coordinates.index(self.vehicle2.position)
            self.logger.debug("Index1: {}".format(index1))
            self.logger.debug("Index2: {}".format(index2))
            route_list = [*self.routes]
            index3 = route_list.index(self.vehicle1.route_name)
            self.logger.debug("Index3: {}".format(index3))
            index4 = route_list.index(self.vehicle2.route_name)
            self.logger.debug("Index4: {}".format(index4))


            # first 28 ints give the position of vehicle 1
            # second 28 ints give position of vheilce 2
            # next 3 ints give route of vehicle 1
            # last 3 ints give route of vehicle 2
            self.state = np.array([index1, index2, index3, index4])
            self.logger.debug("Current State: {}".format(self.state))

        elif function == "set":
            self.logger.debug("state: {}".format(state))
            self.vehicle1.position = list_valid_coordinates[state[0]]
            self.vehicle2.position = list_valid_coordinates[state[1]]
            route_list = [*self.routes]
            self.vehicle1.route_name = route_list[state[2]]
            self.vehicle2.route_name = route_list[state[3]]
        else:
            print("Invalid mode for function: {}".format(function))

    def _get_reward(self):
        """ Based on current changes give a reward"""

        reward = 0

        if self.vehicle1.diff_start > self.vehicle1.diff_end:
            reward += 5
        elif self.vehicle1.diff_start < self.vehicle1.diff_end:
            reward -= 20
        else:
            reward -= -5

        if self.vehicle1.current_route != self.vehicle2.current_route:
            reward += 20
        else:
            reward -= -5

        if self.vehicle1.msx_tx['Msg_Status'] == 0:
            reward += 0
        elif self.vehicle1.msx_tx['Msg_Status'] == 1:
            reward -= 10

        return reward

    def step(self, action=0):

        start_time = time.time()
        vehicles = [self.vehicle1]

        action_msg, status, crash_point, crashed_route = self._take_action(action)


        self.vehicle1.main()
        self.vehicle2.main()

        snd_msg = self.vehicle1.send_msg(status, crash_point, crashed_route, action)
        self.vehicle2.receive_msg(snd_msg)

        remainder = self.runtime - (time.time() - start_time)
        if remainder > 0:
            self.logger.info("Sleeping for {}".format(remainder))
            time.sleep(remainder)

        self.goal_position = self.vehicle1.destination
        self.logger.info("Goal position {}".format(self.goal_position))
        done = bool(self.vehicle1.position == self.goal_position)# and self.data['vehicle1']['route'] != self.data['vehicle2']['route'])
        self.logger.info("Done: {}".format(done))
        reward = self._get_reward()

        self.state = (self.vehicle1.position, self.vehicle2.current_route)
        # return np.array(self.state, dtype=np.float32), reward, done, {}
        return np.array(self.state), reward, done, {}

    def reset(self):
        """ In case simulation space is called to reintialise state"""
        # self.logger.info("Reseting Vehicles and simulation map")
        self.logger.info("Reseting Vehicles")
        # self.end_sim_time = data['simulation']['sim_time']
        # self.map = data['simulation']['map']

        # Reset both Vehicles to their start points
        self._statemap(function="set", state=self.reset_state)
        self.vehicle1.status = "In Progress"
        self.vehicle2.status = "In Progress"
        # self.vehicle1 = Vehicle(self.logger, data['vehicle1'], data['simulation'], data['routes'], self.map)
        # self.vehicle2 = Vehicle(self.logger, data['vehicle2'], data['simulation'], data['routes'], self.map)

    def render(self, mode='human', close=False):
        #     # Render the environment to the screen
        out = self.map
        self.logger.info("Vehicle1 point {}".format(self.vehicle1.position))
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
# if __name__ == '__main__':
#     with open('./../../scenarios/scenario1.json') as json_file:
#         data = json.load(json_file)
#     test = V2VSimulationEnv(data)
#     test._statemap()
#     x = 0
#     while x <= data['simulation']['sim_time']:
#         s_new, rew, done, info = test.step(action=1)
#         print("State Number {}".format(s_new))
#         test.render()
#         x += 1





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


## Another reference working with Custom
# https://blog.paperspace.com/creating-custom-environments-openai-gym/

# This one helped alot with the initilisation sequence and correct format
# https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952


