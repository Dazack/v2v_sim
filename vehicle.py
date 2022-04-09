import random
import time
import operator
from copy import deepcopy

class Vehicle:

    def __init__(self, logger, vehicle_data, sim_data, route_data, map):
        self.vehicle_id = vehicle_data['vehicle_id']
        self.vehicle_data = vehicle_data
        self.speed = vehicle_data['speed']
        # Row, Col
        self.position = vehicle_data['start']
        self.logger = logger
        self.veh_type = "default"
        self.route_name = vehicle_data['route']
        self.current_route = route_data[vehicle_data['route']]
        self.current_speed = [self.speed[0], self.speed[0], self.speed[1], self.speed[1]]
        self.destination = self.current_route["destination"]
        self.__route_map = deepcopy(route_data)
        self.map = map
        self.msx_rx = {
            "Timestamp": time.time(),
            "Vehicle_id": self.vehicle_id,
            "Vehicle_postion": self.position,
            "Route": self.current_route,
            "Msg_Status": 0,
            "Crash": [],
            "Broken_route": None,
            "Action": None
        }
        self.msx_tx = {
            "Timestamp": time.time(),
            "Vehicle_id": self.vehicle_id,
            "Vehicle_postion": self.position,
            "Route": self.current_route,
            "Msg_Status": 0,
            "Crash": [],
            "Broken_route": None,
            "Action": None
        }
        self.status = "In Progress"
        self.other_vehicle_position = [-1, -1]

        #DISABLELOGGER  self.logger.info("Intialising new vehicle {}".format(self.vehicle_id))

    def direction_sel(self, direction):

        if direction == "up":
            self.move_up()

        if direction == "down":
            self.move_down()

        if direction == "left":
            self.move_left()

        if direction == "right":
            self.move_right()

    def move_up(self):
        #DISABLELOGGER   self.logger.info("Moving {} Up".format(self.vehicle_id))
        self.position[0] -= self.current_speed[0]
        #DISABLELOGGER   self.logger.info("New position {}".format(self.position))

    def move_down(self):
        #DISABLELOGGER    self.logger.info("Moving {} Down".format(self.vehicle_id))
        self.position[0] += self.current_speed[1]
        #DISABLELOGGER   self.logger.info("New position {}".format(self.position))

    def move_right(self):
        #DISABLELOGGER self.logger.info("Moving {} Right".format(self.vehicle_id))
        self.position[1] += self.current_speed[2]
        #DISABLELOGGER   self.logger.info("New position {}".format(self.position))

    def move_left(self):
        #DISABLELOGGER  self.logger.info("Moving {} Left".format(self.vehicle_id))
        self.position[1] -= self.current_speed[3]
        #DISABLELOGGER  self.logger.info("New position {}".format(self.position))

    def find_biggest_diff(self, list, priority=""):

        #DISABLELOGGER    self.logger.debug("Finding biggest difference in list for {}: {}".format(self.vehicle_id, list))

        if abs(list[0]) > abs(list[1]) or priority.startswith("y") is True:
            #DISABLELOGGER    self.logger.debug("{} Should go up/down".format(self.vehicle_id))
            value = list[0]
            if value == 0:
                p = "x no change"
                #DISABLELOGGER          self.logger.debug("{} No change - p {} ".format(self.vehicle_id, p))
            elif value < 0:
                p = "x down"
                #DISABLELOGGER      self.logger.debug("{} Go Down - p {}".format(self.vehicle_id, p))
            elif value > 0:
                p = "x up"
                #DISABLELOGGER       self.logger.debug("{} Go Up - p{} ".format(self.vehicle_id, p))
        elif abs(list[0]) < abs(list[1]) or priority.startswith("x") is True:
            #DISABLELOGGER        self.logger.debug("{} Should go left/right".format(self.vehicle_id))
            value = list[1]
            if value == 0:
                p = "y no change"
                #DISABLELOGGER       self.logger.debug("{} No change - p {}".format(self.vehicle_id, p))
            elif value > 0:
                p = "y right"
                #DISABLELOGGER      self.logger.debug("{} Go Right - p {}".format(self.vehicle_id, p))
            elif value < 0:
                p = "y left"
                #DISABLELOGGER       self.logger.debug("{} Go Left - p {}".format(self.vehicle_id, p))
        else:
            #DISABLELOGGER  self.logger.debug("{} Go any direction".format(self.vehicle_id))
            p = "any"

        return p


    def manouver(self):
        """
        Handle vehicles traffic flow
        :return:
        """

        action = 0
        options = self.get_valid_road()
        #DISABLELOGGER     self.logger.info("{} Need to cover distance: {}".format(self.vehicle_id, self.diff_start))

        # Find the greatest difference between the 2 co-ordinates for the route, prioritise the larger gap
        p1 = self.find_biggest_diff(self.diff_start)
        p2 = self.find_biggest_diff(self.diff_start, p1)

        try:
            if self.current_route != []:
                #DISABLELOGGER   self.logger.debug("Preferred Route for {} is: {} and options are: {}".format(self.vehicle_id, self.current_route[str(self.position)], options))
                if self.current_route[str(self.position)] in options:
                    self.direction_sel(self.current_route[str(self.position)])
                    action += 1
                    #DISABLELOGGER         self.logger.debug("{} Following preferred_route, going {}".format(self.vehicle_id, self.current_route[str(self.position)]))
                else:
                    self.logger.debug("{} Unable to follow preferred_route. {}".format(self.vehicle_id, self.current_route[str(self.position)]))
        except Exception as gen_err:
            self.logger.debug("Hit a general exception running route. Error: {}".format(gen_err))

        if p1 == "any" and action == 0:
            self.direction_sel(random.choice(options))
            action += 1
        else:
            self.logger.debug("{} All action already used".format(self.vehicle_id))

        p1 = p1[2:]
        p2 = p2[2:]

        if p1 == "no change" and action == 0:
            self.logger.debug("{} No change - Next priority").format(self.vehicle_id)

        elif p1 in options and action == 0:
            #DISABLELOGGER   self.logger.info("{} Move {}".format(self.vehicle_id, p1))
            self.direction_sel(p1)
            action += 1
        else:
            self.logger.debug("{} All action already used".format(self.vehicle_id))

        if p2 in options and action == 0:
            #DISABLELOGGER    self.logger.info("{} Move {}".format(self.vehicle_id, p1))
            self.direction_sel(p2)
            action += 1
        else:
            self.logger.debug("{} All action already used".format(self.vehicle_id))

    def get_valid_road(self):

        options = []

        # print("Current Postion {}".format(self.map[self.position[0]][self.position[1]]))
        #DISABLELOGGER  self.logger.info("Len Map rows {}".format(len(self.map)))
        #DISABLELOGGER    self.logger.info("Len Map columns {}".format(len(self.map[0])))

        # print(f"Printing Vehicle ID: {self.vehicle_id}")
        # print(f"Printing Vehicle Route: {self.current_route}")
        # print(f"Printing Other Vehicle position: {self.other_vehicle_position}")
        # print(f"Printing Current Vehicle position: {self.position}")

        for i in range(self.speed[0]):
            # print(f"I for down {i}")
            try:
                if (self.position[0] + i) > len(self.map):
                    self.logger.info("{} Can't move down, at boarder".format(self.vehicle_id))
                    break
                elif (self.position[0] + i) == self.other_vehicle_position[0] and self.position[1] == self.other_vehicle_position[1]:
                    self.logger.info("{} Can't move down, Another vehicle is there".format(self.vehicle_id))
                    break
                elif self.map[self.position[0] + i][self.position[1]] == 1:
                    #DISABLELOGGER    self.logger.info("{} Down is valid".format(self.vehicle_id))
                    options.append("down")
                    self.current_speed[1] = i
            except IndexError as index_err:
                self.logger.debug("At boarder: {}".format(index_err))

        for i in range(self.speed[0]):
            # print(f"I for up {i}")
            try:
                if (self.position[0] - i) < 0:
                    self.logger.info("{} Can't move up, at boarder".format(self.vehicle_id))
                    break
                elif (self.position[0] - i) == self.other_vehicle_position[0] and self.position[1] == self.other_vehicle_position[1]:
                    self.logger.info("{} Can't move up, Another vehicle is there".format(self.vehicle_id))
                    break
                elif self.map[self.position[0] - i][self.position[1]] == 1:
                    #DISABLELOGGER      self.logger.info("{} Up is valid".format(self.vehicle_id))
                    options.append("up")
                    self.current_speed[0] = i
            except IndexError as index_err:
                self.logger.debug("At boarder: {}".format(index_err))

        for i in range(self.speed[1]):
            # print(f"I for right {i}")
            try:
                if (self.position[1] + i) > len(self.map[0]):
                    self.logger.info("{} Can't move right, at boarder".format(self.vehicle_id))
                    break
                elif (self.position[1] + i) == self.other_vehicle_position[1] and self.position[0] == self.other_vehicle_position[0]:
                    self.logger.info("{} Can't move right, Another vehicle is there".format(self.vehicle_id))
                    break
                elif self.map[self.position[0]][self.position[1] + i] == 1:
                    #DISABLELOGGER        self.logger.info("{} Right is valid".format(self.vehicle_id))
                    options.append("right")
                    self.current_speed[2] = i
            except IndexError as index_err:
                self.logger.debug("{} At boarder: {}".format(self.vehicle_id, index_err))

        for i in range(self.speed[1]):
            # print(f"I for left {i}")
            try:
                if (self.position[1] - i) < 0:
                    self.logger.info("{} Can't move left, at boarder".format(self.vehicle_id))
                    break
                elif (self.position[1] - i) == self.other_vehicle_position[1] and self.position[0] == self.other_vehicle_position[0]:
                    self.logger.info("{} Can't move left, Another vehicle is there".format(self.vehicle_id))
                    break
                elif self.map[self.position[0]][self.position[1] - i] == 1:
                    #DISABLELOGGER     self.logger.info("{} Left is valid".format(self.vehicle_id))
                    options.append("left")
                    self.current_speed[3] = i
            except IndexError as index_err:
                self.logger.debug("{} At boarder: {}".format(self.vehicle_id, index_err))

        return options

    def send_msg(self, status, crash_point, crashed_route, action):
        """
        Send message data.
        """
        # [0, self.vehicle_id, self.position, self.preferred_route, self.message]

        #DISABLELOGGER   self.logger.info("Preparing message")
        message_data = [status, crash_point, crashed_route, action]
        self.msx_tx = {
            "Timestamp": time.time(),
            "Vehicle_id": self.vehicle_id,
            "Vehicle_postion": self.position,
            "Route": self.current_route,
            "Msg_Status": message_data[0],
            "Crash": message_data[1],
            "Broken_route": message_data[2],
            "Action": message_data[3]
        }

        return self.msx_tx

    def receive_msg(self, msx_rx):
        """
        Receive message data.
        """
        # [0, self.vehicle_id, self.position, self.preferred_route, self.message]

        #DISABLELOGGER  self.logger.info(" {} Received message: {}".format(self.vehicle_id, msx_rx))
        if msx_rx["Msg_Status"] == 1:
            #DISABLELOGGER    self.logger.info("Message and crash data")

            try:
                if self.current_route == self.__route_map[msx_rx["Broken_route"]]:
                    #DISABLELOGGER self.logger.debug("Need to take alternative route to {}".format(msx_rx["Broken_route"]))
                    if len(self.__route_map) >= 2:
                        # print(f"Route map before {self.__route_map}")
                        del self.__route_map[msx_rx["Broken_route"]]
                        # print(f"Route map after {self.__route_map}")
                        good_keys = list(self.__route_map.keys())
                        self.route_name = random.choice(good_keys)
                        #DISABLELOGGER          self.logger.info("New route selected {}".format(self.route_name))
                        self.current_route = self.__route_map[self.route_name]
                        self.destination = self.current_route["destination"]

                else:
                    self.logger.info("Crash does not effect my immediate route")
            except Exception as no_key:
                self.logger.info("Key is not present {}".format(no_key))

        self.other_vehicle_position = msx_rx["Vehicle_postion"]


    def main(self):

        # self.time = time

        if self.status == "Done":
            #DISABLELOGGER    self.logger.info("{} is finished, removed from simulation".format(self.vehicle_id))
            self.position = [4, 0]
        elif self.position != self.destination:
            self.diff_start = list(map(operator.sub, self.destination, self.position))
            self.manouver()
            self.diff_end = list(map(operator.sub, self.destination, self.position))
        else:
            self.status = "Done"
            #DISABLELOGGER   self.logger.info("{} Destination Reached".format(self.vehicle_id))

