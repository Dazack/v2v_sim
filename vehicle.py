import random
import time
import operator

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
        self.destination = self.current_route["destination"]
        self.route_map = route_data
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

        self.logger.info("Intialising new vehicle")

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
        self.logger.info("Moving Vehicle Up")
        self.position[0] -= self.speed[0]
        self.logger.info("New position {}".format(self.position))

    def move_down(self):
        self.logger.info("Moving Vehicle Down")
        self.position[0] += self.speed[0]
        self.logger.info("New position {}".format(self.position))

    def move_right(self):
        self.logger.info("Moving Vehicle Right")
        self.position[1] += self.speed[1]
        self.logger.info("New position {}".format(self.position))

    def move_left(self):
        self.logger.info("Moving Vehicle Left")
        self.position[1] -= self.speed[1]
        self.logger.info("New position {}".format(self.position))

    def find_biggest_diff(self, list, priority=""):

        self.logger.debug("Finding biggest difference in list: ".format(list))

        if abs(list[0]) > abs(list[1]) or priority.startswith("y") is True:
            self.logger.debug("Should go up/down")
            value = list[0]
            if value == 0:
                p = "x no change"
                self.logger.debug("No change - p".format(p))
            elif value > 0:
                p = "x down"
                self.logger.debug("Go Down - p".format(p))
            elif value < 0:
                p = "x up"
                self.logger.debug("Go Up - p".format(p))
        elif abs(list[0]) < abs(list[1]) or priority.startswith("x") is True:
            self.logger.debug("Should go left/right")
            value = list[1]
            if value == 0:
                p = "y no change"
                self.logger.debug("No change - p".format(p))
            elif value > 0:
                p = "y right"
                self.logger.debug("Go Right - p".format(p))
            elif value < 0:
                p = "y left"
                self.logger.debug("Go Left - p".format(p))
        else:
            self.logger.debug("Go any direction")
            p = "any"

        return p


    def manouver(self):
        """
        Handle vehicles traffic flow
        :return:
        """

        action = 0
        options = self.get_valid_road()
        self.logger.info("Need to cover distance: {}".format(self.diff_start))

        # Find the greatest difference between the 2 co-ordinates for the route, prioritise the larger gap
        p1 = self.find_biggest_diff(self.diff_start)
        p2 = self.find_biggest_diff(self.diff_start, p1)

        try:
            if self.current_route != []:
                self.logger.debug("Preferred Route is: {} and options are: {}".format(self.current_route[str(self.position)], options))
                if self.current_route[str(self.position)] in options:
                    self.direction_sel(self.current_route[str(self.position)])
                    action += 1
                    self.logger.debug("Following preferred_route, going {}".format(self.current_route[str(self.position)]))
                else:
                    self.logger.debug("Unable to follow preferred_route. {}".format(self.current_route[str(self.position)]))
        except Exception as gen_err:
            self.logger.debug("Hit a general exception running route. Error: {}".format(gen_err))

        if p1 == "any" and action == 0:
            self.direction_sel(random.choice(options))
            action += 1
        else:
            self.logger.debug("All action already used")

        p1 = p1[2:]
        p2 = p2[2:]

        if p1 == "no change" and action == 0:
            self.logger.debug("No change - Next priority")

        elif p1 in options and action == 0:
            self.logger.info("Move {}".format(p1))
            self.direction_sel(p1)
            action += 1
        else:
            self.logger.debug("All action already used")

        if p2 in options and action == 0:
            self.logger.info("Move {}".format(p1))
            self.direction_sel(p2)
            action += 1
        else:
            self.logger.debug("All action already used")

    def get_valid_road(self):

        options = []

        # print("Current Postion {}".format(self.map[self.position[0]][self.position[1]]))
        self.logger.info("Len Map rows {}".format(len(self.map)))
        self.logger.info("Len Map columns {}".format(len(self.map[0])))

        try:
            if (self.position[0] + self.speed[0]) > len(self.map):
                self.logger.info("Can't move down, at boarder")
            elif self.map[self.position[0] + self.speed[0]][self.position[1]] == 1:
                self.logger.info("Down is valid")
                options.append("down")
        except IndexError as index_err:
            self.logger.debug("At boarder: {}".format(index_err))
        try:
            if (self.position[0] - self.speed[0]) < 0:
                self.logger.info("Can't move up, at boarder")
            elif self.map[self.position[0] - self.speed[0]][self.position[1]] == 1:
                self.logger.info("Up is valid")
                options.append("up")
        except IndexError as index_err:
            self.logger.debug("At boarder: {}".format(index_err))
        try:
            if (self.position[1] + self.speed[1]) > len(self.map[0]):
                self.logger.info("Can't move right, at boarder")
                self.logger.debug("Right would be co-ordiante {}, {}".format(self.position[0],self.position[1] + self.speed[1]))
            elif self.map[self.position[0]][self.position[1] + self.speed[1]] == 1:
                self.logger.info("Right is valid")
                options.append("right")
        except IndexError as index_err:
            self.logger.debug("At boarder: {}".format(index_err))
        try:
            if (self.position[1] - self.speed[1]) < 0:
                self.logger.info("Can't move left, at boarder")
            elif self.map[self.position[0]][self.position[1] - self.speed[1]] == 1:
                self.logger.info("Left is valid")
                options.append("left")
        except IndexError as index_err:
            self.logger.debug("At boarder: {}".format(index_err))

        return options

    def send_msg(self, status, crash_point, crashed_route, action):
        """
        Send message data.
        """
        # [0, self.vehicle_id, self.position, self.preferred_route, self.message]

        self.logger.info("Preparing message")
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

        self.logger.info(" {} Received message: {}".format(self.vehicle_id, msx_rx))
        if msx_rx["Msg_Status"] == 1:
            self.logger.info("Message and crash data")

            try:
                if self.current_route == self.route_map[msx_rx["Broken_route"]]:
                    self.logger.debug("Need to take alternative route to {}".format(msx_rx["Broken_route"]))
                    if len(self.route_map) >= 2:
                        self.route_map.pop(msx_rx["Broken_route"])
                        good_keys = list(self.route_map.keys())
                        self.route_name = random.choice(good_keys)
                        self.logger.info("New route selected {}".format(self.route_name))
                        self.current_route = self.route_map[self.route_name]
                        self.destination = self.current_route["destination"]

                else:
                    self.logger.info("Crash does not effect my immediate route")
            except Exception as no_key:
                self.logger.info("Key is not present {}".format(no_key))


    def main(self):

        # self.time = time

        if self.position != self.destination:
            self.diff_start = list(map(operator.sub, self.destination, self.position))
            self.manouver()
            self.diff_end = list(map(operator.sub, self.destination, self.position))
        else:
            self.status = "Done"
            self.logger.info("Destination Reached")

