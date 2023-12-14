# The purpose of this file is to handle all vehicle to vehicle communications between the vehicle nodes.

import logging

class Comms:

    def __init__(self):
        self.receiver = None
        self.sender = None
        self.msg = ""
        self.logger = logging.getLogger('test.log')

    def send(self, msg):
        """ Sending message between vehicles"""
        self.logger.info("Message sent: {}".format(msg))
        self.logger.info("Sending too: {}".format(self.receiver))

        self.receiver
