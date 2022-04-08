import logging
# import main


##  Helper doc - https://docs.python.org/3/howto/logging-cookbook.html
class Logger:
    def __init__(self):
        self.logger = "test.log"

    def get_logger(self, logger, time):
        # create logger with 'logger'
        logger = logging.getLogger(logger)
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('test.log')
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

# logger.info('creating an instance of auxiliary_module.Auxiliary')
# a = main.Scenari()
# logger.info('created an instance of auxiliary_module.Auxiliary')
# logger.info('calling auxiliary_module.Auxiliary.do_something')
# a.do_something()
# logger.info('finished auxiliary_module.Auxiliary.do_something')
# logger.info('calling auxiliary_module.some_function()')
# auxiliary_module.some_function()
# logger.info('done with auxiliary_module.some_function()')