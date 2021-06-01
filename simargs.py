"""Definition of the necessary parameters for the simulation of a mission execution."""
import argparse
from pathlib import Path


class SimArgs(object):
    """Class responsible for parsing the arguments."""

    def __init__(self):
        """Definition of the arguments."""
        self.parser = argparse.ArgumentParser(
            description='Mission execution simulation')

        self.parser.add_argument('--config_file',
                                 action='store',
                                 required=True,
                                 help='full path to the simulation config file.')

        self.parser.add_argument('--plan_file',
                                 action='store',
                                 required=True,
                                 help='full path to the file with the initial planning.')

        self.parser.add_argument('--result_file',
                                 action='store',
                                 required=True,
                                 help='full path to the result file to be created. \
                                        If the file exists, it will be overwritten.')

        self.parser.add_argument('--strategy',
                                 action='store',
                                 required=True,
                                 help='strategy used when an agent fails: [non-adapt, ours].')

        # # # # # # # # # # # # # #
        #  Parsing the arguments  #
        # # # # # # # # # # # # # #
        args = self.parser.parse_args()

        self.config_file = args.config_file
        if not Path(self.config_file).is_file():
            self.parser.error("{:s} is not a valid file.".format(self.config_file))

        self.plan_file = args.plan_file
        if not Path(self.plan_file).is_file():
            self.parser.error("{:s} is not a valid file.".format(self.plan_file))

        self.result_file = args.result_file
        self.strategy = args.strategy
        if self.strategy not in ['non-adapt', 'ours']:
            self.parser.error("Available strategies: [non-adapt, ours]")
