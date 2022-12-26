# The ALaaS Docker Server Entry.

import argparse
from alaas.server import Server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input optional guidance for starting ALaaS.")
    parser.add_argument("--config", default="../examples/image/resnet18.yml", type=str, help="Please indicate the path of your configuration file.")
    Server.start_by_config(parser.parse_args().config)