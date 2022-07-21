"""
Configuration file manager for active learning as a service.
@author: huangyz0918 (huangyz0918@gmail.com)
@author: Li Yuanming
@date: 01/05/2022
"""

import yaml

from alaas.types.models.config import Config


class ConfigManager:

    def __init__(self, config_path):
        self.config: Config = ConfigManager.load_config(config_path)

    @staticmethod
    def load_config(config_path):
        with open(config_path) as f:
            config_obj = yaml.safe_load(f)
        return Config.parse_obj(config_obj)

    @property
    def name(self):
        return self.config.name

    @property
    def version(self):
        return self.config.version

    @property
    def strategy(self):
        return self.config.active_learning.strategy

    @property
    def al_worker(self):
        return self.config.active_learning.al_worker