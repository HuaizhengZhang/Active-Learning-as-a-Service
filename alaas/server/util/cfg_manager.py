"""
Configuration file manager for active learning as a service.
@author: huangyz0918 (huangyz0918@gmail.com)
@date: 01/05/2022
"""

import yaml


class ConfigManager:

    def __init__(self, config_path):
        self.config_dict = ConfigManager.load_config(config_path)

    @property
    def dry_run(self):
        return self.config_dict.get('dry_run', False)

    @staticmethod
    def load_config(config_path):
        result_dict = None
        with open(config_path, 'r') as stream:
            try:
                result_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return result_dict

    def get_job_name(self):
        return self.config_dict['name']

    def get_job_version(self):
        return self.config_dict['version']

    def get_al_config(self):
        return self.config_dict['active_learning']
