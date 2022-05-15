from alaas.server.util import ConfigManager

example_config_path = '/Users/huangyz0918/desktop/zeef/service/example/resnet_triton.yml'


class Server:
    def __int__(self, config_path):
        self.cfg_manager = ConfigManager(config_path)

    def start(self):
        pass
