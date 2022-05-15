from service.server.util import ConfigManager

config_path = '/Users/huangyz0918/desktop/zeef/service/example/resnet_triton.yml'

if __name__ == '__main__':
    cfg_manager = ConfigManager(config_path)

    print(cfg_manager.get_job_name(), '\n')
    print(cfg_manager.get_job_version(), '\n')
    print(cfg_manager.get_al_config(), '\n')
