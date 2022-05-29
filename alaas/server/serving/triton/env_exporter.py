from logging import Logger
from pathlib import Path

from conda_pack import CondaEnv


logger = Logger('Conda Env Exporter')


class CondaEnvExporter(object):
    def __init__(self, model_repository_path: Path):
        self.model_repository_path = model_repository_path

    def export(self, env_name: str, output_name: str = None, override=False):
        output_name = output_name or env_name
        output_path = (self.model_repository_path / output_name).with_suffix('.tar.gz')
        if output_path.exists():
            if override:
                logger.warning(f'Override the existing file {output_path}.')
            else:
                logger.warning(f'File {output_path} already exists, no action is performed.')
                return

        CondaEnv.from_name(env_name).pack(output=str(output_path))
        logger.info(f'Successfully export the conda environment {env_name} to {output_path}.')
