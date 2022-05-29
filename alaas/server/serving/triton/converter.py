"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: May 26, 2022
"""
from logging import Logger
from pathlib import Path
from google.protobuf import json_format, text_format
from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelParameter

from alaas.types.models.infer_model import TorchHubInferModelConfig

logger = Logger('Triton Python Model Converter')


class TritonPythonModelConverter(object):
    def __init__(self, model_repository_path: Path):
        from alaas.server.serving.triton import template

        self.model_repository_path = model_repository_path
        self.template_path = Path(template.__file__).absolute().parent

    def from_torch_hub(self, torch_hub_config: TorchHubInferModelConfig):
        model_path = self.model_repository_path / f'{torch_hub_config.name}'
        model_path.mkdir(exist_ok=True)
        # TODO: configurable version?
        model_version_dir = model_path / '1'
        model_version_dir.mkdir(exist_ok=True)

        # Build model config from template
        with open(self.template_path / 'config.pbtxt') as f:
            model_config_pb2 = text_format.Parse(f.read(), ModelConfig())

        torch_hub_config_obj = torch_hub_config.to_model_config_dict()
        model_config_pb2 = json_format.ParseDict(torch_hub_config_obj, model_config_pb2)
        # Add environment parameters
        # TODO: configurable execution env parameters
        model_parameter = ModelParameter(string_value="$$TRITON_MODEL_DIRECTORY/../my-pytorch.tar.gz")
        model_config_pb2.parameters['EXECUTION_ENV_PATH'].CopyFrom(model_parameter)
        model_config_msg_str = text_format.MessageToString(
                model_config_pb2,
                use_short_repeated_primitives=True,
        )
        with open(model_path / 'config.pbtxt', 'w') as f:
            f.write(model_config_msg_str)

        # Build Python Model from template
        with open(self.template_path / 'pytorch_hub_model.py') as f:
            pytorch_hub_model_py_template = f.read()
        pytorch_hub_model_py_str = pytorch_hub_model_py_template.format(
            hub_name=torch_hub_config.hub_name,  model=torch_hub_config.model,
            args=torch_hub_config.args, kwargs=torch_hub_config.kwargs,
        )
        with open(model_version_dir / 'model.py', 'w') as f:
            f.write(pytorch_hub_model_py_str)

        logger.info(f'Finish converting to {str(model_path)}')
