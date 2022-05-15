import numpy as np

import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


class ServeEngine:
    """
    ServeEngine: Pytorch serving engine built-in.
    """

    def batch_predict(self, inputs, is_prob=False):
        pass


class ServeClient:
    """
    ServeClient: for inference service provided by Triton Inference Server (NVIDIA).
    """

    @staticmethod
    def infer(inputs_data, model_name, address="localhost:8900"):
        with httpclient.InferenceServerClient(address) as cli:
            inputs = [httpclient.InferInput("INPUT0", inputs_data.shape, np_to_triton_dtype(inputs_data.dtype))]
            inputs[0].set_data_from_numpy(inputs_data)

            outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

            response = cli.infer(
                model_name,
                inputs,
                request_id=str(1),
                outputs=outputs
            )

            outputs_data = response.as_numpy("OUTPUT0")
            return outputs_data


if __name__ == '__main__':
    shape = (4, 3, 224, 224)
    input_data = np.random.rand(*shape).astype(np.float32)
    client = ServeClient()
    outputs = client.infer(input_data, 'resnet')
    print(np.shape(outputs))
