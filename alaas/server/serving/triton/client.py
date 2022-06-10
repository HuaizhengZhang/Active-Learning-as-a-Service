"""
Triton inference Client-side function
"""
import numpy as np

import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


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


def triton_inference_func(source_data, batch_size, model_name, address):
    probs = []
    for batch in np.array_split(source_data, batch_size):
        for result in ServeClient().infer(batch, model_name, address=address):
            probs.append(result)
    return probs


if __name__ == '__main__':
    shape = (16, 3, 224, 224)
    np.random.seed(0)
    input_data = np.random.rand(*shape).astype(np.float32)

    # client = ServeClient()
    # outputs_0 = client.infer(input_data, 'resnet', '54.251.31.100:8900')

    # output_1 = client.infer(input_data, 'resnet', '54.251.31.100:8900')
    outputs_0 = triton_inference_func(input_data, 1, 'resnet', '54.251.31.100:8900')
    outputs_1 = triton_inference_func(input_data, 4, 'resnet', '54.251.31.100:8900')
    # print(np.shape(outputs_0))
    print(np.amax(outputs_0, axis=0).argsort()[:1], np.shape(outputs_0))
    print(np.amax(outputs_1, axis=0).argsort()[:1], np.shape(outputs_1))
    # print("\n")
    # print(np.argpartition(outputs_1, -4)[-4:])

    # input_data = np.array([[0, 1], [0, 2], 2, 3, 4, 5, 6, 7, 8, 9])
    # print(list(chunks(input_data, 4)))
