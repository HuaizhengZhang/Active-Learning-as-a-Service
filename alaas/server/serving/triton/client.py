"""
Triton inference Client-side function
"""
import uuid
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
                request_id=str(uuid.uuid4()),
                outputs=outputs
            )

            outputs_data = response.as_numpy("OUTPUT0")
            return outputs_data


def triton_inference_func(source_data, batch_size, model_name, address):
    probs = []
    for batch in np.array_split(source_data, int(len(source_data) / batch_size)):
        for result in ServeClient().infer(batch, model_name, address=address):
            probs.append(result)
    return probs

