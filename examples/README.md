# Example of ResNet

This is the example of using ALaaS to build a resnet image classification active learning service.

## Quick Start

### Define the global configuration

We use a global configuration file to control the whole system, an example looks like:

```yaml
name: "resnet_image_classification_03"
version: 0.1
active_learning:
  budget: 1000
  infer_model:
    name: "ResNet50"
    framework: "Pytorch"
    input_shape:
      - 224
      - 224
      - 3
    batch_size: 4
    input_dtype: "float32"
  strategy: "LeastConfidence"
```

The `infer_model` is designed for inference service inside some active learning strategies, if you are using the
built-in model inference server, please add an attribute `model_weights` which points to the path of your model weights.

### Start the 3-party inference service

if you want to use the 3-party inference services (e.g., TensorFlow-Serving, Triton Inference Server) to power up the
active learning, you can start the service by yourself and set up the configuration. We provide a script for you to
start the NVIDIA Triton Inference Server of resnet example.

```bash
cd zeef/service/scripts/
sh start_triton.sh
```

### Start the active learning

```bash
cd zeef/service/examples
python resnet50.py
```

the example output will be something like:

```bash
preparing data...
Files already downloaded and verified
start active learning, query number: 1000...
[ 295 7289 6148 9521 6351 8452 2825   97 5786 9244 3503 6778 5415 7982
 9757 2867  267 3224  236   43 3024 7506 8739 2335 2962 1556 2015 9260
 3444 1391 4772 3085 5501 3393 ...  5473 4524 8193]
```

which are the index of input data selected by active learning.