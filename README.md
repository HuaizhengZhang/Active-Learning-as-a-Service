# ALaaS: Active learning as a service.

## Installation

You can easily install the AlaaS by `pip`,

```bash
pip install alaas
```

## Quick Start

### Define the global configuration

We use a global configuration file to control the whole system, an example looks like:

```yaml
name: "IMG_CLASSIFICATION"
version: 0.1
active_learning:
  budget: 10
  strategy:
    algorithm: "LeastConfidence"
    infer_model:
      model_hub: "huggingface"
      name: "resnet"
      batch_size: 4
      input_dtype: "float32"
  al_server:
    address: "54.251.31.100:8900"
    gpus: 'all'
```

The `infer_model` is designed for inference service inside some active learning strategies, by setting the `al_server`,
we can help you automatically deploy the inference model from the model repository.

### Start the active learning server

You need to start an active learning service before conducting the data selection.

```python
from alaas import Server

al_server = Server(config_path='./you_config.yml')
al_server.start()
```

### Define the active learning client and perform querying

You can easily start the data selection by the following code,

```python 
from alaas import Client 

al_client = Client(server_url="127.0.0.1:8888")
al_client.push(data_list, asynchronous=True)
al_client.query(budget=100)
```

### Example output

the example output will be something like:

```bash
preparing data...
Files already downloaded and verified
start active learning, query number: 1000...
[uri_1, uri_2, uri_3, ...., uri_n]
```

which are the uri of input data selected by active learning.