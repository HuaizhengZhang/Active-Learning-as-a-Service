# ALaaS: Active learning as a service.

## Installation

Using Pypi,

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
al_server

```


### Define the active learning client and perform querying

You can easily start the data selection by the following code, 

```python 
from alaas import Client 

al_client = Client(server_url="127.0.0.1:8888")
al_client.push(data_list, asynchronous=True)
al_client.query(budget=100)
```

### Start the active learning from example

```bash
cd zalaas/examples
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