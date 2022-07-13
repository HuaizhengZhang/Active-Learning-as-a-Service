# ALaaS: Active Learning as a Service.

![PyPI](https://img.shields.io/pypi/v/alaas?color=green) [![Downloads](https://pepy.tech/badge/alaas)](https://pepy.tech/project/alaas) [![Testing](https://github.com/MLSysOps/alaas/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/MLSysOps/alaas/actions/workflows/main.yml) ![GitHub](https://img.shields.io/github/license/MLSysOps/alaas)

![](./docs/images/logo.svg)

ALaaS is featured for

- :rocket: **Fast** Use the stage-level parallel method to achieve over 10x speedup than normal active learning process.
- :collision:	**Elastic** Scale up and down multiple active workers on single or multiple GPU devices.
- :hatching_chick: **Easy-to-use** With <10 lines of code to start APIs that prototype an active learning workflow.


## Installation

You can easily install the ALaaS by [PyPI](https://pypi.org/project/alaas/),

```bash
pip install alaas
```

## Quick Start


### Start the active learning server

You need to start an active learning server before conducting the data selection.

```python
from alaas import Server

al_server = Server(config_path='./you_config.yml')
al_server.start()
```

How to customize a configuration for your deployment scenarios can be found [here](./docs/configuration.md).

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


## License

The theme is available as open source under the terms of the [Apache 2.0 License](./LICENSE).
