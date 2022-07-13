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
start active learning, query number: 100...
query results: [uri_1, uri_2, uri_3, ...., uri_n]
```

which are the a list of data samples selected by active learning.

## Support Strategy 

Currently we supported several active learning strategies shown in the following table,

|Type|Setting|Abbr|Strategy|Year|Reference|
|:--:|:--:|:--:|:--:|:--:|:--:|
|Random|Pool-base|RS|Random Sampling|-|-|
|Uncertainty|Pool-base|LC|Least Confidence Sampling|1994|[DD Lew et al.](https://arxiv.org/pdf/cmp-lg/9407020)|
|Uncertainty|Pool-base|MC|Margin Confidence Sampling|2001|[T Scheffer et al.](https://link.springer.com/chapter/10.1007/3-540-44816-0_31)|
|Uncertainty|Pool-base|RC|Ratio Confidence Sampling|2009|[B Settles et al.](https://research.cs.wisc.edu/techreports/2009/TR1648.pdf)|
|Uncertainty|Pool-base|ES|Entropy Sampling|2009|[B Settles et al.](https://research.cs.wisc.edu/techreports/2009/TR1648.pdf)|
|Uncertainty|Pool-base|BALD|Bayesian Active Learning Disagreement|2017|[Y Gal et al.](https://arxiv.org/abs/1703.02910)|
|Clustering|Pool-base|KCG|K-Center Greedy Sampling|2017|[Ozan Sener et al.](https://www.semanticscholar.org/paper/A-Geometric-Approach-to-Active-Learning-for-Neural-Sener-Savarese/82fb7661d892a7412726de6ead14269139d0310c)|
|Clustering|Pool-base|KM|K-Means Sampling|2011|[Z Bodó et al.](http://proceedings.mlr.press/v16/bodo11a/bodo11a.pdf)|
|Clustering|Pool-base|CS|Core-Set Selection Approach|2018|[Ozan Sener et al.](https://arxiv.org/abs/1708.00489?context=cs)|
|Diversity|Pool-base|DBAL|Diverse Mini-batch Sampling|2019|[Fedor Zhdanov](https://arxiv.org/abs/1901.05954)|
|Adversarial|Pool-base|DFAL|DeepFool Active Learning|2018|[M Ducoffe et al.](https://arxiv.org/abs/1802.09841)|

## License

The theme is available as open source under the terms of the [Apache 2.0 License](./LICENSE).
