# ALaaS: Active Learning as a Service.

![PyPI](https://img.shields.io/pypi/v/alaas?color=green) [![Downloads](https://pepy.tech/badge/alaas)](https://pepy.tech/project/alaas) [![Testing](https://github.com/MLSysOps/alaas/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/MLSysOps/alaas/actions/workflows/main.yml) ![GitHub](https://img.shields.io/github/license/MLSysOps/alaas)

![](./docs/images/logo.svg)

Active Learning as a Service (ALaaS) is a fast and scalable service framework for users to conduct the data selection
before human labeling. It can be easily integrated with existing data processing and labeling platforms as a
microservice.

ALaaS is featured for

- :rocket: **Fast** Use the stage-level parallel to achieve over 10x speedup than normal active learning process.
- :collision:    **Elastic** Scale up and down multiple active workers on single or multiple GPU devices.
- :hatching_chick: **Easy-to-use** With <10 lines of code to start APIs that prototype an active learning workflow.

## Try It Out

You may just want to use the pre-trained model as the active data selector to help you select the most informative data
samples from the unlabeled data pool. We have a CPU-based server for data selection demonstration (least confidence
strategy with ResNet-18), try it by yourself!

<table>
<tr>
<td> <b> HTTP </b> </td>
<td> <b> gRPC </b> </td>
</tr>
<tr>
<td>

```bash
curl \
-X POST http://13.213.8.21:8081/post \
-H 'Content-Type: application/json' \
-d '{"data":[{"uri": "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png"},
            {"uri": "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane2.png"},
            {"uri": "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane3.png"},
            {"uri": "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png"},
            {"uri": "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane5.png"}], 
    "parameters": {"budget": 3},
    "execEndpoint":"/query"}'
```

</td>
<td>

```python
from alaas.client import Client

url_list = [
    'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png',
    'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane2.png',
    'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane3.png',
    'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png',
    'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane5.png'
]
client = Client('grpc://13.213.8.21:60035')
print(client.query_by_uri(url_list, budget=3))
```

</td>
</tr>
</table>

Then you will see 3 data samples has been selected from all the 5 data points by active learner. 

## Installation

You can easily install the ALaaS by [PyPI](https://pypi.org/project/alaas/),

```bash
pip install alaas
```

The package of ALaaS contains both client and server parts. You can build an active data selection service on your own
servers or just apply the client to perform data selection.

## Start the active learning server

You need to start an active learning server before conducting the data selection.

```python
from alaas import Server

Server(config_path='./you_config.yml').start()
```

How to customize a configuration for your deployment scenarios can be found [here](./docs/configuration.md).

## Querying data from client
You can easily start the data selection by the following code,

```python 
from alaas.client import Client

client = Client('http://0.0.0.0:60035')
queries = client.query_by_uri(<url_list>, budget=<budget number>)
```

The output data is a subset uris/data in your request, which means the selection results for further data labeling.

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
