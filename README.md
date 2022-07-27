# ALaaS: Active Learning as a Service.

![PyPI](https://img.shields.io/pypi/v/alaas?color=green) [![Downloads](https://pepy.tech/badge/alaas)](https://pepy.tech/project/alaas) [![Testing](https://github.com/MLSysOps/alaas/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/MLSysOps/alaas/actions/workflows/main.yml) ![GitHub](https://img.shields.io/github/license/MLSysOps/alaas)

![](./docs/images/logo.svg)

Active Learning as a Service (ALaaS) is a fast and scalable framework for automatically selecting a subset to be labeled
from a full dataset so to reduce labeling cost. It provides a out-of-the-box and standalone experience for users to quickly
utilize active learning.


ALaaS is featured for

- :hatching_chick: **Easy-to-use** With <10 lines of code to start the system to employ active learning.
- :rocket: **Fast** Use the stage-level parallellism to achieve over 10x speedup than under-optimized active learning process.
- :collision:    **Elastic** Scale up and down multiple active workers, depending on the number of GPU devices.

*The project is still under the active development. Welcome to join us!*

## Try It Out :coffee:

**Free ALaaS demo on AWS**

Use least confidence sampling with [ResNet-18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) 
to select images to be labeled for your tasks! 

We have deployed ALaaS on AWS for demonstration. Try it by yourself!

<table>
<thead>
<tr>
<th>
call ALaaS with HTTP 🌐
</th>
</tr>
</thead>
<tbody>
<tr>
<td>
            
```bash
curl \
-X POST http://13.213.29.8:8081/post \
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
</tr>
<tr>
<th>
call ALaaS with gRPC 🔐
</th>
</tr>
<tr>
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
client = Client('grpc://13.213.29.8:60035')
print(client.query_by_uri(url_list, budget=3))
```    
            
</td>
</tr>
</tbody>
</table>

Then you will see 3 data samples (the most informative) has been selected from all the 5 data points by ALaaS. 

## Installation :construction:

You can easily install the ALaaS by [PyPI](https://pypi.org/project/alaas/),

```bash
pip install alaas
```

The package of ALaaS contains both client and server parts. You can build an active data selection service on your own
servers or just apply the client to perform data selection.

:warning: For deep learning frameworks like [TensorFlow](https://www.tensorflow.org/) and [Pytorch](https://pytorch.org/), you may need to install manually since the version to meet your deployment can be different.

## Step by Step

**0. Start the active learning server**

You need to start an active learning server before conducting the data selection.

```python
from alaas import Server

Server(config_path='./you_config.yml').start()
```

How to customize a configuration for your deployment scenarios can be found [here](./docs/configuration.md).

**1. Querying data from client**

You can easily start the data selection by the following code,

```python 
from alaas.client import Client

client = Client('http://0.0.0.0:60035')
queries = client.query_by_uri(<url_list>, budget=<budget number>)
```

The output data is a subset uris/data in your request, which means the selection results for further data labeling.

## Support Strategy :art:

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


## Citation

Our tech report is available on [arxiv](https://arxiv.org/abs/2207.09109). Please cite as:

```bash
@article{huang2022active,
  title={Active-Learning-as-a-Service: An Efficient MLOps System for Data-Centric AI},
  author={Huang, Yizheng and Zhang, Huaizheng and Li, Yuanming and Lau, Chiew Tong and You, Yang},
  journal={arXiv preprint arXiv:2207.09109},
  year={2022}
}
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://huangyz.name"><img src="https://avatars.githubusercontent.com/u/15646062?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yizheng Huang</b></sub></a><br /><a href="#infra-huangyz0918" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/MLSysOps/ALaaS/commits?author=huangyz0918" title="Tests">⚠️</a> <a href="https://github.com/MLSysOps/ALaaS/commits?author=huangyz0918" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/YuanmingLeee"><img src="https://avatars.githubusercontent.com/u/36268431?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yuanming Li</b></sub></a><br /><a href="https://github.com/MLSysOps/ALaaS/commits?author=YuanmingLeee" title="Tests">⚠️</a> <a href="https://github.com/MLSysOps/ALaaS/commits?author=YuanmingLeee" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Acknowledgement

- [Jina](https://github.com/jina-ai/jina) - Build cross-modal and multimodal applications on the cloud


## License

The theme is available as open source under the terms of the [Apache 2.0 License](./LICENSE).
