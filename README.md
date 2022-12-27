# ALaaS: Active Learning as a Service.

![PyPI](https://img.shields.io/pypi/v/alaas?color=green) [![Downloads](https://pepy.tech/badge/alaas)](https://pepy.tech/project/alaas) [![Testing](https://github.com/MLSysOps/alaas/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/MLSysOps/alaas/actions/workflows/main.yml) ![GitHub](https://img.shields.io/github/license/MLSysOps/alaas) ![Docker Pulls](https://img.shields.io/docker/pulls/huangyz0918/alaas)

![](./docs/images/logo.svg)

Active Learning as a Service (ALaaS) is a fast and scalable framework for automatically selecting a subset to be labeled
from a full dataset so to reduce labeling cost. It provides a out-of-the-box and standalone experience for users to quickly
utilize active learning.


ALaaS is featured for

- :hatching_chick: **Easy-to-use** With <10 lines of code to start the system to employ active learning.
- :rocket: **Fast** Use the stage-level parallellism to achieve over 10x speedup than under-optimized active learning process.
- :collision:    **Elastic** Scale up and down multiple active workers, depending on the number of GPU devices.

*The project is still under the active development. Welcome to join us!*

- [Demo on AWS](https://github.com/MLSysOps/Active-Learning-as-a-Service#demo-on-aws-coffee)
- [Installation](https://github.com/MLSysOps/Active-Learning-as-a-Service#installation-construction)
- [Quick Start](https://github.com/MLSysOps/Active-Learning-as-a-Service#quick-start-truck)
- [ALaaS Server Customization (for Advance users)](https://github.com/MLSysOps/Active-Learning-as-a-Service#alaas-server-customization-wrench)
- [Strategy Zoo](https://github.com/MLSysOps/Active-Learning-as-a-Service#strategy-zoo-art)
- [Citation](https://github.com/MLSysOps/Active-Learning-as-a-Service#citation)

## Demo on AWS :coffee:

**Free ALaaS demo on AWS (Support HTTP & gRPC)**

Use least confidence sampling with [ResNet-18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) 
to select images to be labeled for your tasks! 

We have deployed ALaaS on AWS for demonstration. Try it by yourself!

<table>
<tr>
<td> Call ALaaS with HTTP 🌐 </td>
<td> Call ALaaS with gRPC 🔐 </td>
</tr>
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
<td>

```python
# pip install alaas
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
</table>


Then you will see 3 data samples (the most informative) has been selected from all the 5 data points by ALaaS. 

## Installation :construction:

You can easily install the ALaaS by [PyPI](https://pypi.org/project/alaas/),

```bash
pip install alaas
```

The package of ALaaS contains both client and server parts. You can build an active data selection service on your own
servers or just apply the client to perform data selection.

:warning: For deep learning frameworks like [TensorFlow](https://www.tensorflow.org/) and [Pytorch](https://pytorch.org/), you may need to install manually since the version to meet your deployment can be different (as well as [transformers](https://pypi.org/project/transformers/) if you are running models from it).

You can also use [Docker](https://www.docker.com/) to run ALaaS: 

```bash
docker pull huangyz0918/alaas
```

and start a service by the following command:

```bash
docker run -it --rm -p 8081:8081 \
        --mount type=bind,source=<config path>,target=/server/config.yml,readonly huangyz0918/alaas:latest
```

## Quick Start :truck:

After the installation of ALaaS, you can easily start a local server, here is the simplest example that can be executed with only 2 lines of code. 

```python
from alaas.server import Server

Server.start()
```

The example code (by default) will start an image data selection (PyTorch ResNet-18 for image classification task) HTTP server in port `8081` for you. After this, you can try to get the selection results on your own image dataset, a client-side example is like


```bash
curl \
-X POST http://0.0.0.0:8081/post \
-H 'Content-Type: application/json' \
-d '{"data":[{"uri": "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png"},
            {"uri": "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane2.png"},
            {"uri": "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane3.png"},
            {"uri": "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png"},
            {"uri": "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane5.png"}], 
    "parameters": {"budget": 3},
    "execEndpoint":"/query"}'
```

You can also use `alaas.Client` to build the query request (for both `http` and `grpc` protos) like this,


```python
from alaas.client import Client

url_list = [
    'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png',
    'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane2.png',
    'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane3.png',
    'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png',
    'https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane5.png'
]
client = Client('http://0.0.0.0:8081')
print(client.query_by_uri(url_list, budget=3))
```

The output data is a subset uris/data in your input dataset, which indicates selected results for further data labeling.


## ALaaS Server Customization :wrench:

We support two different methods to start your server, 1. by input parameters 2. by YAML configuration


### Input Parameters

You can modify your server by setting different input parameters, 

```python
from alaas.server import Server

Server.start(proto='http',                      # the server proto, can be 'grpc', 'http' and 'https'.
    port=8081,                                  # the access port of your server.
    host='0.0.0.0',                             # the access IP address of your server.
    job_name='default_app',                     # the server name.
    model_hub='pytorch/vision:v0.10.0',         # the active learning model hub, the server will automatically download it for data selection.
    model_name='resnet18',                      # the active learning model name (should be available in your model hub).
    device='cpu',                               # the deploy location/device (can be something like 'cpu', 'cuda' or 'cuda:0'). 
    strategy='LeastConfidence',                 # the selection strategy (read the document to see what ALaaS supports).
    batch_size=1,                               # the batch size of data processing.
    replica=1,                                  # the number of workers to select/query data.
    tokenizer=None,                             # the tokenizer name (should be available in your model hub), only for NLP tasks.
    transformers_task=None                      # the NLP task name (for Hugging Face [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)), only for NLP tasks.
)
```

### YAML Configuration

You can also start the server by setting an input YAML configuration like this,

```python
from alaas import Server

# start the server by an input configuration file.
Server.start_by_config('path_to_your_configuration.yml')
```

Details about building a configuration for your deployment scenarios can be found [here](./docs/configuration.md).



## Strategy Zoo :art:

Currently we supported several active learning strategies shown in the following table,

|Type|Setting|Abbr|Strategy|Year|Reference|
|:--:|:--:|:--:|:--:|:--:|:--:|
|Random|Pool-base|RS|Random Sampling|-|-|
|Uncertainty|Pool|LC|Least Confidence Sampling|1994|[DD Lew et al.](https://arxiv.org/pdf/cmp-lg/9407020)|
|Uncertainty|Pool|MC|Margin Confidence Sampling|2001|[T Scheffer et al.](https://link.springer.com/chapter/10.1007/3-540-44816-0_31)|
|Uncertainty|Pool|RC|Ratio Confidence Sampling|2009|[B Settles et al.](https://research.cs.wisc.edu/techreports/2009/TR1648.pdf)|
|Uncertainty|Pool|VRC|Variation Ratios Sampling|1965|[EH Johnson et al.](https://academic.oup.com/sf/article-abstract/44/3/455/2228590?redirectedFrom=fulltext)|
|Uncertainty|Pool|ES|Entropy Sampling|2009|[B Settles et al.](https://research.cs.wisc.edu/techreports/2009/TR1648.pdf)|
|Uncertainty|Pool|MSTD|Mean Standard Deviation|2016|[M Kampffmeyer et al.](https://ieeexplore.ieee.org/document/7789580)|
|Uncertainty|Pool|BALD|Bayesian Active Learning Disagreement|2017|[Y Gal et al.](https://arxiv.org/abs/1703.02910)|
|Clustering|Pool|KCG|K-Center Greedy Sampling|2017|[Ozan Sener et al.](https://www.semanticscholar.org/paper/A-Geometric-Approach-to-Active-Learning-for-Neural-Sener-Savarese/82fb7661d892a7412726de6ead14269139d0310c)|
|Clustering|Pool|KM|K-Means Sampling|2011|[Z Bodó et al.](http://proceedings.mlr.press/v16/bodo11a/bodo11a.pdf)|
|Clustering|Pool|CS|Core-Set Selection Approach|2018|[Ozan Sener et al.](https://arxiv.org/abs/1708.00489?context=cs)|
|Diversity|Pool|DBAL|Diverse Mini-batch Sampling|2019|[Fedor Zhdanov](https://arxiv.org/abs/1901.05954)|
|Adversarial|Pool|DFAL|DeepFool Active Learning|2018|[M Ducoffe et al.](https://arxiv.org/abs/1802.09841)|


## Citation

Our tech report of ALaaS is available on [arxiv](https://arxiv.org/abs/2207.09109) and [NeurIPS 2022](https://neurips-hill.github.io/). Please cite as:

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
    <td align="center"><a href="https://huaizhengzhang.github.io"><img src="https://avatars.githubusercontent.com/u/5894780?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Huaizheng</b></sub></a><br /><a href="#content-HuaizhengZhang" title="Content">🖋</a> <a href="https://github.com/MLSysOps/ALaaS/commits?author=HuaizhengZhang" title="Tests">⚠️</a> <a href="https://github.com/MLSysOps/ALaaS/commits?author=HuaizhengZhang" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/YuanmingLeee"><img src="https://avatars.githubusercontent.com/u/36268431?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yuanming Li</b></sub></a><br /><a href="https://github.com/MLSysOps/ALaaS/commits?author=YuanmingLeee" title="Tests">⚠️</a> <a href="https://github.com/MLSysOps/ALaaS/commits?author=YuanmingLeee" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Acknowledgement

- [Jina](https://github.com/jina-ai/jina) - Build cross-modal and multimodal applications on the cloud.
- [Transformers](https://github.com/huggingface/transformers) - State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.


## License

The theme is available as open source under the terms of the [Apache 2.0 License](./LICENSE).
