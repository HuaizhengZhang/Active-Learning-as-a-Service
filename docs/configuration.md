## How to define a server configuration

We use a global configuration file to control the whole system, an example looks like:

```yaml
name: "IMG_CLASSIFICATION"
version: 0.1
active_learning:
  strategy:
    type: "LeastConfidence"
    model:
      name: "resnet_cifar"
      hub: "pytorch/vision:release/0.12"
      model: "resnet18"
      batch_size: 1
      device: "cuda:0"
  al_worker:
    protocol: "http"
    host: "0.0.0.0"
    port: 60035
    replicas: 1
```

You can define the active learning settings in `active_learning`, which includes the `strategy` for the learning
algorithm and `al_worker` for worker deployments.

### Active Learning Strategy Settings

The active learning data query strategy is defined under the `active_learning.strategy`. It includes the strategy type,
and the learning model configurations. The `strategy.model` is designed for machine learning service inside some active
learning strategies, it will automatically download the pre-trained model for data selection from model
hub (`strategy.model.hub`) and according to the given model name (`strategy.model.name`). Current our ALaaS supports
models from both [Pytorch Hub](https://pytorch.org/hub/)
and [HuggingFace](https://huggingface.co/).

The built-in active learning strategies are list in the [table](https://github.com/MLSysOps/alaas#support-strategy), the
corresponding types inside our framework are,

|Strategy|Type|
|:--:|:--:|
|Random Sampling|`RandomSampling`|
|Least Confidence Sampling|`LeastConfidence`|
|Margin Confidence Sampling|`MarginConfidence`|
|Ratio Confidence Sampling|`RatioConfidence`|
|Variation Ratios Sampling|`VarRatioSampling`|
|Entropy Sampling|`EntropySampling`|
|Bayesian Active Learning Disagreement|`BayesianDisagreement`|
|Mean Standard Deviation|`MeanSTDSampling`|
|K-Center Greedy Sampling|`KCenterSampling`|
|K-Means Sampling|`KMeansSampling`|
|Core-Set Selection Approach|`CoreSet`|
|Diverse Mini-batch Sampling|`DiverseMiniBatch`|
|DeepFool Active Learning|`DeepFool`|

You should fill the configuration file with above types.

### Active Learning Worker Settings

To configure the active learning worker on the server, you should turn to `al_worker`, currently we support `grpc`
, `http` and `ws` (Websocket) three different protocols. And you can set the host address (`al_worker.host`) and data
I/O port (`al_worker.port`) as your wish. The `al_worker.replicas` means for this worker, how many instance should be
launched at the same time.
