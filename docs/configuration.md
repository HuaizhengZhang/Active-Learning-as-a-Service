## How to define a server configuration 

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
    address: "localhost:8900"
    gpus: 'all'
```

The `infer_model` is designed for inference service inside some active learning strategies, by setting the `al_server`,
we can help you automatically deploy the inference model from the model repository.