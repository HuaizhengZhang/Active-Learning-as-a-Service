from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from alaas.server.serving import TorchServe
from alaas.server.strategy import LeastConfidence, LeastConfidenceTriton
from alaas.server.util import ConfigManager


def prepare_data(batch_size):
    """
    Data pre-processing function for Resnet.
    @param batch_size: the input batch size for data loader (not for active learning).
    @return: torch data loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def triton_example(model_name, budget, batch_size, address='localhost:8900'):
    # print("preparing data...")
    inputs, targets = next(iter(prepare_data(100)))
    # print(f"start active learning, query number: {budget}...")
    strategy = LeastConfidenceTriton(source_data=inputs.numpy(), model_name=model_name, batch_size=batch_size,
                                     address=address)
    data_index_list = strategy.query(budget)
    print("selected data index: ", data_index_list)


def torch_example(budget, batch_size=10000):
    resnet_50_path = "/Users/huangyz0918/Downloads/torchscript/1.zip"
    server = TorchServe(resnet_50_path)

    for batch_idx, (inputs, targets) in enumerate(prepare_data(batch_size)):
        strategy = LeastConfidence(server.batch_predict, source_data=inputs)
        data_index_list = strategy.query(budget)
        print("selected data index: ", data_index_list)


if __name__ == '__main__':
    # read the global configuration.
    config_path = 'resnet_triton.yml'
    cfg_manager = ConfigManager(config_path)
    strategy = cfg_manager.get_al_config()['strategy']
    algorithm = strategy['algorithm']
    batch_size = strategy['infer_model']['batch_size']
    model_name = strategy['infer_model']['name']
    budget = cfg_manager.get_al_config()['budget']
    address = cfg_manager.get_al_config()['al_server']['address']

    # run the built-in pytorch server example.
    # torch_example(cfg_manager.get_al_config()['budget'])
    # run th NVIDIA triton inference server example.
    print("=====================")
    print(f"active learning: {algorithm}, budget: {budget}")
    print(f"AL model: {model_name}, inference batch: {batch_size}")
    print("=====================")
    triton_example(model_name=model_name, batch_size=batch_size, budget=budget, address=address)
