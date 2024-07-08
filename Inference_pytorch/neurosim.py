import torch
import os
from torch import nn, Tensor
from .utee import hook
from typing import Tuple
# import neurosim_cpp # type: ignore
from typing import List
import importlib
# from .build.neurosim_cpp import PPA # type: ignore
from .build import neurosim_cpp


def write_model_network(model: nn.Module, model_name: str) -> str:
    assert os.path.exists(f"./layer_record_{model_name}"), f"Directory ./layer_record_{model_name} does not exist"

    network_file: str = f"./layer_record_{model_name}/NetWork_{model_name}.csv"
    with open(network_file, "w") as f:
        for _, layer in model.named_modules():
            if len(list(layer.children())) != 0:
                continue
            if isinstance(layer, nn.Linear):
                f.write(f"1,1,{layer.in_features},1,1,{layer.out_features},0,1\n")
            else:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

    return network_file

def neurosim_ppa(model_name:str, model: nn.Module, x_test: Tensor, ram_size: int, frequency: int, temperature: int, cell_bit: int, device: str) -> Tuple[float, float, float, float]:
    """
    Args:
        model_name: model name
        model: model
        test_loader: test data loader
        ram_size: ram size
        frequency: frequency
        temperature: temperature
        cell_bit: cell bit
    Returns:
        energy: uJ
        latency: us
        area: mm^2
        clock period: us
    """
    # for data, target in test_loader:
    data_file: List[str] = []

    hook_handle_list = hook.hardware_evaluation(model,8,8,ram_size, ram_size, model_name, "WAGE")
    with torch.no_grad():
        _ = model(x_test)
    data_file = hook.remove_hook_list(hook_handle_list)

    net_file: str = write_model_network(model, model_name)
    cell_type: int = 2

    importlib.reload(neurosim_cpp)
    return neurosim_cpp.PPA(net_file, cell_type, frequency, temperature, ram_size, cell_bit, data_file)

if __name__ == "__main__":
    from datasets import load_dataset
    from typing import Any, Dict
    data_args: Dict[str, Any] = {
        "dataset": "mnist",
        "train_batch_size": 2048,
        "test_batch_size": 16,
        "num_workers": 4,
        "train_ratio": 0.8,
    }
    train_loader, _, test_loader = load_dataset("mnist", data_args, True)
    from models import HDFactory
    # params
    hd_dim: int = 2048
    f1: int = 28
    d1: int = 64
    reram_size: int = 64
    frequency: int = int(1e9)
    binarize_type: bool = False
    kron: bool = False

    # construct hd
    hd_factory = HDFactory(
        28 * 28, hd_dim, 10, binarize_type, "cpu"
    )
    if kron:
        # pass
        hd_factory.set_kronecker(d1, f1)
    hd_factory.bernoulli()
    hd_factory.binarize(binarize_type)
    hd_factory.init_buffer(train_loader)

    model = hd_factory.create()
    print(neurosim_ppa("HD", model, next(iter(test_loader))[0], 64, int(1e9), 300, 1, "cuda:0"))