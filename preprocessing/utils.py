from typing import Any, List, Dict, Tuple
from numpy.typing import NDArray
import torch
import torch.nn as nn
from preprocessing.models import (
    get_sapiens_feature,
    get_sapiens_model
)

def get_feature_outputs(
    model_name: str, model: nn.Module, batch_inputs: torch.Tensor, dtype: torch.dtype = torch.float16):
    if "sapiens" in model_name:
        features = get_sapiens_feature(model, batch_inputs, dtype=dtype)
    else:
        raise NotImplementedError(f"model {model_name} is not supported")

    return features
  
def get_model(model_name: str) -> Tuple[nn.Module, Any]:
    if "sapiens" in model_name:
        model, processor = get_sapiens_model(model_name)
    else:
        raise NotImplementedError(f"{model_name} is not implemented")
    return model, processor


def get_models(
    model_names: List[str]) -> Tuple[Dict[str, nn.Module], Dict[str, Any]]:
    models: Dict[str, nn.Module] = {}
    processors: Dict[str, Any] = {}
    for model_name in model_names:
        model, processor = get_model(model_name)
        models[model_name.replace("/", "_")] = model
        processors[model_name.replace("/", "_")] = processor
    return models, processors