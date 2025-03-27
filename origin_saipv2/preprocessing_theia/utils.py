from typing import Any, List, Dict, Tuple
from numpy.typing import NDArray
import torch
import torch.nn as nn
from preprocessing_theia.models import (
    get_sapiens_feature,
    get_sapiens_model
)

def get_feature_outputs(
    model_name: str, model: nn.Module, processor: Any, batch_images: torch.Tensor, dtype: torch.dtype = torch.float16, device='cpu'):
    features: Dict[str, Dict[str, torch.Tensor]] = {model_name: {}}
    if "sapiens" in model_name:
        feature = get_sapiens_feature(model, processor, batch_images, device=device)
        features[model_name] = {
            "embedding": feature.detach().cpu().to(dtype).contiguous()
        }
    else:
        raise NotImplementedError(f"model {model_name} is not supported")

    return features
  
def get_model(model_name: str, device) -> Tuple[nn.Module, Any]:
    if "sapiens" in model_name:
        if "sapiens_0.3b_coco_best_coco_AP_epoch_98_512x384" in model_name:
            target_size=(512,384)
        elif "sapiens_0.3b_coco_best_coco_AP_796" in model_name:
            target_size=(1024,768)
        else:
            raise ValueError
        model, processor = get_sapiens_model(model_name, target_size=target_size, device=device)
    else:
        raise NotImplementedError(f"{model_name} is not implemented")
    return model, processor


def get_models(
    model_names: List[str], device) -> Tuple[Dict[str, nn.Module], Dict[str, Any]]:
    models: Dict[str, nn.Module] = {}
    processors: Dict[str, Any] = {}
    for model_name in model_names:
        model, processor = get_model(model_name, device)
        models[model_name.replace("/", "_")] = model
        processors[model_name.replace("/", "_")] = processor
    return models, processors