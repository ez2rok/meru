from typing import Union
from collections.abc import Sequence

import torch
from torch.nn.parallel import DistributedDataParallel

from meru.models import MERU, CLIPBaseline

def compare_models(
    model1: Union[DistributedDataParallel, MERU, CLIPBaseline],
    model2: Union[DistributedDataParallel, MERU, CLIPBaseline],
    exceptions: Sequence[str] = [
        'curv', 'visual_alpha', 'textual_alpha', 'logit_scale',
        'textual_proj.weight', 'visual_proj.weight',
        ]
    ) -> bool:
    """Compare two models."""
    
    
    def _compare(model1, model2):
        """
        Make sure all params in model1 are in model2.
        """
        
        for name, param1 in model1.named_parameters():
            if name in exceptions:
                continue
            param2 = model2.state_dict()[name]
            if not torch.allclose(param1, param2):
                print(f'{name} is different')
                return False
        return True

    model1 = model1.module if isinstance(model1, DistributedDataParallel) else model1
    model2 = model2.module if isinstance(model2, DistributedDataParallel) else model2
    return _compare(model1, model2) and _compare(model2, model1)