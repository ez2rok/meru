from typing import Union

from torch.nn.parallel import DistributedDataParallel

from meru.models import MERU, CLIPBaseline

def get_run_name(
    model: Union[MERU, CLIPBaseline, DistributedDataParallel],
    seperator: str = '-',
    ) -> str:
    
    if isinstance(model, DistributedDataParallel):
        model = model.module
        
    model_name = model.__class__.__name__.lower()
    if model_name == 'clipbaseline':
        model_name = 'clip'
        
    embd_dim = str(model.visual_proj.weight.shape[0]).zfill(4)
    
    param_count = sum(p.numel() for p in model.parameters())
    if param_count < 1e8:
        model_size = 'small'
    elif param_count < 2e8:
        model_size = 'base'
    else:
        model_size = 'large'
    
    run_name = f'{seperator}'.join([model_name, model_size, embd_dim])
    return run_name