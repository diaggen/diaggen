import importlib

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'ElasticSLatEncoder': 'structured_latent_vae',
    'ElasticSLatGaussianDecoder': 'structured_latent_vae',
    'ElasticSLatRadianceFieldDecoder': 'structured_latent_vae',
    'ElasticSLatMeshDecoder': 'structured_latent_vae',
    
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    'SLatFlowModelWOPE': 'structured_latent_flow_wo_partpe',
    'ElasticSLatFlowModelWOPE': 'structured_latent_flow_wo_partpe',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")
    
    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    
    # Load state dict
    state_dict = load_file(model_file)
    model_state_dict = model.state_dict()
    
    import torch
    # Handle missing keys with specific initialization methods
    missing_keys = [k for k in model_state_dict.keys() if k not in state_dict]
    if missing_keys:
        print(f"Initializing missing keys in state_dict ({len(missing_keys)} keys):")
        for k in missing_keys:
            print(f"shape of {k}: {model_state_dict[k].shape}")
            # Apply specific initializations based on parameter name
            if k == "layer_pe.weight":
                state_dict[k] = torch.normal(mean=0.0, std=0.02, size=model_state_dict[k].shape,
                                            device=model_state_dict[k].device, 
                                            dtype=model_state_dict[k].dtype)
                print(f"  - {k}: initialized with normal distribution (mean=0.0, std=0.02)")
            elif k in ["layer_pe_proj.weight", "layer_pe_proj.bias", 
                      "group_emb_proj.weight", "group_emb_proj.bias"]:
                state_dict[k] = torch.zeros_like(model_state_dict[k])
                print(f"  - {k}: initialized with zeros")
            elif k == "group_embedding.weight":
                state_dict[k] = torch.normal(mean=0.0, std=0.02, size=model_state_dict[k].shape,
                                            device=model_state_dict[k].device, 
                                            dtype=model_state_dict[k].dtype)
                print(f"  - {k}: initialized with normal distribution (mean=0.0, std=0.02)")
            # Use default initialization patterns for other parameters
            elif k.endswith('.weight'):
                state_dict[k] = torch.zeros_like(model_state_dict[k])
                print(f"  - {k}: initialized with zeros")
            elif k.endswith('.bias'):
                state_dict[k] = torch.ones_like(model_state_dict[k])
                print(f"  - {k}: initialized with ones")
            else:
                state_dict[k] = model_state_dict[k].clone()
                print(f"  - {k}: initialized with default values")
    
    # Load state dict with strict=False to allow missing keys
    model.load_state_dict(state_dict, strict=False)

    return model

# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import (
        SparseStructureEncoder, 
        SparseStructureDecoder,
    )
    
    from .sparse_structure_flow import SparseStructureFlowModel
    
    from .structured_latent_vae import (
        SLatEncoder,
        SLatGaussianDecoder,
        SLatRadianceFieldDecoder,
        SLatMeshDecoder,
        ElasticSLatEncoder,
        ElasticSLatGaussianDecoder,
        ElasticSLatRadianceFieldDecoder,
        ElasticSLatMeshDecoder,
    )
    
    from .structured_latent_flow import (
        SLatFlowModel,
        ElasticSLatFlowModel,
    )

    from .structured_latent_flow_wo_partpe import (
        SLatFlowModelWOPE,
        ElasticSLatFlowModelWOPE,
    )
