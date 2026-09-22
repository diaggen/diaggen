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
    # print(f"is local: {is_local}, path: {path} because {os.path.exists(f'{path}.json')} and {os.path.exists(f'{path}.safetensors')}")
    
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
    
    # print(f"Config loaded successfully: {config.get('name', 'Name not found in config')}")
    
    if 'name' not in config:
        raise ValueError(f"Config file missing required 'name' field")
    
    model_class = config['name']
    if model_class.lower() in [k.lower() for k in __attributes.keys()]:
        # Try to find case-insensitive match
        for k in __attributes.keys():
            if k.lower() == model_class.lower():
                model_class = k
                break
        # print(f"Using model class: {model_class}")
    
    try:
        model_constructor = __getattr__(model_class)
    except AttributeError as e:
        print(f"Model lookup failed: {e}")
        raise ValueError(f"Model class '{model_class}' not found in available models: {list(__attributes.keys())}")

    # print(f"Initializing model with args: {config.get('args', {})}")
    model = model_constructor(**config.get('args', {}), **kwargs)
    
    # Load state dict
    state_dict = load_file(model_file)

    # print(f"State dict loaded successfully from {model_file}")
    
    # Check key compatibility
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    missing_keys = model_keys - loaded_keys
    unexpected_keys = loaded_keys - model_keys
    if missing_keys:
        print(f"Missing keys in state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys in state dict: {unexpected_keys}")

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
