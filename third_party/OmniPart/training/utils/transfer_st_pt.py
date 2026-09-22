import torch
from safetensors import safe_open
from safetensors.torch import save_file

def load_safetensors_content(safetensors_file_path):
    """
    Load and return the content from a safetensors file without saving
    
    Args:
        safetensors_file_path (str): Path to the input safetensors file
        
    Returns:
        dict: Dictionary containing the tensors from the safetensors file
    """
    try:
        # Load the safetensors file
        tensors = {}
        with safe_open(safetensors_file_path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        
        print(f"Successfully loaded content from {safetensors_file_path}")
        return tensors
    except Exception as e:
        print(f"Error loading {safetensors_file_path}: {str(e)}")
        return None

def convert_safetensors_to_pt(safetensors_file_path, pt_file_path):
    """
    Convert a safetensors file to PyTorch (.pt) format
    
    Args:
        safetensors_file_path (str): Path to the input safetensors file
        pt_file_path (str): Path where the output PyTorch file will be saved
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        tensors = load_safetensors_content(safetensors_file_path)
        if tensors is None:
            return False
        
        # Save as PyTorch file
        torch.save(tensors, pt_file_path)
        
        print(f"Successfully converted {safetensors_file_path} to {pt_file_path}")
        return True
    except Exception as e:
        print(f"Error converting {safetensors_file_path} to {pt_file_path}: {str(e)}")
        return False

def load_pt_content(pt_file_path):
    """
    Load and return the content from a PyTorch (.pt) file
    
    Args:
        pt_file_path (str): Path to the input PyTorch file
        
    Returns:
        dict: Dictionary containing the tensors from the PyTorch file
    """
    try:
        # Load the PyTorch file
        tensors = torch.load(pt_file_path)
        
        print(f"Successfully loaded content from {pt_file_path}")
        return tensors
    except Exception as e:
        print(f"Error loading {pt_file_path}: {str(e)}")
        return None

def convert_pt_to_safetensors(pt_file_path, safetensors_file_path=None):
    """
    Convert a PyTorch (.pt) file to safetensors format
    
    Args:
        pt_file_path (str): Path to the input PyTorch file
        safetensors_file_path (str, optional): Path where the output safetensors file will be saved.
            If None, saves to the same location as pt_file with .safetensors extension.
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    save_path_name = None
    try:
        # Generate default output path if not provided
        if safetensors_file_path is None:
            if pt_file_path.endswith('.pt'):
                safetensors_file_path = pt_file_path[:-3] + '.safetensors'
                save_path_name = pt_file_path[:-3]
            else:
                safetensors_file_path = pt_file_path + '.safetensors'
                save_path_name = pt_file_path
        
        tensors = load_pt_content(pt_file_path)
        if tensors is None:
            return False
        
        # Save as safetensors file
        save_file(tensors, safetensors_file_path)
        
        print(f"Successfully converted {pt_file_path} to {safetensors_file_path}")
        return save_path_name
    
    except Exception as e:

        raise RuntimeError(f"Error converting {pt_file_path} to {safetensors_file_path}: {str(e)}")

def save_json(model_name, st_path_name):
    """
    Save the model name and path to a JSON file.
    
    Args:
        model_name (str): Name of the model
        st_path_name (str): Path to the safetensors file
    """
    import json
    import os
    import shutil
    # Define the source directory where JSON files are stored
    source_dir = "/mnt/pfs/users/yangyunhan/tool-repo/trellis_dev/ckpts"
    
    # Define known model paths
    model_paths = {
        "sparse_structure_decoder": "ss_dec_conv3d_16l8_fp16",
        "sparse_structure_flow_model": "ss_flow_img_dit_L_16l8_fp16",
        "slat_decoder_gs": "slat_dec_gs_swin8_B_64l8gs32_fp16",
        "slat_decoder_rf": "slat_dec_rf_swin8_B_64l8r16_fp16",
        "slat_decoder_mesh": "slat_dec_mesh_swin8_B_64l8m256c_fp16",
        "slat_flow_model": "slat_flow_img_dit_L_64l8p2_fp16"
    }

    ckpt_name = model_paths[model_name]
    
    # Construct path to the source JSON file
    source_json_path = os.path.join(source_dir, f"{ckpt_name}.json")
    print(source_json_path)

    # Check if the source JSON exists
    if os.path.exists(source_json_path):
        # Copy the JSON file to the destination
        shutil.copy(source_json_path, f"{st_path_name}.json")
        print(f"Copied JSON for model {model_name} from {source_json_path}")
    else:
        print(f"JSON for model {model_name} not found in {source_dir}, creating new one")


# Example usage
if __name__ == "__main__":
    import argparse
    from safetensors.torch import save_file
    
    parser = argparse.ArgumentParser(description="Convert safetensors file to PyTorch format or return content")
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path")
    parser.add_argument("--output", type=str, help="Output PyTorch file path (if not provided, content will just be loaded)")
    parser.add_argument("--print-keys", action="store_true", help="Print the keys of tensors in the file")
    
    args = parser.parse_args()
    
    if args.output:
        convert_safetensors_to_pt(args.input, args.output)
    else:
        tensors = load_safetensors_content(args.input)
        if tensors and args.print_keys:
            print("Keys in the safetensors file:")
            for key in tensors.keys():
                print(f"- {key}: {tensors[key].shape}")
