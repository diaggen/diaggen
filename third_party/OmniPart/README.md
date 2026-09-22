# OmniPart: Part-Aware 3D Generation with Semantic Decoupling and Structural Cohesion [SIGGRAPH Asia 2025]

<div align="center">

[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-blue.svg)](https://omnipart.github.io/)
[![Paper](https://img.shields.io/badge/📑-Paper-green.svg)](https://arxiv.org/abs/2507.06165)
[![Model](https://img.shields.io/badge/🤗-Model-yellow.svg)](https://huggingface.co/omnipart)
[![Online Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/omnipart/OmniPart)

</div>

![teaser](assets/doc/teaser.jpg)

## 🔥 Updates

### 📅 October 2025
- Pretrained models, interactive demo, training code and data processing. 

## 🔨 Installation

Clone the repo:
```bash
git clone https://github.com/HKU-MMLab/OmniPart
cd OmniPart
```

Install packages for headless rendering:
```bash
sudo apt install xvfb libgl1-mesa-dev
```

Create a conda environment. The HAG4R-validated environment uses PyTorch
2.7.0 with CUDA 12.8 wheels, whose arch list includes `sm_120` and
`compute_120`. The local validation run was performed on an Ada GPU; rerun the
same validation on Blackwell hardware before claiming hardware-level sm120
coverage.

```bash
OMNIPART_ENV_PREFIX=/media/eric/data/conda_envs/omnipart-sm120-master
conda create -p "$OMNIPART_ENV_PREFIX" python=3.10 -y
conda activate "$OMNIPART_ENV_PREFIX"
```

Install dependencies. `CUDA_HOME` must point at a CUDA 12.x toolkit with
`nvcc`; the validated setup used `/usr/local/cuda` pointing to CUDA 12.8.
`TORCH_CUDA_ARCH_LIST` includes the local Ada GPU (`8.9`) and Blackwell/sm120
(`12.0`) so source-built CUDA extensions contain both targets.

```bash
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

pip install --upgrade pip wheel "setuptools<81" ninja==1.13.0 packaging==24.2
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-github.txt
pip install --no-build-isolation -r requirements-huggingface.txt
```

Validate the environment:

```bash
python -m pip check
python -c "import torch; assert 'sm_120' in torch.cuda.get_arch_list(); print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
python -c "import flash_attn, spconv, cumm, nvdiffrast; from diff_gaussian_rasterization import GaussianRasterizationSettings; assert 'kernel_size' in GaussianRasterizationSettings._fields; print('OmniPart CUDA extensions import successfully')"
python -m scripts.inference_omnipart --help
```

The requirements above install the HAG4R-validated Step 16 manifest inference
path. That path consumes a precomputed segmentation manifest and does not use
OmniPart's native SAM/detectron2 image segmentation stack. The interactive demo
(`app.py`) and legacy native 2D segmentation modules still import
`segment-anything` and `detectron2`; install `segment-anything` and a
`detectron2` build compatible with `torch==2.7.0+cu128` separately before using
that path.

## 💡 Usage

### Launch Demo

```bash
python app.py
```

### Inference Scripts

If running OmniPart with command lines, you need to obtain the segmentation mask of the input image first. The mask is saved as a .exr file with the shape [h, w, 3], where the last dimension contains the 2D part_id replicated across all three channels.

```bash
./scripts/run_inference.sh <IMAGE_PATH>
```

The required model weights will be automatically downloaded:
- OmniPart model from [OmniPart](https://huggingface.co/omnipart) → local directory `ckpt/`

### Training

#### Data processing

Step 1: Render multi-view images of parts and overall shapes, following [TRELLIS Step 4](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md#step-4-render-multiview-images).

Step 2: Voxelize parts and overall shapes with `dataset_toolkits/voxelize_part.py` and `dataset_toolkits/voxelize_overall.py`.

Step 3: Extract DINO features of parts and overall shapes, following [TRELLIS Step 6](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md#step-6-extract-dino-features).

Step 4: Encode SLat of parts and overall shapes, following [TRELLIS Step 8](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md#step-8-encode-slat).

Step 5: Merge SLat of parts and overall shapes with `dataset_toolkits/merge_slat.py`.

Step 6: Render image and mask conditions with `dataset_toolkits/blender_render_img_mask.py`.

#### Training code
Fill in the values for `data_root`, `train_mesh_list`, `val_mesh_list` and `denoiser` in `configs/training_part_synthesis.json`. The `denoiser` field requires the path to a diffusion model checkpoint in `.pt` format (using `training/utils/transfer_st_pt.py`) that you wish to finetune, for example: `ckpt/slat_flow_img_dit_L_64l8p2_fp16.pt`.

```bash
python train.py --config configs/training_part_synthesis.json --output_dir {OUTPUT_PATH} --data_dir {SLat_PATH}
```

## ⭐ Acknowledgements

We would like to thank the following open-source projects and research works that made OmniPart possible:

- [TRELLIS](https://github.com/microsoft/TRELLIS)
- [PartField](https://github.com/nv-tlabs/PartField)
- [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)

We are grateful to the broader research community for their open exploration and contributions to the field of 3D generation.

## 📚 Citation

```
@article{yang2025omnipart,
        title={Omnipart: Part-aware 3d generation with semantic decoupling and structural cohesion},
        author={Yang, Yunhan and Zhou, Yufan and Guo, Yuan-Chen and Zou, Zi-Xin and Huang, Yukun and Liu, Ying-Tian and Xu, Hao and Liang, Ding and Cao, Yan-Pei and Liu, Xihui},
        journal={arXiv preprint arXiv:2507.06165},
        year={2025}
}
```
