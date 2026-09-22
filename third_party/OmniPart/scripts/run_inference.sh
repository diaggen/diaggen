#!/bin/bash
# Run inference in offscreen mode using xvfb

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate omnipart

export PYVISTA_OFF_SCREEN=true
export DISPLAY=:99
export MESA_GL_VERSION_OVERRIDE=3.2

xvfb-run -a -s "-screen 0 1024x1024x24" python -m scripts.inference_omnipart --image_input $1