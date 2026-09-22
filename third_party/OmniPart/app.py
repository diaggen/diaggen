import gradio as gr
import spaces
import os
import shutil
os.environ['SPCONV_ALGO'] = 'native'
from huggingface_hub import hf_hub_download

from app_utils import (
    generate_parts,
    prepare_models,
    process_image,
    apply_merge,
    DEFAULT_SIZE_TH,
    TMP_ROOT,
)

EXAMPLES = [
    ["assets/example_data/knight.png", 1800, "6,0,26,20,7;13,1,22,11,12,2,21,27,3,24,23;5,18;4,17;19,16,14,25,28", 42],
    ["assets/example_data/car.png", 2000, "12,10,2,11;1,7", 42],
    ["assets/example_data/warhammer.png", 1800, "7,1,0,8", 0],
    ["assets/example_data/snake.png", 3000, "2,3;0,1;4,5,6,7", 42],
    ["assets/example_data/Batman.png", 1800, "4,5", 42],
    ["assets/example_data/robot1.jpeg", 1600, "0,5;10,14,3;1,12,2;13,11,4;7,15", 42],
    ["assets/example_data/astronaut.png", 2000, "0,4,6;1,8,9,7;2,5", 42],
    ["assets/example_data/crossbow.jpg", 2000, "2,9;10,12,0,7,11,8,13;4,3", 42],
    ["assets/example_data/robot.jpg", 1600, "7,19;15,0;6,18", 42],
    ["assets/example_data/robot_dog.jpg", 1000, "21,9;2,12,10,15,17;11,7;1,0;13,19;4,16", 0],
    ["assets/example_data/crossbow.jpg", 1600, "9,2;10,15,13;7,14,8,11;0,12,16;5,3,1", 42],
    ["assets/example_data/robot.jpg", 1800, "1,2,3,5,4,16,17;11,7,19;10,14;18,6,0,15;13,9;12,8", 0],
    ["assets/example_data/robot_dog.jpg", 1000, "2,12,10,15,17,8,3,5,13,19,6,14;11,7;1,0,21,9,11;4,16", 0],
]

HEADER = """

# OmniPart: Part-Aware 3D Generation with Semantic Decoupling and Structural Cohesion

🔮 Generate **part-aware 3D content** from a single 2D image with **2D mask control**.

## How to Use

**🚀 Quick Start**: Select an example below and click **"▶️ Run Example"**


**📋 Custom Image Processing**:
1. **Upload Image** - Select your image file
2. **Click "Segment Image"** - Get initial 2D segmentation  
3. **Merge Segments** - Enter merge groups like `0,1;3,4` and click **"Apply Merge"** (Recommend keeping **2-15 parts**)
4. **Click "Generate 3D Model"** - Create the final 3D results
"""


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_ROOT, str(req.session_hash))
    shutil.rmtree(user_dir)


with gr.Blocks(title="OmniPart") as demo:
    gr.Markdown(HEADER)
    
    state = gr.State({})
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<div style='text-align: center'>\n\n## Input\n\n</div>")
            
            input_image = gr.Image(label="Upload Image", type="filepath", height=250, width=250)
            
            with gr.Row():
                segment_btn = gr.Button("Segment Image", variant="primary", size="lg")
                run_example_btn = gr.Button("▶️ Run Example", variant="secondary", size="lg")
            
            size_threshold = gr.Slider(
                minimum=600, 
                maximum=4000, 
                value=DEFAULT_SIZE_TH, 
                step=200, 
                label="Minimum Segment Size (pixels)",
                info="Segments smaller than this will be ignored"
            )
            
            gr.Markdown("### Merge Controls")
            merge_input = gr.Textbox(
                label="Merge Groups", 
                placeholder="0,1;3,4", 
                lines=2,
                info="Specify which segments to merge (e.g., '0,1;3,4' merges segments 0&1 together and 3&4 together)"
            )
            merge_btn = gr.Button("Apply Merge", variant="primary", size="lg")
            
            gr.Markdown("### 3D Generation Controls")
            
            seed_slider = gr.Slider(
                minimum=0,
                maximum=10000,
                value=42,
                step=1,
                label="Generation Seed",
                info="Random seed for 3D model generation"
            )
            
            cfg_slider = gr.Slider(
                minimum=0.0,
                maximum=15.0,
                value=7.5,
                step=0.5,
                label="CFG Strength",
                info="Classifier-Free Guidance strength"
            )
            
            generate_mesh_btn = gr.Button("Generate 3D Model", variant="secondary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("<div style='text-align: center'>\n\n## Results Display\n\n</div>")
            
            with gr.Row():
                initial_seg = gr.Image(label="Init Seg", height=220, width=220)
                pre_merge_vis = gr.Image(label="Pre-merge", height=220, width=220)
                merged_seg = gr.Image(label="Merged Seg", height=220, width=220)
                
            with gr.Row():
                bbox_mesh = gr.Model3D(label="Bounding Boxes", height=350)
                whole_mesh = gr.Model3D(label="Combined Parts", height=350)
                exploded_mesh = gr.Model3D(label="Exploded Parts", height=350)
            
            with gr.Row():
                combined_gs = gr.Model3D(label="Combined 3D Gaussians", clear_color=(0.0, 0.0, 0.0, 0.0), height=350)
                exploded_gs = gr.Model3D(label="Exploded 3D Gaussians", clear_color=(0.0, 0.0, 0.0, 0.0), height=350)

    with gr.Row():
        examples = gr.Examples(
            examples=EXAMPLES,
            inputs=[input_image, size_threshold, merge_input, seed_slider],
            cache_examples=False,
        )

    demo.load(start_session)
    demo.unload(end_session)

    segment_btn.click(
        process_image,
        inputs=[input_image, size_threshold],
        outputs=[initial_seg, pre_merge_vis, state]
    )
    
    merge_btn.click(
        apply_merge,
        inputs=[merge_input, state],
        outputs=[merged_seg, state]
    )
    
    generate_mesh_btn.click(
        generate_parts,
        inputs=[state, seed_slider, cfg_slider],
        outputs=[bbox_mesh, whole_mesh, exploded_mesh, combined_gs, exploded_gs]
    )

    run_example_btn.click(
        fn=process_image,
        inputs=[input_image, size_threshold],
        outputs=[initial_seg, pre_merge_vis, state]
    ).then(
        fn=apply_merge,
        inputs=[merge_input, state],
        outputs=[merged_seg, state]
    ).then(
        fn=generate_parts,
        inputs=[state, seed_slider, cfg_slider],
        outputs=[bbox_mesh, whole_mesh, exploded_mesh, combined_gs, exploded_gs]
    )

if __name__ == "__main__":
    os.makedirs("ckpt", exist_ok=True)
    sam_ckpt_name = "sam_vit_h_4b8939.pth"
    partfield_ckpt_name = "partfield_encoder.ckpt"
    bbox_gen_ckpt_name = "bbox_gen.ckpt"

    sam_ckpt_path = os.path.join("ckpt", sam_ckpt_name)
    if not os.path.exists(sam_ckpt_path):
        sam_ckpt_path = hf_hub_download(
            repo_id="omnipart/OmniPart_modules",
            filename=sam_ckpt_name,
            local_dir="ckpt",
        )

    partfield_ckpt_path = os.path.join("ckpt", partfield_ckpt_name)
    if not os.path.exists(partfield_ckpt_path):
        partfield_ckpt_path = hf_hub_download(
            repo_id="omnipart/OmniPart_modules",
            filename=partfield_ckpt_name,
            local_dir="ckpt",
        )

    bbox_gen_ckpt_path = os.path.join("ckpt", bbox_gen_ckpt_name)
    if not os.path.exists(bbox_gen_ckpt_path):
        bbox_gen_ckpt_path = hf_hub_download(
            repo_id="omnipart/OmniPart_modules",
            filename=bbox_gen_ckpt_name,
            local_dir="ckpt",
        )

    prepare_models(sam_ckpt_path, partfield_ckpt_path, bbox_gen_ckpt_path)

    port = int(os.getenv("PORT", "8080"))
    demo.launch(share=False, server_name="0.0.0.0", server_port=port)