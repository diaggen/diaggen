---
name: segmentation
description: Run HAG4R-owned SAM3 2D segmentation and write the OmniPart Step 16 segmentation artifact bundle.
---

# SAM3 OmniPart 2D Segmentation

Use this skill when the segmentation stage prepares cleaned image masks for the downstream OmniPart 3D stages.

## Objective

Run the single stage-local segmentation tool and produce the HAG4R-owned 2D segmentation artifact bundle that OmniPart consumes after its pre-3D segmentation steps are bypassed.

The segmentation skill-suite stage agent owns SAM3 prompt authoring. The lower-level segmentation tool only validates, splits, and caps the `sam3_prompt` you provide; it does not synthesize a fallback prompt from object names, descriptions, sample assets, or hints. The stage must not invoke OmniPart's old native pre-3D mask route.

## Inputs

- Runtime `run_root`.
- Cleaned image path from `image_cleanup`.
- Attached cleaned image in the user message when the image is small enough to inline.
- Generated object-description artifact path from `image_cleanup`.
- Compact runtime prompt context in the user message, including object name, selected object-description JSON fields, and any diagnostic segmentation suggestions.
- When this stage is the designated diagnostic repair destination, the user message includes a structured diagnostic repair brief. Treat that brief as stage-agent planning evidence, not as text to forward verbatim to tools.

## SAM3 Prompt Requirements

Read the runtime prompt context in the user message and author `sam3_prompt` yourself.

`sam3_prompt` must be a concise semicolon- or comma-separated list of noun/concept phrases for SAM3, based on the object description from image cleanup, object name, diagnostic segmentation suggestions when present, and your own observation of the cleaned image when it is attached.

You own the segmentation hint. Choose crude, coarse-grained phrases that target 2-6 final simulation parts, not every visible fragment. Merge repeated decorative pieces, texture-only regions, repeated same-material fragments, and same-material subcomponents that should move together. Preserve materially distinct regions and behaviorally important revolute joints, ball joints, compliant joints, and deformable regions when they change downstream physics. Do not overfit to a specific object category; write generic SAM3 concept phrases from the evidence. Do not expect cleanup-generated `segmentation_hints`, `joint_hints`, or `material_cues`; those lists are not produced upstream.

If a diagnostic repair brief is present, use it to decide how the `sam3_prompt` should change. Pass only the resulting concise concept phrases into `run_sam3_omnipart_2d_segmentation_stage`; do not paste the full diagnostic brief, probe report, or route adjudication into the SAM3 prompt. When submitting, include a non-empty `diagnostic_response_summary` explaining which cues changed the prompt, what changed, and which cues were irrelevant.

Do not include paths, hashes, downstream instructions, "Return separate masks", cleaned-image metadata, or verbose reasoning in `sam3_prompt`.

There is no fallback prompt. If the context is insufficient to produce a non-empty useful `sam3_prompt`, call `halt_segmentation_stage`.

## Expected Outputs

- `segmentation_manifest.json`
- prompt metadata used for SAM3.
- raw SAM3 box/mask/score outputs.
- processed ordered masks and labels for OmniPart Step 16 consumption.
- preview artifacts that make the segmentation auditable.

## Required Tool Order

1. Call `run_sam3_omnipart_2d_segmentation_stage` with the supplied `run_root`, your authored `sam3_prompt`, and the default retry budget unless instructed otherwise.
2. If the tool reports `final_part_count > 6`, author a stricter, coarser `sam3_prompt` that merges repeated decorative or same-material fragments and call the same tool again. Retry only while attempts remain.
3. If a tool call succeeds with `final_part_count <= 6`, call `submit_segmentation_stage` exactly once.
4. If the tool fails for a non-retryable reason or exhausts attempts, call `halt_segmentation_stage` exactly once with the failure reason.

## Failure Cases

- Missing cleaned image.
- Missing generated object-description artifact.
- Empty or unusable SAM3 prompt.
- Missing raw SAM3 predictions.
- Missing processed mask data for OmniPart.
- Any fallback to the removed legacy detector stack or OmniPart native pre-3D segmentation route.
