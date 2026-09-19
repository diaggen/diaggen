---
name: image-cleanup
description: Generate and validate a bounded set of clean simulation references from one raw image.
---

# Image Cleanup

This stage agent owns the complete cleanup loop. It creates the candidate image, authors its paired description, validates that pair itself, optionally regenerates once, selects a pair, and registers it. Do not create a validator agent, subagent, external graph node, or pipeline stage.

## Objective

Produce a clean reference containing exactly one complete instance of the raw-image object while preserving its identity, silhouette, proportions, openings, functional part boundaries, joints, compliance cues, material cues, and useful pose. A prettier but physically changed asset is a failure.

## Inputs and durable paths

- Raw source image, object-name hint, optional user hints, and optional manual description hint.
- `paths.image_cleanup_attempt_1_path` and `paths.image_cleanup_attempt_2_path` for generated candidates.
- `paths.cleaned_image_path`, `paths.object_description_path`, and `paths.cleanup_report_path` are registration outputs; do not write a candidate directly to the canonical cleaned path.

## Required loop

1. Inspect the raw image and author Attempt 1's `inferred_object_name` and concise `object_description` yourself.
2. Generate exactly one complete clean object with built-in Codex imagegen. Save it to `paths.image_cleanup_attempt_1_path`; record a prompt summary and concrete imagegen invocation evidence.
3. Read and execute `cleanup-validation/SKILL.md` on the raw image, Attempt 1 image, and Attempt 1 description. Keep its structured result as Attempt 1's validation record.
4. If Attempt 1 passes, select it and do not generate a second candidate.
5. If Attempt 1 semantically fails or imagegen technically fails, make at most one Attempt 2. Use the Attempt 1 validation evidence to revise the imagegen prompt. Save the candidate at `paths.image_cleanup_attempt_2_path`, author its own description, and validate it using `cleanup-validation/SKILL.md`.
6. Select Attempt 2 if it passes. If both generated candidates semantically fail, compare the two complete candidate pairs and select the better pair. If one imagegen call failed but the other produced a candidate, select the generated pair after both attempts are exhausted. If neither attempt yields a generated candidate, halt without registration.
7. Call `register_image_cleanup_stage(run_root, attempts, selected_attempt_index, selection_reason)`. The tool copies the selected candidate into the canonical paths and writes the v2 report.

There is a hard maximum of two generation attempts. Never make a third attempt. Do not mix an image from one attempt with a description from another.

## Attempt payload

Every generated attempt contains `attempt_index`, `status: generated`, `inferred_object_name`, `object_description`, `imagegen_prompt_summary`, `imagegen_invocation_evidence`, and the full validation record. A technical imagegen failure contains `attempt_index`, `status: generation_failed`, and a non-empty `error` (plus prompt/evidence when available). Missing or malformed validation is a registration error, not a semantic failure.

## Constraints

- Use only built-in Codex imagegen. No external image service, pass-through copy, fallback backend, or Python image-generation SDK.
- Use a spatially uniform pure-white background and remove clutter, labels, text, watermarks, shadows, lighting effects, duplicate instances, and unrelated props, but do not redesign the object.
- The raw image is the preservation baseline. Structural and simulation semantics outweigh aesthetics.
- Diagnostics never route to this stage. Do not accept diagnostic repair briefs or cleanup repair cues.
