---
name: run-diaggen-pipeline
description: Run the HAG4R probabilistic generation pipeline through staged image cleanup, segmentation, OmniPart, material inference, mesh processing, Genesis diagnostics, and final export.
---

# run-diaggen-pipeline

Use this skill as the runtime skill-suite root for one HAG4R asset-generation pipeline run.

The active runtime is volumetric-only. Every completed asset is a tetrahedral `.mesh`
with tet-only heterogeneous parameters and per-part `solid_fill | hollow_wall`
topology provenance. Do not ask for, infer, or preserve an asset-level surface vs
volumetric representation choice.

The Codex agent that invokes this `SKILL.md` is the pipeline orchestrator. Do not launch or emulate that orchestrator from Python.

This skill indexes the stage-level skills used by the pipeline. The orchestrator must delegate each stage to Codex-spawned VLM subagents that read the owning stage skill. Micro-skills live under their owning stage skill and are read-only guidance; they do not spawn subagents and are not global stages.

## Stage Order

1. `image-cleanup`
2. `segmentation`
3. `omnipart`
4. `material-inference`
5. `mesh-processing`
6. `genesis-diagnostics`
7. `post-mesh-texture`
8. `final-export`

When Genesis diagnostics are disabled, skip only `genesis-diagnostics`: after
`mesh-processing`, run `post-mesh-texture` and then `final-export` in that order.

## Runtime Skill Layout

- `image-cleanup/SKILL.md`
- `segmentation/SKILL.md`
- `omnipart/SKILL.md`
- `material-inference/SKILL.md`
- `mesh-processing/SKILL.md`
- `genesis-diagnostics/SKILL.md`
- `post-mesh-texture/SKILL.md`
- `final-export/SKILL.md`

Image-cleanup, material-inference, and Genesis-diagnostics micro-skills are nested below their owning stage skill. They are read-only guidance used by that owning stage agent; do not load them as top-level pipeline stages or create subagents from them.

## Tool Boundary

Pipeline skills describe stage policy and expected tool order. Python code is limited to deterministic tools, tests, scripts, and external tool invocations such as image cleanup, SAM3, OmniPart, material validation/write tools, mesh processing, Genesis diagnostics, and final export.

`state.json` is a durable runtime ledger and artifact contract. It records paths, revision roots, stage status, diagnostic anchor and probe artifacts, and final export bookkeeping. It is not an agent memory, planner, scheduler, router, or context manager.

## Public Entrypoint

Use `python -m hag4r.agentic.run_diaggen_pipeline` only to initialize paths and `state.json` for one run. The initializer records `runtime_kind=codex_skill`, enforces the repo-local `outputs/` directory contract, and preserves the established `outputs/agentic_asset_refinement/` and `outputs/run_pipeline/` relative artifact layout.

The public initializer has two entry modes only:

- raw-image initialization with `--source_image`;
- diagnostics-only snapshot initialization with `--diagnostics_revision_snapshot_dir`.

There is no public representation flag and no intermediate-stage public entrypoint.

After initialization, continue in this Codex orchestrator skill. Do not call a Python orchestrator, Python stage-agent harness, scheduler, router, or compatibility shim.

## Genesis Diagnostics Parent Boundary

This boundary applies only while the parent handles the `genesis-diagnostics` stage; the parent continues to orchestrate every other pipeline stage normally. The exhaustive parent whitelist for diagnostics is: initialize paths and `state.json`; optionally perform the one-time pending-diagnostics worker-GPU recovery described below; allocate a fresh `agent_invocation_id`; spawn one diagnostic child with the canonical `run_root` and that invocation id; wait; read the durable terminal state, child `diagnostic_recommendation`, and artifacts; call the credential-free `materialize_diagnostic_terminal_artifacts(run_root)` when derived reports/media need deterministic repair; call `apply_diagnostic_recommendation(run_root)` for a validated child revision, or explicitly call it with `diagnostic_verdict="accept"` and a nonempty `diagnostic_verdict_reason` for a validated child accept; then execute the returned parent-owned transition, or start a fresh spawned-child retry when diagnostics itself did not produce a terminal record.

The optional recovery is parent-only and is permitted only for a quiescent `full_image` run whose diagnostics are enabled; every pre-diagnostic stage is successful; diagnostics, post-mesh texture, and final export are still pending; no diagnostic attachment, session, or live-launch evidence exists; and the persisted worker GPU binding is the exact unbound record. Run it before allocating or spawning a fresh diagnostic child, and only with one explicit non-Slurm CUDA token:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 .conda/hag4r/bin/python -c 'from hag4r.agentic.runtime_state import bind_pending_diagnostics_worker_gpu_from_environment as bind; bind("outputs/agentic_asset_refinement/<run-id>")'
```

This recovery only binds the existing pending worker and synchronizes its two runtime ledgers. It is not diagnostics-stage entry, replay, attachment, or session creation. The child never invokes it.

The child's first runtime action is `attach_diagnostic_run(run_root, agent_invocation_id)`. The parent never receives, asks for, stores, reconstructs, or uses `owner_lease_token`, `attachment_id`, or `live_session_handle` for live or finalization work.

For diagnostics the parent must not import or construct `GenesisLiveApiSession`; call attach or `renew_diagnostic_owner_lease`; create, bind, inspect status, or close a diagnostic session; call `simulation_reset`, `simulate`, live geometry, or probe tools; write diagnostic evidence or recommendation; call a terminal or finalize tool; or continue or take over a failed child's episode. This prohibition is exhaustive, not a hint with a clever little loophole hiding behind it.

MCP cleanup/watchdog owns abnormal-child session/process cleanup and records typed `lease_expired_retryable` or `cleanup_retryable` operational state without erasing a durable business recommendation. The parent handles `retry_diagnostics` with a newly created child and fresh `agent_invocation_id`, and handles `retry_diagnostics_cleanup` without consuming revision budget. Never reuse or reconstruct the old token, attachment id, live handle, or in-memory episode context, and never perform parent takeover.

## Diagnostic Recommendation Handoff

The diagnostic child owns only `diagnostic_recommendation`, its route/readiness, evidence, cues, and diagnostic terminal state. It never writes `diagnostic_verdict`, `transition_action`, or top-level `state.status`. Diagnostic attachment moves only the diagnostic stage ledger to `running`; successful synthesis moves that stage to `success`; a diagnostic halt leaves the terminal session `halted` and moves the stage to `failed`.

After the child terminal is durable, consume a validated child revision deterministically from its route and diagnostics cues. Do not author a second revision verdict or reason, and never override a child revision to accept. An accept transition requires the child's already strict validated accept recommendation plus an explicit parent `accept` verdict and nonempty audit reason.

Call `apply_diagnostic_recommendation(run_root)` for revision, or `apply_diagnostic_recommendation(run_root, diagnostic_verdict="accept", diagnostic_verdict_reason=<nonempty audit reason>)` for accept. The deterministic transition preserves the child recommendation and applied decision as separate audit records, selects `transition_action`, and exclusively writes top-level `state.status`. For `transition_action: reroute`, spawn the first returned stage skill and pass its repair brief. For `post_mesh_texture`, consume the returned ordered skills as post-mesh texture then final export; do not spawn final export unless the texture runner succeeded and `submit_post_mesh_texture_stage` wrote its terminal record. For `report_only`, run no further stage; for `halt`, stop with the returned failure metadata. A full-pipeline `revise` with exhausted revision budget must return `halt`, `state.status=failed`, and `failure_kind=revision_budget_exhausted` while preserving the child revision decision.

Only `business_outcome.status=business_halt` is a new-state global halt. Missing MP4, summary, cues, or other derived artifacts are `artifact_incomplete` warnings and may be rebuilt; they never instruct a child to change an `accept` or `revise` into `halt_diagnostics`. Legacy `failed`/`rejected` terminals remain conservatively handled for compatibility.
