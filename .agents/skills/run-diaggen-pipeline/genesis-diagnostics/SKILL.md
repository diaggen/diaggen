---
name: genesis-diagnostics
description: Navigate the Genesis diagnostic session state machine and recommend one HAG4R repair route.
---

# Genesis Diagnostics Index

Use this skill when the Genesis diagnostic loop evaluates whether a generated HAG4R asset appears ready for downstream simulation or should be improved by rerouting through an earlier pipeline stage.

## Objective

V2 diagnostics uses a risk-ranked 2--4-region healthy workflow. An anchor is a setup constraint and its probe is the independently tested semantic region. Every region has its own runtime-owned episode, bounded three-attempt probe ledger, close settlement, and coverage status; do not accept after one healthy probe. Accept requires the full closed, settled, complete-coverage workflow. A revision may instead be submitted as soon as the agent can name one legal repair stage and provide concrete diagnostics cues for it; coverage, probes, reflections, audits, evidence, runtime failure, and settlement are not revision prerequisites. `image_cleanup` remains an upstream repair stage reference; it is not a terminal route, and `object_description` is not a route.

Run one attached diagnostic invocation containing an ordered sequence of episodes, each created from one semantic setup anchor and owning one independent Genesis live session. The diagnostic child chooses semantic anchor regions before simulation, HAG4R turns one anchor into a bounded static setup preview/reflection loop, then into runtime-owned setup/pinning for one executable Genesis episode, and the live loop probes that episode with a small, fixed action surface.

Diagnostics always run the active asset as `TetMesh + FEM.Elastic + HeterogeneousMaterial`.
The diagnostic asset payload is the canonical `.mesh`, tet-only
`heterogeneous_params.npz`, `inferred_params.json`, `volume_topology.json`, and
`metric_mesh_scaling.json`. Do not request removed non-tet loaders, triangle
material arrays, representation decisions, shell-thickness evidence, or wall
thickness evidence.

Route wrong per-part fill mode or material topology decisions to
`material_inference`. Route observed hollow voxel-band, cavity preservation,
orientation, part ownership, tet labeling, loadability, or mesh-realization defects
to `mesh_processing`. A tet-budget warning alone is context, not a repair route.
Diagnostics recommends a route; it never mutates `volume_fill_mode`,
`hollow_wall_band_layers`, voxel band width, fidelities, or mesh artifacts.

Anchor semantics are setup boundary conditions selected before live simulation. Anchors are not runtime probes, generic anti-fall trays, fallback stabilizers, executable Genesis JSON, raw coordinates, pin boxes, probe actions, controller payloads, scene objects, model objects, or mesh paths. After session planning, choose a semantic anchor target with `submit_diagnostic_anchor_target_intent(run_root, agent_invocation_id, owner_lease_token, ...)` using final part-grounding evidence: `selected_part_id`, `region_hint`, `anchor_intent`, and physical boundary condition. Then compile and preview the runtime-owned anchor target with `compile_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, ...)` and `preview_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, ...)`; revise only with bounded relative `revise_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, ...)` edits tied to current same-anchor `static_anchor_preview` evidence. These semantic deterministic calls use the retained credential triplet and no `live_session_handle`. Treat MaterialInference semantics as evidence, not authority. Setup preview uses the active compiled anchor target and current same-anchor, same-compile `static_anchor_preview`; do not pass raw coordinates in session plans, anchor target intent, anchor target edits, setup preview, or semantic revision text. The session-planning mini skill owns the full typed anchor schema.

Diagnostics identify whether the asset is ready or which stage should rerun. The diagnostic child authors only its typed recommendation and terminal/stage outcome. Once a business outcome is final and validated, the parent deterministically consumes that outcome (or a validated accept) without a second business verdict; only strict accept requires an explicit parent accept audit reason. The parent deterministically consumes a validated revision's route and diagnostics cues. The parent retains exclusive ownership of top-level run status.

The only routes you may recommend are:

- `accept`
- `segmentation`
- `material_inference`
- `mesh_processing`

Route only to a stage that can modify the evidence responsible for the issue. If the immutable reference image or description is wrong or lacks necessary evidence and none of the three repair stages can correct it, halt diagnostics rather than inventing a route.

## Mandatory Part Semantic-ID Crosscheck

The v2 plan remains semantics-only. For the active `region_id`, an anchor must
validate only `setup_anchor.anchor_region`; a probe must validate only
`semantic_region`. These plan fields are non-interchangeable: never validate an
anchor against `semantic_region` or a probe against
`setup_anchor.anchor_region`, even when a preview looks plausible.

Immediately before **every** anchor intent and **every** probe intent, including
each retry, revision, and later episode, call
`compute_part_grounding_context(run_root, agent_invocation_id, owner_lease_token)`.
Look up the chosen ID in that fresh authoritative result. A visible table from a
previous turn, intent, compile, or episode, a remembered mapping, or
MaterialInference prose is not a lookup.

Persist the fresh lookup in the applicable existing `candidate_part_reasoning`
field with this exact ordered binding, using the canonical values copied from
the authoritative table:

```text
<ROLE>: planned semantic -> selected_part_id -> canonical part_name -> canonical part_semantics/geometry role
```

Before runtime probing, retain these two separate auditable lines in the
existing reasoning/reflection fields; one sentence that merely calls a shape a
brim or crown is insufficient:

```text
ANCHOR: planned setup_anchor.anchor_region=<...> -> selected_part_id=<...> -> canonical part_name=<...> -> canonical part_semantics=<...>
PROBE: planned semantic_region=<...> -> selected_part_id=<...> -> canonical part_name=<...> -> canonical part_semantics=<...>
Roles are not swapped.
```

After every compile and its current preview, re-read the exact
`selected_part_id`, `part_name`, and `part_semantics` for that same compile from
the compiler response; when a compact response omits a value, read that
same-call persisted compiled record. Compare the exact compiled triple with the
intent binding and the correct plan role. Do not compare a remembered map or a
different compile record. The anchor comparison is against its anchor intent
and `setup_anchor.anchor_region`; the probe comparison is against its probe
intent and `semantic_region`.

Any ID, name, semantics/geometry-role, plan-role, or cross-role disagreement is
`mismatched`, never `matched`. It permits only a bounded same-target
revision -> recompile -> current preview -> recheck cycle. Once such a mismatch
is recorded, until the triple and role check passes, do not submit `matched`,
`define_episode`, apply or dispatch a compiled probe, or call `simulate`. A
semantic mismatch is not an overlap exception and cannot be converted into one.
A probe discovered after an episode was defined still cannot dispatch a target,
define a later episode, or simulate until its own correction passes. Initial
plain reset/observe simulation remains legal before a probe target/reflection
exists; the dual-role audit is a compiled-probe dispatch gate.

## Attachment Ownership Contract

**First runtime/MCP call:** `attach_diagnostic_run(run_root, agent_invocation_id)`. `agent_invocation_id` must be a non-empty stable ID for the current invocation.

Attachment credentials are the same retained triplet `run_root`, `agent_invocation_id`, and `owner_lease_token` established by the one successful `attach_diagnostic_run` call. Retain `attachment_id` as durable ownership identity/evidence for the entire child invocation. The `live_session_handle` returned by `create_genesis_live_session` is a selector, not authentication; it selects one already-owned live session and never replaces any attachment credential.

Before allocating or spawning the child, the quiescent parent may optionally perform the narrowly authorized one-time pending-diagnostics worker-GPU recovery documented by the pipeline root skill. That parent action is not diagnostics-stage entry, replay, attachment, or session creation. The diagnostic child never invokes recovery.

The child's first runtime/MCP action is exactly `attach_diagnostic_run(run_root, agent_invocation_id)`. Nothing may load/record a diagnostic plan, author an episode, change diagnostic state, or call a live lifecycle tool before attach. Every later lifecycle or deterministic call carries the same retained `run_root`, `agent_invocation_id`, and `owner_lease_token`. Only `inspect_genesis_runtime_logs`, `simulation_reset`, `simulate`, and `query_live_geometry_context` receive `live_session_handle` among tuple tools; lifecycle bind, `get_genesis_live_session_status`, and close calls also receive the handle. Probe intent/compile/preview/revise/reflection and evidence calls carry the triplet without inventing a handle argument.

Terminal closure first durably commits the validated business recommendation, then performs server-owned cleanup and derived artifact materialization, and atomically invalidates the lease. Cleanup, lease, MP4, or report failure is typed operational retry state and cannot erase or replace that recommendation. A revision request also performs server-owned cleanup of any owned live session; agent-authored closure is preferred when practical but is not a revision gate. After terminal closure the child makes no further request: no post-terminal finalizer or other child call is allowed; only the parent may call the credential-free deterministic artifact materializer.

## Lease And Failure Contract

Ordinary authenticated calls renew the lease at request entry and exit; do not add redundant heartbeats. `renew_diagnostic_owner_lease(run_root, agent_invocation_id, owner_lease_token)` is permitted only after a child model turn has fully completed and immediately before an intentional idle interval in which no ordinary authenticated request is expected before the current lease deadline. Never heartbeat during model generation, while a tool request is in flight, periodically in the background, or after terminal closure.

If credentials are stale, expired, superseded, mismatched, or invalid, stop the diagnostic invocation immediately. Do not reattach, rotate identity, retry under another invocation, reuse a handle, continue authoring/probing, or attempt a terminal call. MCP cleanup/watchdog owns session/process cleanup and records typed retryable operational state; it does not turn an already valid recommendation into business halt. The parent owns only authorization and spawning for a permitted fresh-child retry; it never performs cleanup, close, or takeover. For a typed MCP shutdown or live transport loss, the MCP server—not either child—archives and resets the old diagnostic execution attempt before the parent starts a fresh child. A retry is a fresh diagnostic attempt: a newly spawned child, fresh `agent_invocation_id`, fresh one-time attach, and no old token, attachment id, handle, or episode context.

When raw logs invalidate initialization, advance, retrieve, or physics-evidence trustworthiness, record one typed `runtime_failure` reflection citing the exact nonempty raw `tool_result:<1-based-index>`, close the same session, and enter synthesis. Do not inspect again, reset, simulate, query geometry, or target a probe after that reflection. Live probing decides only stop versus continue; synthesis diagnoses and routes. If authentication or close itself fails, stop and rely on MCP cleanup; do not call halt through a broken attachment.

## Complete Child Order

Canonical healthy form: `attach -> plan/author -> define -> create -> bind -> inspect(line 1) -> reset -> inspect -> simulate -> inspect ... -> evidence -> close -> synthesis -> exactly one terminal`. Canonical blocking form: `attach -> plan/author -> define -> create -> bind -> inspect -> runtime_failure evidence -> close -> synthesis -> exactly one terminal`.

1. The parent supplies canonical `run_root` and a fresh `agent_invocation_id`; the child first calls `attach_diagnostic_run(run_root, agent_invocation_id)` and retains `attachment_id` plus `owner_lease_token`.
2. Call `record_diagnostic_session_plan(run_root, agent_invocation_id, owner_lease_token, ...)`, then author exactly one planned anchor with the same attachment credentials through setup reflection and `define_episode(run_root, agent_invocation_id, owner_lease_token, ...)`. No prior session may remain open.
3. Call `create_genesis_live_session(run_root, agent_invocation_id, owner_lease_token, episode_id)`, retain its fresh `live_session_handle`, then call `bind_genesis_live_handlers(run_root, agent_invocation_id, owner_lease_token, live_session_handle)`.
4. Call `inspect_genesis_runtime_logs(run_root, agent_invocation_id, owner_lease_token, live_session_handle, stream="both", cursors={"stdout":1,"stderr":1})` immediately after bind and before reset. On the healthy path, call `simulation_reset(run_root, agent_invocation_id, owner_lease_token, live_session_handle)`, inspect both streams again from retained cursors, and inspect after every simulate.
5. Call `record_diagnostic_evidence(run_root, agent_invocation_id, owner_lease_token, ...)`, then `close_genesis_live_session(run_root, agent_invocation_id, owner_lease_token, live_session_handle)`.
6. Only after close, author the next planned anchor and create a new independent session, or enter synthesis. Never reuse a prior episode's handle. Close precedes both next-episode authoring and terminal synthesis.
7. For accept, close every session and complete every planned region before calling `submit_diagnostic_recommendation`. For revise, submit as soon as one legal route plus non-empty `diagnostic_cues` is known; the server cleans up any owned live session. Otherwise call `halt_diagnostics`. Use exactly one terminal tool, accept its final response, and issue no request afterward.

## Session State Machine

Read this index first, then read only the mini skill needed for the active phase. Do not preload every mini skill up front.

0. Attachment: call `attach_diagnostic_run(run_root, agent_invocation_id)` before loading/invoking planning and retain the resulting attachment identity.
1. Session planning: read `/.agents/skills/run-diaggen-pipeline/genesis-diagnostics/session-planning/SKILL.md`, then call `record_diagnostic_session_plan(run_root, agent_invocation_id, owner_lease_token, ...)`.
2. Episode authoring: read `/.agents/skills/run-diaggen-pipeline/genesis-diagnostics/episode-authoring/SKILL.md`, then for exactly one planned anchor call fresh `compute_part_grounding_context(run_root, agent_invocation_id, owner_lease_token)` immediately before each anchor intent; bind `setup_anchor.anchor_region` to its authoritative ID/name/semantics triple; call `submit_diagnostic_anchor_target_intent(run_root, agent_invocation_id, owner_lease_token, ...)`, `compile_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, ...)`, and `preview_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, ...)`; re-read and compare that same compile's exact triple before setup reflection. A disagreement is `mismatched`: first call current same-anchor `preview_diagnostic_episode_setup(run_root, agent_invocation_id, owner_lease_token, ...)` to obtain the trial/evidence, then record the typed mismatched setup reflection, and only then take bounded revise/recompile/current-preview/new-setup-preview/recheck, never `matched` or `define_episode`; after a passing check put the `ANCHOR:` audit line in the intent reasoning and setup reflection, then call `preview_diagnostic_episode_setup(run_root, agent_invocation_id, owner_lease_token, ...)`; inspect the static setup preview; call `record_diagnostic_setup_reflection(run_root, agent_invocation_id, owner_lease_token, ...)`; repeat only when useful and under three setup trials; then call `define_episode(run_root, agent_invocation_id, owner_lease_token, ...)`. None of these calls accepts a live handle.
3. Live probing: after final episode definition, read `/.agents/skills/run-diaggen-pipeline/genesis-diagnostics/live-probing/SKILL.md`; create, bind, then inspect both runtime streams from line 1 before reset. On the healthy path, reset, inspect again, and inspect after every simulate. On the blocking path, record `runtime_failure`, close, and enter synthesis without further live/probe work. Before the first healthy-path BoxEE probe target, read `/.agents/skills/run-diaggen-pipeline/genesis-diagnostics/probe-targeting/SKILL.md`; immediately before every probe intent use fresh grounding, bind only `semantic_region` to its authoritative triple, compile/preview, and re-read the same compile's exact triple. Call `record_diagnostic_probe_target_reflection(run_root, agent_invocation_id, owner_lease_token, ...)` with its current-preview evidence and semantic match result. Record the `PROBE:` audit line plus `Roles are not swapped.` only after the role check passes. No compiled-probe `simulate` may occur until the accepted anchor setup reflection has the `ANCHOR:` line and the current matched probe reflection has the `PROBE:` line for the same region with opposite declared roles.
   - Use `query_live_geometry_context(run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)` as an order-neutral live read of the current `env_local` bbox when you need fresh geometry before a BoxEE probe.
4. Only after close, repeat episode authoring/live probing with a new handle for the next planned anchor when budget allows.
5. Read synthesis before the terminal call. Require all sessions closed and full settled coverage for accept. A revision requires only a legal repair route plus non-empty `diagnostic_cues`; the terminal server owns session cleanup. Call exactly one terminal tool.

## Tool Phases

- `record_diagnostic_session_plan`: session planning.
- `compute_part_grounding_context`: authenticated final-mesh part grounding context for anchor target intent and probe targeting; use it to choose `selected_part_id` and `region_hint`.
- `submit_diagnostic_anchor_target_intent`: semantic anchor target selection for one planned anchor with selected final-mesh part id, region hint, boundary-condition intent, uncertainty, and concerns.
- `compile_diagnostic_anchor_target`: runtime compilation of the selected semantic anchor part/region into an anchor target box.
- `preview_diagnostic_anchor_target`: stitched `top | ne_3q | sw_3q` static preview of the compiled anchor target, part colors, and selected anchor region.
- `revise_diagnostic_anchor_target`: bounded relative edit of the compiled anchor target; requires current same-anchor `static_anchor_preview` evidence refs and never accepts raw coordinates.
- `preview_diagnostic_episode_setup`: pre-simulation static setup preview for one semantic anchor; uses the active compiled anchor target and requires current same-anchor, same-compile `static_anchor_preview` evidence.
- `record_diagnostic_setup_reflection`: setup-preview reflection, concerns, and semantic-only revision request.
- `define_episode`: one-anchor final episode definition after setup preview/reflection.
- `inspect_genesis_runtime_logs`: order-neutral bounded raw stdout/stderr evidence. It never diagnoses, reads the KB, classifies severity, or recommends a route.
- `simulation_reset`: healthy-path live initialization after the first post-bind log inspection; returns initial stitched `top | ne_3q | sw_3q` RGB triptych evidence.
- `simulate`: bounded live advancement with auto-pause; returns sampled stitched `top | ne_3q | sw_3q` RGB triptych evidence and accepts optional compiled-probe/release action.
- `submit_diagnostic_probe_target_intent`: semantic probe target selection with selected OmniPart/final-mesh part id and region hint.
- `compile_diagnostic_probe_target`: runtime compilation of the selected semantic part/region into a BoxEE target.
- `preview_diagnostic_probe_target`: stitched `top | ne_3q | sw_3q` static preview of the compiled grasp box, part colors, and anchor/pin box.
- `revise_diagnostic_probe_target`: bounded relative edit of the compiled target; no raw coordinates.
- `record_diagnostic_probe_target_reflection`: current-preview typed semantic match/mismatch evidence for one region-owned probe compile; it carries the retained credential triplet and no live handle.
- Probe execution/release: use the derived default `simulate(steps=100, action={"type":"compiled_probe", "compiled_probe_target_id": ...}, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)` and `simulate(steps=100, action={"type":"release_probe", "compiled_probe_target_id": ...}, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)`; raw BoxEE payloads remain runtime-owned. The current default is `100` because the default diagnostic simulate window is `0.1s / 0.001s`. When pursuing accept, each episode requires at least one successful compiled probe and allows at most three successful compiled probes. Revision submission may bypass this healthy-probe workflow once a legal route and concrete diagnostics cues are known. Use plain `simulate(steps=100, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)` for settling or continued observation when the probe target is already adequate.
- `record_diagnostic_evidence`: live observation, typed `runtime_failure`, probe interpretation, and optional synthesis evidence. A runtime-failure reflection has `knowledge_base_entry_ids: []`, empty `route_relevance`, one raw tool-result ref, and `next_action: close_genesis_live_session`.
- `submit_diagnostic_recommendation`, `halt_diagnostics`: session synthesis.

Do not call earlier-stage agents or communicate with them directly. After a successful child terminal, the parent deterministically applies a revision route and cues, or explicitly audits a strict child accept, then passes any resulting repair brief to the selected stage.
