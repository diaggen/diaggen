---
name: diagnostic-episode-authoring
description: Create one runtime-owned Genesis diagnostic episode from one planned anchor.
---

# Diagnostic Episode Authoring

## V2 region ownership

Use `region_id` on anchor intent and `define_episode`, never a free-standing anchor identity. The runtime binds exactly one episode to each planned region; a closed/settled episode is retained and cannot be superseded for coverage. Setup reflection is typed current-preview evidence: anchor semantic `matched`, tested DOF `preserved`, and one `resolved` disposition for every plan-authored setup concern are all required before `define_episode`. For a plan-authored `overlap_exception`, the runtime has already persisted one runtime-owned `anchor_probe_overlap` concern: carry and `acknowledged` that exact concern with current setup-preview evidence before `define_episode`. This is acknowledgement, not resolution; only current post-probe evidence may later disposition it `resolved` or `unresolved`. A mismatch, unpreserved DOF, missing acknowledgement, or unresolved concern is durable evidence that blocks definition; revise/preview rather than erase it.

Use this mini skill after a session plan exists and before live Genesis tools are called for an anchor.

The root `Attachment Ownership Contract` is authoritative. Require a successful attach, retained `attachment_id`, the same attachment credentials `run_root`, `agent_invocation_id`, and `owner_lease_token`, and the recorded session plan. Prove no prior live session is open before any authoring call; for a later anchor, the prior episode's `close_genesis_live_session(run_root, agent_invocation_id, owner_lease_token, live_session_handle)` must already have succeeded. If credentials are stale/expired/superseded/mismatched/invalid, stop the invocation immediately without reattach or continued authoring. This phase owns no heartbeat/renew.

The ordered authoring envelope is grounding -> anchor intent -> compile -> preview/revise -> setup preview -> setup reflection -> `define_episode`. Every call supplies the retained triplet. In explicit form: `compute_part_grounding_context(run_root, agent_invocation_id, owner_lease_token)`, `submit_diagnostic_anchor_target_intent(run_root, agent_invocation_id, owner_lease_token, ...)`, `compile_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, ...)`, `preview_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, ...)`, optional `revise_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, ...)`, `preview_diagnostic_episode_setup(run_root, agent_invocation_id, owner_lease_token, ...)`, `record_diagnostic_setup_reflection(run_root, agent_invocation_id, owner_lease_token, ...)`, and `define_episode(run_root, agent_invocation_id, owner_lease_token, ...)`. These deterministic calls do not accept a `live_session_handle`.

This phase ends at `define_episode`; create/bind/inspect/reset belong to live probing. Never define or author a next episode while the prior session is open, and never reuse the prior episode's handle.

## Anchor Target Intent

For the active `region_id`, use only the plan's `setup_anchor.anchor_region` as the planned anchor semantic. Do not use that region's `semantic_region` to select or validate an anchor. Immediately before **every** `submit_diagnostic_anchor_target_intent`, including a retry or revision, call `compute_part_grounding_context(run_root, agent_invocation_id, owner_lease_token)`.

The fresh authoritative table is the only source for the selected ID's canonical `part_name` and `part_semantics`; a table that was already visible, a mapping remembered from a probe or prior episode, and MaterialInference prose are not substitutes.

Choose one planned anchor and call `submit_diagnostic_anchor_target_intent(run_root, agent_invocation_id, owner_lease_token, anchor_id, anchor_intent=...)` with one complete exact-shape `anchor_intent` payload, not progressive schema discovery. Its exact fields are `schema_version`, `anchor_id`, `anchor_intent`, `selected_part_id`, `region_hint`, `candidate_part_reasoning`, `physical_boundary_condition`, `uncertainty`, and `concerns`; use `schema_version: "hag4r-diagnostic-anchor-target-intent-v1"` exactly and no extra fields. The top-level call `anchor_id`, the nested payload `anchor_id`, and the active `region_id` must be the same exact non-empty ID. `anchor_intent`, `candidate_part_reasoning`, and `physical_boundary_condition` are non-empty strings; `selected_part_id` is a non-negative integer from the fresh grounding table; `region_hint` is one of `full_part`, `support_contact`, `stable_base`, `grip_root`, `hinge_side`, `root`, `tip`, or `edge_band`; `uncertainty` is `low`, `medium`, or `high`; and `concerns` is a list of non-empty strings. In `candidate_part_reasoning`, copy the fresh authoritative binding exactly in this order and compare it only to the anchor plan field:

```text
ANCHOR: planned setup_anchor.anchor_region=<...> -> selected_part_id=<...> -> canonical part_name=<...> -> canonical part_semantics=<...>
```

Natural-language shorthand such as “this is the brim” or “this is the crown” does not establish the lookup. MaterialInference semantics are corroborating evidence only; they do not replace the fresh final part ID/name/semantics lookup. Anchors and probes are not interchangeable: do not swap their plan fields or their role lines.

Do not put raw coordinates, pin boxes, controller payloads, mesh paths, Genesis JSON, or executable runtime objects in the semantic anchor target intent.

## Anchor Compile And Preview

After `submit_diagnostic_anchor_target_intent(run_root, agent_invocation_id, owner_lease_token, anchor_id, anchor_intent=...)` succeeds, call `compile_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, anchor_id)`. The runtime derives the anchor box from the final monolithic mesh labels, selected part id, and region hint. Do not provide final coordinates.

Then call `preview_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, compile_id)` and inspect the stitched top | ne_3q | sw_3q `static_anchor_preview` triptych. After this current preview, re-read the exact `selected_part_id`, `part_name`, and `part_semantics` returned by this same anchor compile; when a compact compiler response omits a value, read that same-call persisted compiled record. Compare the compiled triple to the anchor intent's authoritative binding and to the active `setup_anchor.anchor_region`, never to `semantic_region`. Put the `ANCHOR:` line above, plus `Roles are not swapped.`, in both `candidate_part_reasoning` and the eventual `anchor_semantic_explanation` of the setup reflection.

Any ID/name/semantics/geometry-role/plan-role disagreement is an anchor semantic `mismatched`, not `matched`, even if the static preview looks plausible. First call `preview_diagnostic_episode_setup(run_root, agent_invocation_id, owner_lease_token, anchor_id)` against the current same-anchor, same-compile target to obtain its current `trial_id` and same-anchor setup-preview evidence. Then call `record_diagnostic_setup_reflection(run_root, agent_invocation_id, owner_lease_token, trial_id, anchor_semantic_match="mismatched", ...)` with that current setup-preview evidence; do not hide it in untyped prose. Then take only the bounded `revise_diagnostic_anchor_target` -> recompile -> current static preview -> new `preview_diagnostic_episode_setup` -> exact-triple and setup-role recheck path. The edit uses bounded relative `face_adjust_percent` and `translation_fraction` fields plus `evidence_refs`, `observed_mismatch`, `view_basis`, and `reason`; these are relative adjustments, not coordinates. Cite only current same-anchor, same-compile `static_anchor_preview` evidence refs such as `triple_view:<id>`. Until the recheck passes, do not submit `anchor_semantic_match="matched"`, call `define_episode`, apply a target, dispatch a compiled probe, or call `simulate`.

## Compiled Anchor Setup Preview

Setup preview uses the active compiled anchor target for the planned anchor. It also requires the current same-anchor, same-compile `static_anchor_preview` from `preview_diagnostic_anchor_target` or the latest accepted `revise_diagnostic_anchor_target` result.

Call `preview_diagnostic_episode_setup(run_root, agent_invocation_id, owner_lease_token, anchor_id)` with exactly one `anchor_id` from the active session plan. Do not pass raw coordinates, pin boxes, mesh paths, Genesis JSON, or controller payloads. HAG4R uses the compiled local mesh-frame anchor target, transforms it to executable Genesis pinning, and returns a static setup preview image plus validation status and warning codes.

For one `record_diagnostic_setup_reflection`, every evidence-bearing field --
`anchor_semantic_evidence_refs`, `tested_dof_evidence_refs`, and each
`setup_concern_dispositions[].evidence_refs` -- must use the same exact current
preview reference: `{"kind":"visual_evidence","ref": <returned
anchor_preview_evidence_id>}`. `static_anchor_preview` and
`static_setup_preview` are descriptive labels, not accepted ref kinds. Do not
substitute the trial id or an earlier preview.

If setup preview mismatches the intended boundary condition, call `record_diagnostic_setup_reflection(run_root, agent_invocation_id, owner_lease_token, trial_id, verdict="revise_setup", ...)` with a semantic `revision_request`, then call `revise_diagnostic_anchor_target(run_root, agent_invocation_id, owner_lease_token, compile_id, edit)` using current `static_anchor_preview` evidence. Neither call accepts a live handle. Do not try to repair setup by supplying coordinates to setup preview.

## One Anchor, One Episode

For the current anchor, use the anchor target intent, compiled anchor preview, and setup preview image to judge whether the setup is local, global, or suspicious. Only after the exact compiled-triple and anchor-role check passes, call `record_diagnostic_setup_reflection(run_root, agent_invocation_id, owner_lease_token, trial_id, verdict=...)` with `accept_setup`, `revise_setup`, or `proceed_with_concerns`. Its existing `anchor_semantic_explanation` must repeat the exact `ANCHOR:` audit line and `Roles are not swapped.` so the setup and rollout ledger are auditable.

The setup loop is inspired by define/render/analyze/modify/rerun: propose semantic setup, render a static preview, analyze whether it matches the intended support/contact/grip/hinge semantics, revise if useful, then rerender. This is a mental model, not a rigid SOP.

Use at most three setup trials per anchor. If the third trial still has concerns, record `proceed_with_concerns` and move on. Do not halt merely because the static setup preview is imperfect.

After an accepted setup or proceed-with-concerns setup reflection whose anchor semantic result is `matched` only after the passing recheck, call `define_episode(run_root, agent_invocation_id, owner_lease_token, anchor_id, episode_markdown=...)` with the same `anchor_id`. A mismatch may not define an episode. Do not call the low-level suite authoring function directly.

HAG4R resolves the planned semantic anchor into runtime-owned setup/pinning metadata, writes the executable Genesis scene/model/body JSON, and records per-episode anchor metadata. The agent supplies only the anchor id and optional episode notes.

Anchor type, uncertainty, setup reflection, concerns, compiled anchor id, static anchor preview evidence id, and setup validation are audit metadata preserved in the session plan, setup trial, episode record, and runtime-owned diagnostic logs. Executable Genesis pin JSON remains runtime-owned and stripped down; do not provide executable pins, controller payloads, scene objects, model objects, or mesh paths.

Revision requests are semantic only. Ask for changes like "use the lower flat support foot instead of the side wall"; never pass raw coordinates, pin boxes, Genesis JSON, controller payloads, probe actions, mesh paths, or other runtime authoring details.

## Episode Boundary

After defining an episode, all live observations and probes belong to that episode through its independently created live session. Live probing must call `record_diagnostic_evidence(run_root, agent_invocation_id, owner_lease_token, ...)` and `close_genesis_live_session(run_root, agent_invocation_id, owner_lease_token, live_session_handle)` before returning for a later anchor or terminal synthesis. A later anchor must define a new episode and receive a fresh handle before any live tool is used again; reject reuse of the previous handle.

The default live simulate window derives from `0.1s / 0.001s`, currently `simulate(steps=100, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)`. Rendered RGB triptych frames are captured every 10 simulation steps. Smaller positive `simulate.steps` values are schema-valid under the configured cap. Accept requires the complete healthy probe and settlement workflow; a revision may instead be submitted before episode definition when a deterministic static defect already identifies one legal repair route and concrete diagnostics cues.

Missing Genesis frames, failed suite generation, or failed live startup are diagnostic evidence for the current episode. Live probing records that evidence and closes any authenticated open session before synthesis decides whether `halt_diagnostics` is required.
