---
name: diagnostic-probe-targeting
description: Select, compile, preview, revise, and apply semantic OmniPart-grounded BoxEE probe targets during Genesis diagnostics.
---

# Diagnostic Probe Targeting

## V2 probe contract

Submit and compile the probe with the owning `region_id`; probes/compiles never cross regions. After the current preview, record typed probe semantic evidence. Only `matched` may be dispatched; `mismatched` or `unresolved` requires bounded revision. The runtime checks anchor/probe separation and owns any overlap-exception concern. Do not author AABBs, overlap IDs, camera poses, result status, or concern state.

Use this mini skill before the first live BoxEE probe in an anchored Genesis diagnostic episode.

## Ownership Boundary

The root diagnostic child must already hold a valid attachment identified by
`attachment_id` and authenticated by the exact credential triplet `run_root`,
`agent_invocation_id`, and `owner_lease_token`. The current episode must have a
fresh, bound, reset-or-active `live_session_handle`.

This mini skill does not call `create_genesis_live_session`,
`bind_genesis_live_handlers`, `renew_diagnostic_owner_lease`, or
`close_genesis_live_session`, and does not call a terminal tool. Calls to
`compute_part_grounding_context`,
`submit_diagnostic_probe_target_intent`, `compile_diagnostic_probe_target`,
`preview_diagnostic_probe_target`, and `revise_diagnostic_probe_target` carry the
credential triplet and no handle. Probe execution and release use `simulate` with
both the credential triplet and the current handle, and only after reset.

The handle is a selector, not authentication, for the episode-local live session and
cannot replace the credential triplet. If any credential is stale or rejected,
stop immediately without reattaching, rotating identity, continuing the probe, or
calling a terminal tool. Return successful or failed probe observations to the
live-probing skill so it can record evidence and close the session before another
episode or synthesis.

## Contract

The diagnostic agent chooses the semantic probe target. Runtime owns the executable BoxEE AABB and backend-owned gentle/compliant light-touch (`轻拨`) controller behavior.

Do not hand-author raw `aabb_box`, controller payloads, vertices, final coordinates, BoxEE pose, dimensions, vectors, speed, or executable Genesis JSON. Those details are runtime-owned. A v2 intent may declare only one semantic signed cardinal `motion_axis`; it is not a raw vector or controller control.

Controller stiffness, controller strength, strength_rate, constraint_strength, soft-constraint flags such as is_soft_constraint, force, and gain are not model controls. Do not request, guess, encode, or tune them.

### Canonical probe-intent shape

Produce this complete canonical object at `target_intent`.  V1 requires exactly
these canonical fields: `schema_version`, `target_intent`,
`physical_hypothesis`, `desired_interaction`, `candidate_part_reasoning`,
`selected_part_id`, `region_hint`, `uncertainty`, and `concerns`.  A
non-singleton semantic group additionally requires `semantic_group_part_ids`
and an ordered `semantic_group_part_grounding` list; every entry is exactly
`part_id`, `part_name`, `part_semantics`, and
`member_to_planned_region_rationale`.  V2 additionally requires
`mechanics_probe_mode`, `motion_axis`, and `motion_axis_rationale`; V1 gets
the runtime's legacy defaults.

```json
{
  "schema_version": "hag4r-diagnostic-probe-target-intent-v1",
  "target_intent": "check whether the brim edge retains its shape after a light probe",
  "physical_hypothesis": "the edge band may be too compliant for its visible brim role",
  "desired_interaction": "a light semantic edge-band probe followed by observation",
  "candidate_part_reasoning": "PROBE: planned semantic_region=edge_band -> selected_part_id=3 -> canonical part_name=brim -> canonical part_semantics=hat brim edge band",
  "selected_part_id": 3,
  "region_hint": "edge_band",
  "uncertainty": "medium",
  "concerns": ["The brim appears overly soft and may sag after the light probe; compare its shape retention with the crown."]
}
```

`concerns` is semantic free text: describe observed problems honestly without
avoiding runtime vocabulary.  Stray top-level fields are accepted for rollout
robustness and discarded by runtime; they never grant semantic-group,
simulator, controller, or pair-policy authority.  Still produce the canonical
shape above rather than relying on ignored extras.

## Workflow

1. For the active `region_id`, identify the intended probe semantic only from the plan's `semantic_region`. Do not use `setup_anchor.anchor_region` to select or validate a probe. Use OmniPart source/part-label views only as corroboration for that semantic target, such as free tip, handle end, hinge-side strip, root, thin rim, or edge band.
2. Immediately before **every** `submit_diagnostic_probe_target_intent`, including a retry or revision, call `compute_part_grounding_context(run_root, agent_invocation_id, owner_lease_token)`. Look up the intended primary `selected_part_id` and any `semantic_group_part_ids` in this fresh authoritative result and copy each canonical `part_name` and `part_semantics` exactly. Do not inherit an anchor mapping, a prior episode mapping, a table already visible in the workspace index, or MaterialInference prose.
3. Call `submit_diagnostic_probe_target_intent(run_root, agent_invocation_id, owner_lease_token, target_intent=...)`.
   - For an ordinary compatibility probe, use the v1 exact-shape payload and its implicit `mechanics_probe_mode: "none"` / `motion_axis: "+Y"`. **Both v1 and v2 target-intent payloads must include `uncertainty` exactly `low`, `medium`, or `high`; it remains required when adding mechanics fields.** For a mechanics response or any non-`+Y` direction, use `schema_version: "hag4r-diagnostic-probe-target-intent-v2"` and additionally provide `mechanics_probe_mode` (`none`, `bending`, `compliance`, `relative_structural_response`, or `stretch_tension`), one `motion_axis` from `+X|-X|+Y|-Y|+Z|-Z`, and a nonempty semantic `motion_axis_rationale`. Do **not** put `target_id` in this nested payload: the top-level `region_id` owns the planned region and runtime creates the target ID.
   - Only for a non-singleton semantic group may this payload additionally include `semantic_group_part_ids` and its matching `semantic_group_part_grounding` entries under the existing group contract below. For the ordinary singleton, omit both optional group fields.
   - Select the closest fresh-authoritative `selected_part_id`.
   - Omit `semantic_group_part_ids` for the legacy single-part case; runtime persists it as `[selected_part_id]`. For a non-singleton group, the agent owns the semantic grouping judgment and must provide one ordered `semantic_group_part_grounding` entry per group ID: `part_id`, exact fresh `part_name`, exact fresh `part_semantics`, and a nonempty `member_to_planned_region_rationale`. The group must be nonempty, unique, and include the primary `selected_part_id`; never add an unrelated part to inflate grabbed-vertex purity. Runtime verifies only ID/snapshot lineage exactly against the fresh grounding table; it does not infer semantic equivalence from lexical overlap or synonyms.
   - Choose `region_hint` from `full_part`, `tip`, `root`, `edge_band`. For a declared paired member with v2 `bending`, `compliance`, or `relative_structural_response`, runtime requires `tip` or `edge_band` and compiles a local anchor-distance distal patch; `full_part` and `root` are rejected. `stretch_tension` still needs its signed-axis rationale but does not by itself force a local patch.
   - Choose a transverse anchor-to-distal axis for bending/compliance/relative structural response and a longitudinal axis for stretch/tension. The first dispatched mechanics-conditioned member of a pair locks the exact signed axis for its peer; never rotate or sign-flip the peer based on prose.
   - In `candidate_part_reasoning`, bind the probe to its only valid plan field in this exact order: `PROBE: planned semantic_region=<...> -> selected_part_id=<...> -> canonical part_name=<...> -> canonical part_semantics=<...>`. Natural-language claims such as “this is the brim” or “this is the crown” do not replace the exact ID/name lookup. Keep `target_intent`, `physical_hypothesis`, and `desired_interaction` semantic. Do not include coordinates or runtime terms.
   - Anchor and probe roles are non-interchangeable. Do not swap their plan fields or audit lines.
4. Call `compile_diagnostic_probe_target(run_root, agent_invocation_id, owner_lease_token, target_id)`.
   - Runtime compiles the selected OmniPart/final-mesh part into a BoxEE target AABB.
   - Treat warning codes as evidence; do not silently ignore low purity or pin overlap.
5. Call `preview_diagnostic_probe_target(run_root, agent_invocation_id, owner_lease_token, compile_id)`.
   - Inspect the stitched `top | ne_3q | sw_3q` triptych preview.
   - Runtime manifests retain the audited source panel PNGs, camera metadata, hashes, and dimensions.
   - Check whether the red grasp box and runtime-owned signed-axis arrow/label are visible, cover the intended semantic region, avoid obvious pin/anchor overlap, and are not absurdly oversized.
   - After this current preview, re-read the exact `selected_part_id`, `part_name`, and `part_semantics` returned by this same probe compile; when a compact compiler response omits a value, read that same-call persisted compiled record. Compare the compiled triple with the probe intent binding and the active `semantic_region`, never with `setup_anchor.anchor_region`.
   - Only if every ID/name/semantics/geometry-role/plan-role comparison passes, call `record_diagnostic_probe_target_reflection(run_root, agent_invocation_id, owner_lease_token, region_id=region_id, compile_id=compile_id, semantic_match="matched", evidence_refs=..., explanation=...)`. Its existing `explanation` must retain `PROBE: planned semantic_region=<...> -> selected_part_id=<...> -> canonical part_name=<...> -> canonical part_semantics=<...>` and `Roles are not swapped.` as the rollout-auditable probe line.
6. On any disagreement, call `record_diagnostic_probe_target_reflection(run_root, agent_invocation_id, owner_lease_token, region_id=region_id, compile_id=compile_id, semantic_match="mismatched", evidence_refs=..., explanation=...)` with current preview evidence. This is not an overlap exception. Do not submit `matched`, define an episode, apply or dispatch a compiled probe, or call `simulate` until a newly compiled and previewed target passes the exact triple and probe-role recheck.
7. If correction is useful, call `revise_diagnostic_probe_target(run_root, agent_invocation_id, owner_lease_token, compile_id, edit)` only as the bounded mismatch path, then recompile, use a current preview, and repeat the fresh-grounding/intention/triple comparison before any `matched` reflection.
   - Use only `face_adjust_percent`, `translation_fraction`, `evidence_refs`, `observed_mismatch`, `view_basis`, and `reason`.
   - `evidence_refs` must cite the current preview's `triple_view_evidence_id`, for example `triple_view_000001` or `triple_view:triple_view_000001`.
   - `observed_mismatch` must be one of `misses_semantic_region`, `too_large`, `too_small`, `pin_overlap`, `occluded`, `wrong_part`, or `acceptable_no_edit`.
   - `view_basis` must name the triptych panels that justify the edit: `top`, `ne_3q`, and/or `sw_3q`.
   - Positive face percentages expand that face; negative percentages shrink it.
   - Translation fractions move the whole box by a fraction of the edited box span.
   - Do not write final coordinates.
8. When the agent judges that a planned semantic region genuinely spans adjacent canonical parts, declare the exact `semantic_group_part_ids` and immutable per-member `semantic_group_part_grounding` lineage before compilation. Runtime verifies the exact fresh snapshots, persists that lineage, and sums only those members' grabbed labels for the unchanged qualification threshold. If a probe remains non-qualifying, release it and use the bounded revise -> recompile -> current preview -> retry path; do not use grouping to hide unrelated parts or treat the prior result as reflection, material, or asset evidence.
9. Execute the compiled target through the fully authenticated `simulate(steps=..., run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)` calls below only after the accepted anchor setup reflection contains its `ANCHOR:` audit line, the current matched probe reflection contains its `PROBE:` audit line, both name their opposite role fields for the same region, and both say `Roles are not swapped.`.
   - Call `simulate(steps=100, action={"type":"compiled_probe", "compiled_probe_target_id": <compile_id>}, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)` to apply the backend-owned gentle/compliant light-touch compiled probe and advance the bounded observation window. The current default is `100` because the default diagnostic simulate window is `0.1s / 0.001s`.
   - Inspect the sampled RGB triptychs returned by `simulate`.
   - Use `simulate(steps=100, action={"type":"release_probe", "compiled_probe_target_id": <compile_id>}, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)` only for cleanup when a release is needed. Use plain `simulate(steps=100, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)` for settling or continued observation when the compiled probe target is already adequate. Do not exceed three successful compiled probes in one episode.

## Validation

Runtime validates whether the actually grabbed object-local vertices mainly belong to the declared semantic group (or the legacy primary part alone). If validation reports `error`, record evidence and retry only when a better target or bounded edit is clear. If validation reports `warning`, use the warning as diagnostic evidence and decide whether the episode still answers the material/geometry question.

Do not treat a pretty red box as success unless the compiled-target validation and post-probe RGB evidence both support the same semantic target.
