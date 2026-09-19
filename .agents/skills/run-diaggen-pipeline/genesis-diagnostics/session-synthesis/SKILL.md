---
name: diagnostic-session-synthesis
description: Synthesize diagnostics into a strict accept, a route-and-cues repair revision, or a halt.
---

# Diagnostic Session Synthesis

Use this mini skill immediately before exactly one terminal tool. The authenticated
calls are
`submit_diagnostic_recommendation(run_root, agent_invocation_id, owner_lease_token, recommendation)`
or `halt_diagnostics(run_root, agent_invocation_id, owner_lease_token, error)`.

## Choose the terminal arm

Choose `accept` only after the complete healthy workflow. Every planned region
must have a closed `coverage_complete` settlement, qualifying probe lineage, and
the required typed post-probe judgment. A blocking runtime failure, incomplete
coverage, missing endpoint evidence, or incomplete required paired comparison
forbids accept.

Choose `revise` as soon as diagnostics can identify one legal upstream repair
stage and give that stage concrete repair instructions. Revision may occur before
or during a diagnostic session; a deterministic pre-episode defect, such as a
selected final-mesh part having zero primitives, may revise without a probe.

Choose `halt_diagnostics` when no legal asset-repair route is justified, such as
an unresolved infrastructure failure. Do not invent a route merely because the
server will accept any well-formed route-and-cues revision.

The legal repair routes are:

- `segmentation`: repair masks, part boundaries, prompt concepts, material-region
  separation, joint/compliant-region separation, or foreground assignment.
- `material_inference`: repair per-part fill mode, material topology, stiffness,
  density, friction, deformation behavior, stress behavior, or material contrast.
- `mesh_processing`: repair loadability, invalid/disconnected mesh realization,
  hollow voxel bands or cavities, orientation, part ownership, tet labels, metric
  scale, or penetration caused by realized geometry.

Route only to a stage that can modify the faulty artifact. A tet-budget warning
alone is context, not a repair route.

## Evidence and audit use

For every region that was actually probed, read all available durable
observation/reflection records. Synthesize these fields together:

- region identity, semantic intent, and physical hypothesis;
- actual observation and agent reflection;
- cited probe, tool-result, and visual evidence refs;
- uncertainty; and
- route-relevant conclusion.

Use all probed regions, not only the terminal summary, one cue, or one isolated
record. Do not invent missing records or measurements; state relevant uncertainty
in the authored reason. A missing record does not erase usable evidence from the
rest of the run.

Use runtime-admitted completed evidence, RGB/live observations, physical
hypotheses, and paired comparisons when available. The completed record remains
runtime-private; do not read endpoint summaries or reconstruct numerical
telemetry. Material audits and runtime-failure records remain useful diagnostic
inputs, but they do not authorize or block revision submission. If either exposes
a repair, translate the actionable result into ordinary
`diagnostic_cues`; do not copy an obsolete canonical-hint tuple contract.

For a runtime failure, inspect its raw logs and the current rebased state,
episode configs, active mesh, and
`.agents/diaggen_diag_knowledge/diagnostic_faults.md` when present. Identify the
earliest causal error. Revise only if one legal asset stage owns the repair;
otherwise halt.

Synthesis owns diagnosis, knowledge reuse/update, and route selection.

## Author a revision

Before submitting a revision, author and retain `route`, `reason`,
`issue_signals`, `part_indices`, and `diagnostic_cues`. Write `reason` as the
concise diagnosis and why the selected stage owns the repair. Use one or more
concise problem labels in `issue_signals`. Put known affected canonical indices
in `part_indices`; use `[]` for a non-part-specific or unknown scope, and never
guess indices.

`diagnostic_cues` is the only repair instruction delivered to the destination
stage. Make every cue self-contained: identify the affected region, state the
current defect, prescribe a stage-owned change, and name the expected result.
Include every applicable detail below; omit unavailable facts rather than guess.

### Cue detail by route

- `material_inference`: name affected semantic parts and indices; record relevant
  current values (`youngs_modulus_pa`, density, Poisson ratio, friction, or fill
  mode) and observed behavior; state the direction and current value of every
  relevant ratio; give a defensible hard minimum and target band when available;
  and name the fields and behavior to re-author. For stiffness contrast, prefer
  explicit forms such as `E_brim/E_crown`, not an ambiguous "ratio". Keep the
  resulting complete material hypotheses source-plausible and Genesis-stable.
  Never invent a numeric threshold or target band.
- `segmentation`: name the defect type (`missing`, `merged`, `over-split`,
  boundary leakage, foreground/background, or wrong ownership); locate each
  affected region by semantic name and visible color, shape, or position; state
  which masks should split, merge, remain separate, or be excluded; and propose
  concept phrases to add, remove, or replace. Include current and target part
  counts, prompt concepts, mask statistics, or fallback status when observed.
  Explain the downstream physical reason. Do not rely on old part indices alone,
  prescribe unexposed SAM3 thresholds or pixel edits, or paste the cue verbatim
  as the final `sam3_prompt`.
  Example: `Current mask 0 merges the front brim with the crown. Replace the broad
  baseball-cap concept with separate crown and front-projecting-brim concepts;
  preserve their lower-front boundary and keep stitching texture merged.`
- `mesh_processing`: name the affected semantic part and index, the realized-mesh
  defect, and the stage where it appears. Prescribe a supported current-to-target
  change in per-part `low|medium|high` fidelity and/or object-level
  `target_max_dimension_m`; state what remains unchanged, why the change can fix
  the defect, and the expected validated result. Include observed scale, face,
  tet, label, component, orientation, non-manifold, or penetration measurements
  as diagnosis, not as controls. Never prescribe target faces, keep ratio, voxel
  settings, wall thickness or band layers, component selection/deletion, gap
  bridging, fill mode, mask repair, or bypassing hard validation. If fidelity or
  metric scale cannot plausibly repair the defect, choose the owning route or
  halt instead of inventing a mesh action.
  Example: `The current target_max_dimension_m=1.0 exceeds the source-grounded
  0.26-0.32 m cap range. Set it to 0.29 m, keep all part fidelities unchanged,
  and verify the scaled maximum extent and contact behavior at that scale.`

Put part identity and repair direction inside each cue even when they also appear
in synthesis metadata, because the destination stage receives the cue itself.

When the authenticated evidence-recording path is available before the terminal
call, retain this exact metadata in a `record_diagnostic_evidence` synthesis
reflection, for example as a labeled JSON block in its `reflection` text. The
reflection schema has no top-level `reason`, `issue_signals`, or `part_indices`
fields, so never add unsupported keys to the evidence object. If that optional
durable recording path is unavailable, keep the metadata in the agent-authored
synthesis trace and still submit the revision.

These metadata and cue-detail rules govern agent authoring quality only. They do
not add schema, adjudication, evidence, or server/runtime admission gates. The
server still validates only the existing v3 wire contract. Revision does
not require a plan, episode, probe, reflection, evidence reference, material
audit, runtime failure, settlement, completed region, or the cue details above.
The server must not reject a legal route-and-nonempty-cues revision because any
of them are missing. Do not make any field depend on material audits,
unique-route inference, or another evidence shape. This authoring context is not server-side revision admission evidence.

## Revision wire payload

A revision has exactly four fields. `diagnostic_cues` is the sole durable repair
instruction list. Submit only this route-and-cues wire payload, and keep every
cue non-empty after stripping:

```json
{
  "schema_version": "hag4r-genesis-vlm-recommendation-v3",
  "recommendation": "revise",
  "route": "material_inference",
  "diagnostic_cues": [
    "Brim part 1 and crown part 0 currently both use E=5e6 Pa, so E_brim/E_crown=1. Re-author both material hypotheses so the brim is stiffer; use the source-justified hard minimum 30 and target band 100-1000 while keeping absolute values Genesis-stable and the crown comparatively compliant."
  ]
}
```

Do not add `ready`, `reason`, `issue_signals`, `part_indices`, `stage_hints`,
settlement explanations, evidence, audit summaries, or route-adjudication fields
to the wire recommendation. The runtime derives revision readiness as false and
copies the exact ordered route+cues through terminal state, artifacts, the parent
repair brief, `sim_diagnostic_cues`, and the replacement revision.

## Accept payload

Accept retains its strict exact shape, with the v3 version:

```json
{
  "schema_version": "hag4r-genesis-vlm-recommendation-v3",
  "recommendation": "accept",
  "route": "accept",
  "ready": true,
  "issue_signals": [],
  "reason": "The fully settled diagnostics support acceptance.",
  "part_indices": [],
  "stage_hints": {
    "segmentation": [],
    "material_inference": [],
    "mesh_processing": []
  }
}
```

Do not use the small revision shape to weaken acceptance. The runtime still
checks closed sessions, complete coverage, settled probe lineage, typed
post-probe judgments, and blocking runtime failures for accept.

## Terminal ownership

Use the retained `run_root`, `agent_invocation_id`, and `owner_lease_token`.
These attachment credentials and the durable `attachment_id` remain one identity.
If they become stale, fail and let the parent spawn a fresh child to reattach;
never reconstruct them. This skill owns no lease renewal or heartbeat.

Every authored episode should have closed its Genesis live session before an
accept synthesis. A revision may enter synthesis with an owned open session
because the server performs terminal cleanup; close it first when practical.
The child must never reuse an
old handle or add a `live_session_handle` to the terminal call.

A returned success is final and validated, closes the attachment, and invalidates
its lease. After it returns, issue no status, renew, close, evidence, lifecycle,
terminal, or finalizer request. The child authors the recommendation; the parent
independently applies the pipeline transition.
