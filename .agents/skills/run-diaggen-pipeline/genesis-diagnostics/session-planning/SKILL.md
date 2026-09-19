---
name: diagnostic-session-planning
description: Plan ordered semantic anchor regions before Genesis simulation starts.
---

# Diagnostic Session Planning

Use this mini skill before any diagnostic episode is created or any live Genesis tool is called.

The root `Attachment Ownership Contract` is authoritative. Enter this phase only after `attach_diagnostic_run(run_root, agent_invocation_id)` succeeds, while no episode/live session exists and the child retains `attachment_id` plus the same attachment credentials: `run_root`, `agent_invocation_id`, and `owner_lease_token`.

## V2 Planning Contract

The first post-attach deterministic call is `record_diagnostic_session_plan(run_root, agent_invocation_id, owner_lease_token, plan)`. Plan a source-supported 2--4-region v2 session, not a one-anchor v1 session. First enumerate candidates; if fewer than two naturally exist, add distinct spatial or functional subregions, and halt planning if two still cannot be grounded. Prioritize structural boundaries, flexible/high-deformation regions, functional contacts/handles/tips, then missing/fragile/material-contrast risks; break ties by part/spatial diversity. `regions` are ascending risk ranks; when more than four candidates exist select exactly ranks 1--4 and record ranks 5..N in `omitted_region_summary`. Each region declares an anchor as a setup constraint and its independent probe region as the tested target. `relationship_to_probe` is exactly `distinct` or `overlap_exception`; the latter's `relationship_rationale` explicitly says why a distinct anchor is impossible, what remains observable, and how false stability is excluded.

## Submit One Complete Exact-Shape Payload

Construct and submit the whole v2 `plan` once; do not use rejected calls as progressive schema discovery. Its exact top-level fields are `schema_version`, `session_intent`, `candidate_region_count`, `regions`, `omitted_region_summary`, and `paired_comparisons`. Use `schema_version: "hag4r-genesis-diagnostic-session-plan-v2"`; `session_intent` is a non-empty string; `candidate_region_count` is an integer of at least two; and `regions` has two through four selected entries. When `candidate_region_count <= 4`, select every candidate and set `omitted_region_summary` to exactly `[]`. Above four, it contains exact entries with `name`, `semantic_region`, `risk_rank`, and `omission_reason` for every rank after the selected top four.

Each selected region has exactly these ten fields: `region_id`, `name`, `semantic_region`, `physical_hypothesis`, `desired_interaction`, `episode_intent`, `termination_condition`, `risk_rank`, `selection_rationale`, and `setup_anchor`. All except `risk_rank` are non-empty strings; `risk_rank` is a positive integer, selected ranks are unique and ascending, and selected plus omitted ranks are continuous from one through `candidate_region_count`.

Each `setup_anchor` has exactly six fields: `anchor_type`, `anchor_region`, `uncertainty`, `relationship_to_probe`, `relationship_rationale`, and `concerns`. `anchor_type` is exactly one of `support_contact`, `grip_root`, or `joint_hinge`; `uncertainty` is exactly `low`, `medium`, or `high`; and `relationship_to_probe` is exactly `distinct` or `overlap_exception`. Each concern has exactly `concern_id`, `concern_type`, and `summary`, all non-empty strings; every `concern_id` is globally unique across all selected regions. Each `paired_comparisons` item has exactly `pair_id` and `region_ids` (two distinct selected IDs).

`paired_comparisons` is a required top-level list. Use `[]` only when no pair of selected regions poses a relative structural-response or material-contrast question; declare one item for each such two-region comparison: `{"pair_id":"stable-id","region_ids":["region-a","region-b"]}`. When two selected regions test the same two parts with reversed anchor/probe roles for structural or material response, they MUST be declared as one pair; independent episode hypotheses do not make `[]` valid. It records membership only: never include an expected winner, threshold, controller setting, or telemetry request.

When you declare such a pair, put the comparison contract in the two existing `physical_hypothesis` strings, not in the pair object: both hypotheses must neutrally state which target is expected to have the greater/lesser anchor-relative under-load response, the source-semantic strength of that expectation (for example weak, moderate, or strong), and that the pair uses the runtime-owned common policy. This is a test expectation, not an asset-defect claim. Do not write “without presuming a relative response” for a pair created to assess relative structural/material response. Do not put an E value, material-payload number, expected repair route, controller setting, hidden telemetry request, or universal numerical threshold in either the pair object or this planning contract; ordinary unpaired regions may retain non-comparative physical hypotheses.

Author that ordering from a strict source-role evidence hierarchy. Immutable source object-class semantics plus visible functional construction/role come first: shape-holding, reinforced, or structural-support roles are authoritative against a soft textile shell, free flexible tip, or rigid housing. Use current canonical grounding and source description only to locate the component that carries the role. Treat current MaterialInference prose and numeric payload as diagnostic evidence, never as ordering authority: its `flexible` label or value may be exactly what the pair is testing. A surface-cover or geometry adjective alone--for example `cotton-covered`, `thin`, or curved--must not invert an established shape-holding/reinforced role into the more-compliant side. For example, a structured/stiffened visor or brim remains the lesser-response side relative to a soft textile crown even if its visible cover is cotton and thin. If immutable source and visible construction do not establish a role, state only weak/uncertain semantic strength; do not invent the opposite ordering from a material-payload adjective. This still permits a genuinely free flexible tip to be the greater-response side. None of this adds a field to the pair wire object.

Before submitting the plan, inspect every unordered selected-region pair A, B. If `A.semantic_region` and `B.setup_anchor.anchor_region` belong to one semantic part/material group, and `B.semantic_region` and `A.setup_anchor.anchor_region` belong to another such group, A and B MUST appear together in one `paired_comparisons` item. Extra unrelated selected regions and independent hypotheses do not exempt this check. If any reciprocal group is missing its pair declaration, do not submit the plan.

Do not limit this check to reciprocal candidates that were already chosen. When the source evidence and current canonical semantics identify two selected semantic groups whose roles pose a relative structural or material-response comparison--for example, one stiffened or rigid role and one compliant or flexible role--deliberately construct the reciprocal candidates: anchor group A and probe group B, then anchor group B and probe group A. Treat this construction as a pre-ranking reservation: before ranking or filling independent risks, reserve those two reciprocal candidates as selected region slots; only then rank and fill the remaining independent risks. Declare those candidates as their pair and write their neutral expected ordering/semantic strength in both `physical_hypothesis` strings. Do not substitute unrelated third anchors or independent episodes merely to avoid the comparison. This is a planning rule only; it does not predict a repair route, prescribe controller settings or hidden telemetry, or put a threshold into the pair wire object.

## Region semantics

The v2 plan is the only supported contract: submit one `plan` object with ordered
`regions`; do not submit the removed v1 `(session_intent, anchors)` shape. Each
region names a stable semantic target and one setup anchor for its episode, such
as a base, support foot, clamp end, hub, or handle root. The anchor constrains the
setup; the probe tests the planned region, and the runtime records their typed
identities and relationship.

This phase neither creates nor binds a live session and owns no `live_session_handle`, heartbeat/renew, close, evidence synthesis, or terminal action. If the retained credentials are stale, expired, superseded, mismatched, or invalid, end the invocation immediately; do not reattach or continue planning. The parent alone may authorize a fresh diagnostic attempt.

Each planned region must include:

- `region_id`: stable unique id.
- `name`: short unique human-readable name.
- `semantic_region`: the region to grasp/load/probe—the test target. It is not the anchor; anchor one region and probe this other region.
- `physical_hypothesis`: what physical question this anchor tests.
- `desired_interaction`: the intended high-level interaction to test.
- `setup_anchor`: the typed constrained anchor, including its `anchor_region`, `anchor_type`, `uncertainty`, `relationship_to_probe`, `relationship_rationale`, and declared setup `concerns`.
- `episode_intent`: what the corresponding episode should learn.
- `termination_condition`: when the episode has enough evidence.
- `risk_rank`: this region's positive, ascending selection rank.
- `selection_rationale`: why this region earned that rank.

Do not include executable Genesis JSON, raw coordinates, `pinning`, probe actions, controller payloads, scene objects, model objects, or mesh paths in the session plan. HAG4R owns low-level pin-box resolution and executable JSON generation.

If object scale, coarse bounds, or part identity are unclear, record the uncertainty and concerns in the anchor. The episode-authoring phase resolves the planned anchor through part grounding, `submit_diagnostic_anchor_target_intent`, `compile_diagnostic_anchor_target`, and `preview_diagnostic_anchor_target`; session planning must not pass mesh paths, raw coordinates, `pinning`, bbox proposals, or measurement requests.

Valid setup anchor types are exact and exhaustive:

- `support_contact`: region that plausibly represents support/contact setup for the episode.
- `grip_root`: region that plausibly represents a handle/root/grip setup boundary.
- `joint_hinge`: region that plausibly represents a joint, hinge, hub, or articulation setup boundary.

If an anchor is questionable, keep the best matching valid anchor type and record the uncertainty and concerns. Do not invent a fallback anchor type.

## Anchor Choice

Choose anchors that make the object physically interpretable under one-hand probing:

- stable support or base regions first;
- likely handles, tips, compliant regions, or load-bearing joints next;
- avoid redundant anchors that would test the same physical question;
- keep the session small enough to finish within the diagnostic budget.

Record expected route relevance in the hypothesis text rather than as low-level action instructions. The live loop will decide where to grasp and move inside each episode.
