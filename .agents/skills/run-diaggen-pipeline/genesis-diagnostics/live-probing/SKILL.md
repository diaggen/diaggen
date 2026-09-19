---
name: diagnostic-live-probing
description: Run bounded Genesis observe-reflect-probe loops inside one anchored episode.
---

# Diagnostic Live Probing

## V2 coverage accounting

Each region gets at most three runtime-dispatched attempts. Compile/preview/semantic mismatch/separation failure/duplicate-coverage rejection are not attempts; handler failure and purity failure are. A candidate qualifies only after a complete default 100-step (`0.1s / 0.001s`) window, no hard target error, and `grabbed_selected_part_fraction >= 0.50`; 0.49 does not qualify. A non-qualifying dispatched probe is targeting-quality evidence, not reflection, material, or asset evidence. If legal slots remain and the session is healthy, release the current controller, then revise target -> recompile -> current preview -> retry. When the agent judges that the planned semantic region genuinely spans adjacent canonical parts, use the exact declared `semantic_group_part_ids` with immutable per-member fresh-name/semantics snapshots and rationales; runtime verifies that lineage and sums just those labels for the same qualification rule. Never use a group to include an unrelated canonical part. Do not exceed three successful compiled probes in one episode. Clean-close only after a qualifying attempt, attempts are exhausted, or a true blocking runtime/lifecycle condition. The first candidate is frozen, but becomes coverage only when its owned session closes and the runtime writes the settlement. Close before moving to another region.

Use this mini skill after `define_episode(run_root, agent_invocation_id, owner_lease_token, anchor_id, ...)` succeeds for the active anchor.

## Ownership And Lifecycle Preconditions

The diagnostic child must already own a valid diagnostic attachment identified by
`attachment_id` and authenticated by the exact credential triplet `run_root`,
`agent_invocation_id`, and `owner_lease_token`. A successful `define_episode` must
exist for the current anchor, and no prior Genesis live session may remain open.

For this episode, execute the lifecycle in this order:

1. Call `create_genesis_live_session(run_root, agent_invocation_id, owner_lease_token, episode_id)` to obtain a fresh `live_session_handle`.
2. Call `bind_genesis_live_handlers(run_root, agent_invocation_id, owner_lease_token, live_session_handle)`.
3. Call `inspect_genesis_runtime_logs(run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle, stream="both", cursors={"stdout":1,"stderr":1})` before reset. Retain the independent stdout/stderr `next_cursor` values.
4. If logs remain trustworthy, call reset, inspect both streams from the retained cursors, then inspect after every simulate.
5. Run the bounded observe-reflect-probe loop. Tuple live tools
   (`inspect_genesis_runtime_logs`, `simulation_reset`, `simulate`, and `query_live_geometry_context`) always receive
   both the credential triplet and the handle. Semantic probe-target tools and
   `record_diagnostic_evidence` receive the credential triplet and no handle.
6. After healthy probe/release evidence, call `close_genesis_live_session(run_root, agent_invocation_id, owner_lease_token, live_session_handle)` so runtime settlement can complete; then call `record_diagnostic_evidence(run_root, agent_invocation_id, owner_lease_token, ...)` before authoring another episode or entering synthesis.

`live_session_handle` selects one episode-local Genesis session; it is a selector,
not authentication, and never substitutes for any member of the credential
triplet. Lifecycle bind, `get_genesis_live_session_status`, and close calls also
receive the triplet and handle. Reset is illegal before bind. Simulate is illegal
before reset. Live geometry queries are allowed only while the session is bound,
reset, or active. Never reuse a handle across episodes, assets, or diagnostic-agent
invocations.

## Pre-Dispatch Role Crosscheck

Immediately before the first compiled-probe dispatch or application, inspect the accepted anchor setup reflection and the current probe target reflection for this same `region_id`. The setup reflection must contain the durable line `ANCHOR: planned setup_anchor.anchor_region=<...> -> selected_part_id=<...> -> canonical part_name=<...> -> canonical part_semantics=<...>`; the probe reflection must contain the distinct durable line `PROBE: planned semantic_region=<...> -> selected_part_id=<...> -> canonical part_name=<...> -> canonical part_semantics=<...>`.

Both must explicitly say `Roles are not swapped.` The declared fields are opposite roles for the same planned region: an anchor is never validated against `semantic_region`, and a probe is never validated against `setup_anchor.anchor_region`.

If either line, exact triple, plan-role comparison, or anti-swap result is missing, do not dispatch or apply a compiled probe, define a later episode, or submit `matched`; ordinary reset/observe `simulate` remains legal before a probe target/reflection exists. If a mismatch is recorded, do not call `simulate`, dispatch or apply a compiled probe, define a later episode, or submit `matched` until the bounded revision -> recompile -> current preview -> recheck path passes in the relevant mini skill. This prompt precondition does not add a runtime gate or change the definition of a qualifying attempt.

Inspect stdout and stderr immediately after bind, after reset, and after every simulate. Page from retained cursors or use literal `contains` search when output is truncated or the first causal line is missing. Benign warnings may continue. An error or repeated warning that invalidates initialization, advance, retrieve, or evidence trustworthiness is blocking.

On a blocking signal, record exactly one reflection with `phase: "runtime_failure"`, exactly one raw `tool_result:<1-based-index>` ref, `knowledge_base_entry_ids: []`, `route_relevance: []`, and `next_action: "close_genesis_live_session"`. Then close the same handle and enter synthesis. Do not perform further diagnostic or live work, fabricate probe evidence or media, submit a terminal tool while live, or choose a route. Live probing decides only continue versus stop. Do not read the knowledge base, assign causal ownership, or choose a route; session synthesis owns those tasks.

## Lease Discipline

Renew ownership only after a child model turn has fully completed and immediately
before an intentional idle wait in which no ordinary authenticated request is
expected before the current lease deadline. Do not run a background or periodic
heartbeat, do not renew during model generation or an in-flight tool call, and do
not renew after a terminal result.

Any stale or rejected attachment credential is an immediate stop condition. Do not
reattach, rotate identity, continue authoring or probing, close with guessed
credentials, or emit a terminal result from the stale invocation. MCP
cleanup/watchdog owns session/process cleanup and causal failed-stage evidence. The
parent owns only authorization and spawning for a permitted fresh-child retry; it
never performs cleanup, close, or takeover. A retry means a newly spawned child, a
fresh `agent_invocation_id`, one fresh `attach_diagnostic_run(run_root,
agent_invocation_id)`, and no old token, attachment id, handle, or episode context.
If terminal closure has already occurred, do not issue any lifecycle, live,
evidence, renewal, status, or other diagnostic request.

## Loop

After create and bind, every healthy episode uses this loop:

```text
inspect_genesis_runtime_logs(run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle, stream="both", cursors={"stdout":1,"stderr":1})
simulation_reset(run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)
inspect_genesis_runtime_logs(run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle, stream="both", cursors=<retained next cursors>)
inspect the initial stitched top | ne_3q | sw_3q evidence
simulate(steps=100, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)
inspect_genesis_runtime_logs(run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle, stream="both", cursors=<retained next cursors>)
inspect the sampled stitched top | ne_3q | sw_3q evidence
query_live_geometry_context(run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle) when you need the live env_local bbox
reflect on geometry and material plausibility
read diagnostic-probe-targeting skill before the first BoxEE probe
select, compile, preview, and optionally revise one semantic probe target
verify the durable ANCHOR/PROBE exact-triple audit lines and `Roles are not swapped.` before compiled-probe dispatch
call simulate(steps=100, action={"type":"compiled_probe", "compiled_probe_target_id": ...}, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)
inspect the probe result, then reflect on geometry and material plausibility
call simulate(steps=100, action={"type":"release_probe", "compiled_probe_target_id": ...}, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)
simulate(steps=100, run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle) when another observation or settling window is needed
close_genesis_live_session(run_root, agent_invocation_id, owner_lease_token, live_session_handle)
record_diagnostic_evidence(run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, evidence=...)
repeat or enter synthesis
```

For post-probe `artifact_refs`, use the runtime-owned observations digest as the
safe default: `{"kind":"runtime_file","ref":"observations_digest",...}`.
If an indexed live reference is genuinely needed, its format is exact:
`tool_result:<1-based-index>` or `visual_evidence:<0-based-index>`. Never invent
tool-name, frame-id, evidence-id, or path-shaped refs such as
`simulate:frame_ids:...`; omit those refs and use only the
observations digest when the numeric index is not known. A successful
post-probe evidence record uses `next_action: "close_genesis_live_session"`.

## Material Concern Trigger

Use this section only while pursuing the strict accept path. It describes the
evidence needed to justify accept; it is not a prerequisite for submitting a
route-and-cues revision.

For every successful probe, inspect current RGB evidence alongside the region's
`physical_hypothesis`. Runtime-only physical evidence is admitted privately only
after release as one completed record; the one final simulation `tool_result`
is the sole numeric citation; do not reconstruct hidden telemetry or tune any knobs; do not infer vectors from images or tune hidden controller settings.

After the successful clean close, write one current `post_probe` reflection citing
the one final simulation `tool_result:<1-based-index>` and this exact typed field:

```json
"material_response_judgment": {
  "decision": "accept | revise",
  "suggested_route": "accept | segmentation | material_inference | mesh_processing"
}
```

Choose the most-supported binary judgment even when confidence is limited:
`accept` requires `suggested_route: "accept"`; `revise` requires one repair
route. `route_relevance` must exactly match the derived route (`["accept"]` or
the one revise route). A healthy endpoint pair may not use `inconclusive`, an
empty route, or `halt_diagnostics` as a third result. Missing completed evidence,
camera framing issues, and ordinary lifecycle/request failures are attempt-level
inconclusive outcomes; only a backend-reported `invalid_simulation_state` is a
blocking runtime failure.

For a declared pair, the first member's post-close reflection may judge only its
own response. Do not call either member a material contrast until the second
member also clean-closes with qualifying probes under the common runtime policy.
Then the second member's sole post-close reflection must cite the two completed
measurement `tool_result` refs (one completion ref per region) and compare the
two `physical_hypothesis` strings against the observed RGB response. Do not read private controller or vertex telemetry to construct numerical material measurements.

There is no universal numeric threshold. The ordering and any minimum/range are
source-semantic-strength conditioned. For example only, a source-grounded cap
comparison of a structured/stiffened brim with a soft textile crown can require
`D_crown / D_brim >= 10`; a smaller observed separation is a dynamic material
mismatch observation, not a static-E/material-payload trigger and not a rule for
all assets. “The parts are distinct” or a generic causal caveat never excuses
skipping this comparison when pursuing accept. `accept` must explain why the
observed ordering **and** separation satisfies the stated source-semantic
strength. If the agent instead submits a revision, synthesis selects the legal
route and authors concrete diagnostics cues directly; no dynamic mismatch,
material audit, canonical audit hint, or completed-session evidence shape is a
revision prerequisite.

On a healthy path, do not stop at frame zero or recommend `accept` before at least one successful compiled probe and one default-length post-reset simulate window for every planned region. A `revise` recommendation may be probe-less whenever a legal repair stage and concrete diagnostics cues are already known; neither runtime-failure evidence nor any other evidence shape is a revision admission requirement.

## Evidence Source

`simulate` advances bounded simulation steps, auto-pauses, saves the full RGB triptych audit sequence for humans, and returns sparse sampled stitched `top | ne_3q | sw_3q` RGB evidence as image-only blocks when supported. Runtime audit artifacts retain the complete frame path metadata. Inspect those sampled triptychs; do not scrape generated PNG or MP4 directories to replace live tool evidence.

## Allowed Live Actions

The only model-facing live actions are:

- `simulation_reset`
- `inspect_genesis_runtime_logs`
- `simulate`
- `query_live_geometry_context`
- `submit_diagnostic_probe_target_intent`
- `compile_diagnostic_probe_target`
- `preview_diagnostic_probe_target`
- `revise_diagnostic_probe_target`
- `record_diagnostic_evidence`

Before the first BoxEE probe, read `/.agents/skills/run-diaggen-pipeline/genesis-diagnostics/probe-targeting/SKILL.md` and follow that workflow. Never hand-author raw `aabb_box`, controller payloads, vertices, final coordinates, BoxEE dimensions, or executable Genesis JSON.

Compiled probes are backend-owned gentle/compliant light-touch (`轻拨`) interactions. Controller stiffness, controller strength, strength_rate, constraint_strength, soft-constraint flags such as is_soft_constraint, force, and gain are not model controls; do not request, guess, encode, or tune them.

For a v2 mechanics probe, inspect the semantic signed-axis label/arrow in the target preview. The first dispatched mechanics-conditioned member of a declared pair durably locks its exact signed cardinal axis; its peer must use that same sign and axis. This is runtime lineage only and does not add a measurement field or permit vectors/controllers in a model payload.

`query_live_geometry_context` is an order-neutral live read. Use it to fetch the current `env_local` bbox from the deformable mesh before the first BoxEE probe or whenever the live geometry has shifted. It does not advance simulation.

Compiled probe and release are backend-owned `simulate` actions; use the Loop's exact calls and default-window examples, and use ordinary bounded `simulate` for settling when appropriate.

## Reflection Rubric

After each `simulate(steps=..., run_root=run_root, agent_invocation_id=agent_invocation_id, owner_lease_token=owner_lease_token, live_session_handle=live_session_handle)` call, reflect upon the plausibility of the current geometry and material inference. Use the following rubric to guide your reflection:

- Material perspective: Does the object deform, bend, or resist motion as expected for its inferred material, stiffness, density, friction, per-part fill mode, and material contrasts? This is asset material reasoning, not controller stiffness/strength tuning.
- Topology perspective: Do hollow-wall parts preserve an apparent cavity and voxel-band realization where expected, and do solid-fill parts behave as filled material where expected?
- Geometry perspective: Does the tet mesh realization look physically valid, connected where it should be, non-penetrating, and sufficiently clean for simulation?
- Trust perspective: Do logs preserve confidence that initialization and each requested advance/retrieve actually occurred? Defer causal ownership, KB lookup, and route choice to session synthesis.
