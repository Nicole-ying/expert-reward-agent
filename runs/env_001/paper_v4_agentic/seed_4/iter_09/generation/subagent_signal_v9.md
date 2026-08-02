# Subagent Research Signal

**Key Findings**: All 20 eval episodes terminate early (len=68.4, score=-114.07). Generated reward episode sum (+5.26: landing=4.53, progress=1.18, fuel=-0.46) is dwarfed by original_env_reward (-1.65/step → -113/episode). The generated reward has correct sign structure but is ~20× too weak to influence policy.

**Component Anomalies**: fuel_penalty is effectively dead (3.4% active, -0.46/episode, -7.4% signed share). landing dominates generated reward (73.4%) but with only ~4.5/episode can't compete with env reward. progress_delta is gated by a triple-product gate (angle×vel×angvel, gate_min=0.1) that can drop to 0.001, likely suppressing its contribution.

**Training Dynamics**: No component dynamics snapshots available — training monitoring incomplete. Cannot assess temporal trends or scaffold→final drift.

**Signal Quality**: Critical signal reachability problem: the touchdown bonus (w=10, triggered on contact>0.1) is the only high-magnitude generated component, but early termination prevents the policy from ever reaching contact. The approach reward (w=1.0) is the active landing component but too weak. Progress gate triple-product creates a 0.001 floor that may make progress signal invisible.

**Evidence Confidence**: `high`
