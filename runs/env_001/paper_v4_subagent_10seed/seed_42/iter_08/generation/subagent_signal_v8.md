# Subagent Research Signal

**Key Findings**: Eval score negative (-10.97) despite positive per-step shaped reward (1.41); zero terminated, all truncated at 1000 steps.

**Component Anomalies**: proximity_reward dominates (99.2% signed share); soft_landing dead (0% active, never triggered); stability_penalty negligible (0.8% magnitude).

**Training Dynamics**: No temporal snapshots; cannot assess drift or growth patterns.

**Signal Quality**: soft_landing gate never opens (threshold 0.3 not crossed); no episodes terminated; proximity reward likely drives approach but not precise landing.

**Evidence Confidence**: `medium`
