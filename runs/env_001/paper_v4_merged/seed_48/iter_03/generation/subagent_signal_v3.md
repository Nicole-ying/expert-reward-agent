# Subagent Research Signal

**Key Findings**: soft_landing_progress dominates reward at 89.8% signed share; distance_delta only 8.0%. Episode sum mean soft=145.7, distance=13.0.

**Component Anomalies**: soft_landing_progress dominates (>70% share), distance_delta underweighted; engine_penalty small and negative but active 80%.

**Training Dynamics**: no temporal snapshots provided; cannot assess component growth/decay or scaffold drift.

**Signal Quality**: all components active (high active rates), no dead signals. soft_landing_progress overwhelms other signals, potentially masking distance improvements.

**Evidence Confidence**: `medium`
