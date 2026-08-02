# Subagent Research Signal

**Key Findings**: Mean score -115.3, termination 12/20. Soft_landing_penalty dominates (ep sum 64.1, 84.7% signed share). Landing_bonus dead (0 activity, 0 reward). Progress small positive (5.2).

**Component Anomalies**: landing_bonus dead (never triggered). soft_landing_penalty >70% share (84.7%). No self-cancelling signals.

**Training Dynamics**: No monitor snapshots; temporal trends unknown.

**Signal Quality**: landing_bonus threshold never crossed (0% active). soft_landing_penalty always high. No attractor for desired landing.

**Evidence Confidence**: `medium`
