# Subagent Research Signal

**Key Findings**: Score 236.6, termination 85%, episode length ~420. Terminal_success_bonus dominates with 96.3% signed share.

**Component Anomalies**: landing_gentleness_penalty dead (active 0%). orientation_penalty near-dead (active 2.3%, -0.2% share). terminal_success_bonus >70% share.

**Training Dynamics**: No temporal snapshots provided; dynamics over checkpoints unavailable.

**Signal Quality**: Dead gate landing_gentleness. Progress reward active (97.8%) but contributes only 3.2% share. Success bonus fires in 47.9% steps, driving high accumulation.

**Evidence Confidence**: `medium`
