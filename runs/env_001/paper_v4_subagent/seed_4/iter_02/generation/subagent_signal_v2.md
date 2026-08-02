# Subagent Research Signal

**Key Findings**: landing_proxy dominates at 99.7% signed share. Generated total reward ~863/eps, but original env score is -167. Terminated 1/20, truncated 19/20. failure_penalty dead.

**Component Anomalies**: failure_penalty dead (0% active). landing_proxy >99% share, dwarfing progress and stability_penalty.

**Training Dynamics**: No temporal checkpoints provided; cannot assess drift or plateau.

**Signal Quality**: Missing attractor: policy likely exploits landing_proxy (hover) without landing. failure_penalty never fires. No meaningful thresholds crossed.

**Evidence Confidence**: `medium`
