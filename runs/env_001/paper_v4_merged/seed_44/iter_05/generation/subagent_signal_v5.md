# Subagent Research Signal

**Key Findings**: Score 250.9, all episodes terminated (20/20). Shaped reward small per-step (mean 0.055) but high original score. Progress dominates shaped reward (74.1% signed share, 95.9% active), angle_penalty is the only negative component (-16.6% signed share).

**Component Anomalies**: contact_reward dead (0% active). progress dominating (>70% magnitude share). No self-cancelling detected.

**Training Dynamics**: No temporal snapshots provided; dynamics across checkpoints unknown.

**Signal Quality**: Dead gate: contact_reward. landing_reward present but low magnitude (7.4% share). progress and angle_penalty both near-100% active, potentially coupled.

**Evidence Confidence**: `medium`
