# Subagent Research Signal

**Key Findings**: Generated reward per-step mean=0.010 vs original_env_reward=-1.53 — a ~150:1 ratio. The env penalty completely dominates optimization. All 20 eval episodes terminate early (len=68, score=-110.6).

**Component Anomalies**: soft_landing: 0.7% active rate, mean=0.0082/step. Effectively dead gate — fires ~0.5 times per episode yet accounts for 75.5% of episode-sum generated reward (from the rare spikes). progress_gated: 90% active but mean=0.0021, too weak to matter.

**Training Dynamics**: No temporal snapshots available. Final policy shows both components active but at negligible magnitude relative to -1.53 env penalty. The gate structure (angle*vel*angvel floor=0.1) means progress_gated never exceeds 0.001 per step even with perfect behavior.

**Signal Quality**: Main issue is scale mismatch, not gating logic. Even if soft_landing fired every step at max (2.0), it would barely offset the -1.53 env penalty. Generated reward needs ~100x amplitude increase to compete as a learning signal.

**Evidence Confidence**: `high`
