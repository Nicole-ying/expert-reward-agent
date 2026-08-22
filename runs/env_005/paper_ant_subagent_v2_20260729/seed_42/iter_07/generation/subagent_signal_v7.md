# Subagent Research Signal

**Key Findings**: Mean eval score -585.8, 50% terminated, mean length 663.7. Per-step original_env_reward -1.06, shaped reward +0.66/step. Survival_bonus 58% signed share, forward_progress 36%.

**Component Anomalies**: Survival_bonus always active (100%), forward_progress active only 47.8% due to health gating. No dead or self-cancelling components.

**Training Dynamics**: No temporal checkpoint snapshots available; final policy only.

**Signal Quality**: Health gate thresholds (height [0.2, 0.3], [0.9, 1.0]; upright factor) cause forward_progress zero ~53% steps. Sparse forward signal, constant survival bonus may encourage non-task behavior.

**Evidence Confidence**: `low`
