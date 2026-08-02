# Subagent Research Signal

**Key Findings**: mean_eval_score=-116.46, 12/20 episodes terminate early (<150 steps), all 20/20 terminated. Per-step generated_reward total=0.0002 vs original_env_reward=-0.7575 — the shaped reward is ~3800× smaller than the environment penalty, making it invisible as a learning signal. proximity_delta dominates generated components at 91.4% magnitude share but adds only ~0.005/step.

**Component Anomalies**: velocity_penalty active_rate=1.5% — nearly dead. The gate (threshold=0.5, dist_cur computed from origin) almost never opens because the lander rarely reaches within 0.5 of the pad. orientation_penalty is always active (100%) but contributes only 2.9% magnitude share with negligible per-step magnitude.

**Mechanism Hypothesis**: The generated reward components are dwarfed by the raw environment penalty, so the policy receives essentially no shaped guidance. The velocity_penalty gate is excessively strict (dist_cur < 0.5), rendering it a dead component that cannot shape approach behavior or encourage soft landing during descent.

**Decision Implication**: PATCH: (1) scale all generated components up by at least 100× so they are comparable to original_env_reward magnitude; (2) relax the velocity_penalty proximity gate (e.g., threshold ≥5.0 or use a smooth falloff) so it can shape the full descent trajectory, not just the final 0.5 units.

**Confidence**: `high`
