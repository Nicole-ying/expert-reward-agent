# Subagent Research Signal

**Key Findings**: External eval score = -8.04 with 0/20 successful landings (all truncated at 1000 steps). Generated reward (+0.849/step) is strongly misaligned with true objective. progress_reward dominates (63% signed share, 75% magnitude share, episode_sum=1.31), meaning the agent learns to approach the pad but never lands. original_env_reward = -0.088/step confirms poor landing quality.

**Component Anomalies**: landing_contact_bonus: 58.5% nonzero during training but 0% active in final eval (episode_sum=0.0). It shaped early exploration via crashes but vanished in the converged policy. landing_safety_penalty: always active (100%) but negligible magnitude (episode_sum=0.517, per-step mean=0.0017). Both use distance gating that may block gradient flow near the pad.

**Mechanism Hypothesis**: Distance-gated contact bonus creates a chicken-and-egg deadlock: the bonus requires both proximity AND contact, but the agent never learns gentle contact because in the converged policy's trajectory neighborhood, contact only occurs via crashes (which trigger larger penalties). The gate suppresses the bonus precisely when it should guide landing behavior.

**Decision Implication**: REBUILD landing_contact_bonus: remove distance gating and restructure to reward successful termination (episode-end event) rather than per-step contact. Also increase landing_safety_penalty coefficients (currently 0.02-0.03) by at least 5-10x so they compete with progress_reward's dominance. Rationale: the current contact bonus is dead weight in the converged policy.

**Confidence**: `medium`
