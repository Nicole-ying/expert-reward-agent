# Subagent Research Signal

**Key Findings**: Score=-231.68, 80% termination (16/20), ep_len=320. Generated reward per-step is positive (+0.35) but original_env_reward=-0.91/step. The agent earns ~687 generated reward per episode while the environment kills it. Score range [-1129, +32] — one seed hit positive, most crash hard.

**Component Anomalies**: upright_penalty is dead in final policy (0.0% active, ep_sum=-0.01) despite 53% nonzero during training — the hinge at body_up_z<0 is never crossed. forward dominates at 56% magnitude share. lateral_penalty (41% active) and vertical_penalty (27% active) contribute negligible penalty.

**Training Dynamics**: No temporal snapshots available. Cross-sectional evidence: forward grew from per-step training mean 0.65 to final-policy ep_sum 380 (≈1.19/step). height_factor stable at ~1.0. upright penalty vanished from training (53%→0%), suggesting policy learned to avoid tilt entirely.

**Signal Quality**: Dead gate: upright_penalty hinge at body_up_z=0 is never crossed — no corrective gradient when agent is nearly upright but unstable. Generated reward is decoupled from survival: +0.35/step gen vs -0.91 env. The agent optimizes forward velocity successfully but ignores the termination signal that kills it.

**Evidence Confidence**: `high`
