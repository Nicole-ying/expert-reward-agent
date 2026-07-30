# Subagent Research Signal

**Key Findings**: Score=424.6, ep_len=872.5, 15% termination (3/20). generated_reward per-step mean=0.576 vs original_env_reward=-0.564 — they nearly cancel. The policy earns positive shaped reward while the native environment signal is equally negative, suggesting misalignment between shaped and true objectives.

**Component Anomalies**: upright_penalty is dead (0% active, weight=0.0 by design). forward_reward dominates at 63.3% magnitude share, active 95.6% of steps. _height_gate is healthy (99.9% active, mean 0.70) but tracked as component despite being a gate. No pathological dominance or self-cancellation.

**Training Dynamics**: No temporal snapshots available. Final-policy composition is stable: forward_reward and height_gate account for 95.5% of magnitude. lateral_penalty at -4.4% and action_penalty at -0.1% are negligible. The opposing native vs shaped reward structure is the dominant dynamic.

**Signal Quality**: upright_gate (used inside forward_reward as multiplier) is not tracked as its own component — its independent restrictiveness is invisible. This is a blind spot: we cannot tell if forward_reward is driven by v_x or by gate activation. Confidence limited by missing temporal data and hidden upright_gate behavior.

**Evidence Confidence**: `medium`
