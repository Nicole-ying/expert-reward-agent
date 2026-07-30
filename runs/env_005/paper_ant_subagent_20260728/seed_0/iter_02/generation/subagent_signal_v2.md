# Subagent Research Signal

**Key Findings**: mean_ep_length=11.8, 20/20 early terminates. Generated reward mean=-5.58 vs original_env=-0.97 — reward function makes agent 5.8x worse than native. Forward (the intended signal) only +0.22, active 56.9% of steps.

**Component Anomalies**: upright_penalty=-5.67 dominates (5.8x larger than all other components combined). Only fires 43.9% of steps but when active the magnitude is catastrophic. height_penalty=-0.03 is negligible despite w_h=10.0 — z thresholds rarely breached.

**Training Dynamics**: No temporal snapshots available. Single-point view shows agent never escapes penalty regime: upright_penalty dwarfs forward, health gate blocks forward 43% of time, and all episodes truncate early.

**Signal Quality**: Deadly coupling: upright_penalty (w_up=5.0, hinge on up_z<0.7) punishes any tilt. Agent can't explore forward motion (forward=+0.22 active only when health_gate>0), but staying upright enough to unlock forward appears unreachable. The gate blocks the only positive signal.

**Evidence Confidence**: `high`
