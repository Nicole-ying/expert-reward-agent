# Subagent Research Signal

**Key Findings**: Score=-664.53, mean episode length=713.2, 7/20 terminated (35%). Original env reward dominates at -0.989/step while generated reward is only +0.081/step — a 12:1 penalty-to-reward ratio. The agent survives but barely moves (avg body_x_velocity ≈ 0.09).

**Component Anomalies**: upright_bonus is negative (mean=-0.073/step, 100% active) — it's a penalty, not a bonus. At 0.2 * body_up_z, this implies avg body_up_z ≈ -0.37, meaning the agent is persistently tilted. forward (mean=+0.182, 100% active) is positive but dwarfed by env penalty. height_factor=0.771 gates ~23% of forward reward.

**Training Dynamics**: No temporal dynamics available (no monitor snapshots). Static picture: height_factor dominates at 57% signed share of generated reward. forward contributes 12.8% signed share / 29.2% magnitude. upright_bonus is only 2.1% signed share — too weak to correct posture.

**Signal Quality**: Two coupled failure modes: (1) upright_bonus at 0.2 coefficient cannot overcome the tilt — the attractor for upright posture is missing. (2) forward reward at 2.0 * vx with height_factor gating yields only +0.182 — insufficient to overcome -0.989 env penalty. The height gate works (safe zone maintained) but forward coefficient is too low for the env penalty scale.

**Evidence Confidence**: `high`
