# Subagent Research Signal

**Key Findings**: Eval score=-101.97, 80% termination (16/20), ep_len=319.6. upright_penalty dominates at -141.1/ep sum (-34.2% signed share). forward_gated contributes +122.7/ep but can't compensate. health_gate active only 41.1% of eval steps — agent rarely in healthy posture.

**Component Anomalies**: upright_penalty is the single largest magnitude component (34.2%) and is always negative. health_gate active only 41.1% (vs 57.9% in training), creating a feedback loop: when tilted, health_gate→0 drops forward multiplier from 1.0→0.3, while upright_penalty simultaneously applies -0.5 per unit below threshold 0.6. z_velocity_penalty negligible (0.4% share).

**Training Dynamics**: No monitor snapshots available — cannot assess temporal evolution (early vs late training). Unknown whether agent ever learned uprightness then collapsed, or never converged to upright gait at all.

**Signal Quality**: Two signals (upright_penalty threshold=0.6 body_up_z, health_gate ramp=0.2→0.6) overlap but with different mechanics — the penalty punishes tilt while the gate removes forward reward, creating a coupled death spiral with no clear gradient through the (0.2, 0.6) zone. Agent gets penalized AND starved simultaneously when moderately tilted.

**Evidence Confidence**: `medium`
