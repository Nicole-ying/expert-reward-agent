# Subagent Research Signal

**Key Findings**: Score=69.24, 19/20 truncated (full-length episodes), only 1 terminated. Huge score range [-1067, 394] across seeds. Generated reward (0.67/step) nearly cancels original_env_reward (-0.66/step), producing near-zero net signal. Agent survives but doesn't excel.

**Component Anomalies**: upright_bonus is negligible: 0.028 mean/step, only 2.7% signed share (68.0 episode sum vs 1459 forward). height_factor at 0.87 mean — gate is nearly always open, not differentiating. forward dominates at 58.2% share but at only 0.64/step, the gradient is weak after original_reward offset.

**Training Dynamics**: Per-step means from summary: forward=0.64, height_factor=0.87, upright_bonus=0.028, generated_reward=0.67, original_env_reward=-0.66. The height gate stays wide open and the upright bonus is too small (0.1 coefficient on body_up_z averaging ~0.28) to shape policy.

**Signal Quality**: upright_bonus is a dead signal — coefficient 0.1 × body_up_z(~0.28) ≈ 0.028, dwarfed by forward. height_factor saturates near 1.0 for most of training, providing no useful gradient. The effective per-step reward (~0.67) is near the noise floor relative to episode-length variance.

**Evidence Confidence**: `high`
