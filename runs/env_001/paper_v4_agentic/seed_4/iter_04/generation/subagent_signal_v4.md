# Subagent Research Signal

**Key Findings**: Eval score=-113.41, all 20 episodes terminated early (mean len=68.4). Original env reward=-1.635/step dominates; generated reward=0.143/step is an order of magnitude too small to compensate. The lander crashes quickly in all episodes.

**Component Anomalies**: proximity_stability: 94.2% signed share but only 16.5% active rate — high value when it fires but gates rarely open. progress_gated: 91.9% active but only 1.3% share (ep sum=0.15) — gate is permissive but delta_dist is near-zero. fuel_penalty: effectively dead (3.7% active).

**Training Dynamics**: No component dynamics snapshots available — temporal trends across checkpoints could not be inspected. Static picture only: the three generated components sum to ~0.14/step vs env's -1.64/step.

**Signal Quality**: Generated reward cannot reach the agent: it's ~11x smaller than the native negative signal. proximity_stability gate thresholds (th_angle=0.5, th_vel=1.0, th_angvel=2.0, gate_min_stab=0.2) restrict activation to 16.5%. progress_gated is open but progress (delta_dist) is negligible — the lander doesn't move toward origin.

**Evidence Confidence**: `high`
