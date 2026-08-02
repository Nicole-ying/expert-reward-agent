# Subagent Research Signal

**Key Findings**: Score=96.3, 14/20 terminated, ep_len=682. Generated reward per-step mean=0.515 vs original env=-0.343. Landing_quality dominates episode-sum at +35.6 (56% signed share) despite only 7.9% active rate. Engine cost drains -12.5 (20% share) at 91.5% active.

**Component Anomalies**: landing_quality: 56% signed share from only 7.9% active rate — extremely sparse but dominant. engine_cost: -20% signed share at 91.5% active — persistent drain. progress (the main shaping signal): only 7.7% share despite 99% active. No dead components.

**Training Dynamics**: No dynamics snapshots available. From reward code: w_progress=5.0, w_landing=2.0, w_land_vel=10.0. Geometric-mean landing_quality has thresholded product (6 factors all near target before nonzero), creating a cliff-like reward.

**Signal Quality**: landing_quality dominance at 7.9% active rate means the policy sees useful signal on ~1 in 13 steps. The geometric mean creates a hard AND-gate: if any factor is off, landing_quality=0. This sparse dominance may cause the policy to drift blind between rare landing_quality spikes.

**Evidence Confidence**: `medium`
