# Subagent Research Signal

**Key Findings**: Landing_reward dominates (99.5% signed share, ep_sum_mean=686.5) yet final score=-24.96, 0/20 terminated (all truncated at 1000 steps). Progress and attitude components negligible.

**Component Anomalies**: Landing_reward >99% share, not dead. Attitude_penalty mean=-0.0097, near-zero share. No component >70% magnitude share (landing_reward magnitude share 99.5% = dominating).

**Training Dynamics**: No temporal monitor snapshots provided; drift across checkpoints unknown.

**Signal Quality**: All components active 100%, no dead gates. Landing_reward signal fails to induce terminal landings; episodes never terminate early despite high reward sums.

**Evidence Confidence**: `medium`
