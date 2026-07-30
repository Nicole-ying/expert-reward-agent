# Subagent Research Signal

**Key Findings**: eval_score=-442.5, 5/20 terminated (25%), ep_len=837. Generated reward +0.518/step but original env reward -0.827/step. Shaped reward is positive while true objective is negative — the agent is rewarded for behavior the environment penalizes.

**Component Anomalies**: gated_forward dominates at 63.2% signed share (64.4% magnitude). Gate is closed ~18-39% of steps (per-step nonzero=61%, ep-level active=81.6%). height_penalty is dead: 19.4% active rate, near-zero contribution. upright_bonus 22.5% share is secondary.

**Training Dynamics**: No temporal dynamics data available. 0/20 episodes are early-terminal (<150 steps), meaning terminations occur mid-run. Fixed eval seeds show wide score range [-1622, -1.1].

**Signal Quality**: The gate_lower=0.3 threshold may be too strict: 39% of per-step observations have gate=0, killing the forward progress signal. The agent achieves positive shaped reward (+0.518) despite poor true performance (-0.827), indicating the reward components don't align with the environment's success metric.

**Evidence Confidence**: `medium`
