# Subagent Research Signal

**Key Findings**: Final policy achieves mean_eval_reward=232.57, all 20 episodes terminate successfully at mean length 460.1. However, approach_bonus dominates reward composition at 98.8% share (episode sum 139.37), while progress_reward (1.0%, sum=1.38) and landing_safety_penalty (0.2%, sum=0.236) are negligible. Per-step means confirm the imbalance: approach_bonus=0.657 vs progress=0.002 vs safety=0.0007 — a ~329:1 ratio.

**Component Anomalies**: approach_bonus is pathologically dominant (98.8% share), starving the other two components of any learning signal. progress_reward is present (96.7% active) but contributes only 1% of total reward magnitude. landing_safety_penalty is effectively invisible at 0.2% share — far too small to shape safe landing behavior.

**Mechanism Hypothesis**: The 2.0× multiplier on approach_bonus combined with the multiplicative prox-speed-angle structure creates a reward surface where the shaping bonus overwhelms task-aligned progress. The agent can achieve high scores by optimizing approach_bonus alone, bypassing both progress toward target and landing safety constraints. This is a reward-scale dominance problem, not a sparsity problem.

**Decision Implication**: PATCH approach_bonus: reduce its magnitude (e.g., lower the 2.0 coefficient or remove the 0.5 floor on contact_factor) so progress_reward and landing_safety_penalty can contribute meaningfully. Also consider scaling up landing_safety_penalty coefficients (currently 0.02–0.03) by at least 5–10× so it registers against the dominant bonus. Keep the component — don't rebuild.

**Confidence**: `high`
