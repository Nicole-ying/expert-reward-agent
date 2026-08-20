# Response Record

`evidence`：20/20 episodes terminate early (mean len=68.5, score=-106.67). progress_delta is positive (+3.36 episode_sum_mean) but stability_penalty dominates negatives (-4.46). soft_landing_reward is negligible (+0.33, 4% magnitude share) despite 96% active_rate — its product form (nearness × slowness) collapses gradient because both Gaussians must be simultaneously high.

`behavior_diagnosis`：The agent approaches the target (positive progress_delta) but fails to decelerate before entering the critical zone, causing consistent crashes. The soft_landing product provides almost no usable learning signal for deceleration because it only fires appreciably when the agent is already both near AND slow — a state it never reaches.

`signal_completeness`：Two of three necessary duties exist: approach incentive (progress_delta) and near-target stability constraint (gated stability_penalty). The third duty — deceleration guidance — is structurally defective: the product form creates a "needle in a haystack" credit assignment problem where the agent must coincidentally be slow while near to receive any feedback, giving no gradient for the act of slowing down.

`selected_level`：Level 2 — product_to_noncollapsing_joint (more precisely state_to_improvement for the speed component). The current product `nearness * slowness` requires simultaneous satisfaction of two conditions, collapsing the reward surface. Replacing the slowness state-reward with a deceleration improvement-reward creates an actionable gradient: the agent is rewarded for the controlled action of reducing speed near the target, not for already being slow.

`selected_intervention`：Replace `soft_landing_reward` (product of two state Gaussians) with `deceleration_bonus = coeff * max(0, prev_speed - speed) * distance_gate`. This is the only changed component; progress_delta and stability_penalty are untouched.

`falsifiable_hypothesis`：Rewarding speed reduction (an action the agent controls step-by-step) rather than the joint state of being near-and-slow should give the agent a learnable gradient for deceleration. The broader gate radius (4.0 vs previous sigma_dist=0.2) means deceleration feedback begins well before the agent enters the danger zone, giving it time to learn controlled approach.

`expected_next_round`：deceleration_bonus episode_sum_mean should be substantially larger than soft_landing_reward's 0.33. Episode length should increase beyond 68.5 as the agent learns to brake before crashing. Score should improve, though may remain negative if the environment's base crash penalties are large.

`main_risk`：The deceleration bonus could be gamed by speed-up-then-slow-down oscillation near the target. The progress_delta component partially mitigates this (retreating to re-accelerate would be penalized), but if oscillation emerges, the gate may need tightening or a per-episode deceleration cap.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D vehicle-like trajectory optimization:
    Reach and stabilize on the central target pad with minimal engine use.
    """
    # Previous observation
    px_prev = obs[0]
    py_prev = obs[1]
    prev_distance = (px_prev**2 + py_prev**2)**0.5
    prev_speed = (obs[2]**2 + obs[3]**2)**0.5

    # Next observation
    px = next_obs[0]
    py = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # Distance to target pad center
    next_distance = (px**2 + py**2)**0.5
    # Speed magnitude
    speed = (vx**2 + vy**2)**0.5

    # 1. Main learning signal: progress delta reward
    #    Positive when approaching target, negative when retreating.
    progress_delta = 3.0 * (prev_distance - next_distance)

    # 2. Stability constraint: light penalty on high speed, large angle, high angular velocity
    #    Distance-gated: only active when agent is near the target pad (within ~2 units).
    raw_stability = -0.1 * speed - 0.05 * abs(angle) - 0.05 * abs(ang_vel)
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_distance / gate_radius)
    stability_penalty = raw_stability * distance_gate

    # 3. Deceleration bonus: reward reducing speed when near the target.
    #    Replaces the previous soft_landing_reward product (nearness * slowness).
    #    Rewards the controlled action of braking, not the coincidental state.
    deceleration = max(0.0, prev_speed - speed)
    decel_gate_radius = 4.0
    decel_gate = max(0.0, 1.0 - next_distance / decel_gate_radius)
    deceleration_bonus = 2.0 * deceleration * decel_gate

    # Combine components
    total_reward = progress_delta + stability_penalty + deceleration_bonus

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'deceleration_bonus': deceleration_bonus
    }

    return float(total_reward), components
```
