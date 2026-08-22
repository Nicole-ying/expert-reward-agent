`evidence`: V5 acceleration_bonus contributes only 29.06 episode_sum (1.2% share, 46.7% active) vs V3 best forward_speed_bonus ~389; agent survives full 1000 steps at moderate speed (~2.17 m/s avg) with zero failures; last round's state_to_improvement acceleration change dropped score 96 points below best.
`behavior_diagnosis`: Agent has learned a stable moderate-speed hopping gait that sustains full episodes but plateaus; the acceleration bonus fires only during speed increases (46.7% active) and provides negligible incentive to push beyond the current cruising speed.
`signal_completeness`: Forward progress and stability constraints are present and functioning; the missing piece is an effective signal to disproportionately reward higher sustained speeds — the V3 quadratic bonus provided this but was unbounded, causing exploitation in V4; the V5 acceleration replacement structurally fails because steady-speed maintenance produces near-zero delta-v.
`selected_level`: Level 2 — `unbounded_to_bounded`: the V3 quadratic bonus had the correct intent (disproportionately reward high speed) but its unbounded form allowed V4's exploitation; bounding it preserves the gradient while capping extreme-value dominance.
`selected_intervention`: Replace V5's `acceleration_bonus = max(0, delta_v)` with a bounded quadratic speed bonus `w_fwd2 * v_fwd^2 / (1 + alpha * v_fwd^2)`, using V3 best as the base structure.
`falsifiable_hypothesis`: A bounded quadratic saturates at high speeds, preventing the exploitation pattern seen in V4 (where per-step bonus grew to 0.99 vs 0.39 in V3), while still providing ~3x stronger gradient at moderate speeds than linear progress alone, pushing the agent past the plateau.
`expected_next_round`: Score should exceed 3260 (previous best); forward_speed_bonus episode_sum should be 300-500 (bounded against V4's exploitation); episode length stays 1000 with zero failures; active_rate for the speed bonus near 100%.
`main_risk`: If alpha is too small, bounding is too aggressive and the bonus collapses toward a constant (fails to differentiate speeds); if w_fwd2 is too high relative to w_fwd, quadratic gradient near moderate speeds could destabilize the gait before saturating.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V6 reward: V3 best structure + bounded quadratic speed bonus (Level 2 unbounded_to_bounded).
    
    Replaces the unbounded v^2 bonus (which V4 exploited to 0.99/step) with a saturating
    v^2/(1+alpha*v^2) form. This preserves the disproportionate high-speed gradient while
    capping extreme values to prevent destabilization.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress
    w_fwd2 = 0.15        # bounded quadratic bonus coefficient (slightly > V3's 0.1)
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Bounded quadratic bonus: v^2 / (1 + alpha * v^2)
    # At v=2: bonus=0.50, v=3: bonus=0.93, asymptote=3.0
    # Prevents the unbounded exploitation seen in V4 while retaining gradient
    speed_sq = v_fwd ** 2
    alpha = 0.05
    forward_speed_bonus = w_fwd2 * speed_sq / (1.0 + alpha * speed_sq)

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + forward_speed_bonus + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "forward_speed_bonus": forward_speed_bonus,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
    }

    return float(total_reward), components
```