# Response Record

`evidence`: score improved from -123.55 to -50.90 with episode length doubling to 408, but 19/20 episodes still terminate (body height out of range); forward_velocity_reward dominates at 325 mean (53% magnitude), while height penalty at w_h=10 provides at most 0.5 penalty at the termination boundary (body_z=0.20), which is weaker than the ~0.8/step forward reward.

`behavior_diagnosis`: the agent walks forward at decent speed (~0.8 m/s) and maintains some upright posture, but eventually collapses in 19/20 episodes because the height penalty gradient near the failure boundary is too weak to deter the agent from drifting into the terminal zone.

`signal_completeness`: all three components have correct sign and mathematical form — forward progress is rewarded, orientation is rewarded, and height/lateral/angular deviations are penalized. No missing signal category; the failure is purely in the relative scale of the height sub-penalty within stability_penalty.

`selected_level`: Level 1 — the stability_penalty component has correct structure (hinge loss with appropriate margins for the healthy range), but its height sub-coefficient w_h=10 is too small relative to forward_velocity_reward, making the penalty insufficient to prevent terminal body_z excursions.

`selected_intervention`: increase w_h from 10.0 to 50.0, a 5× amplification of the height penalty within the stability_penalty component. No other coefficients, margins, or mathematical forms are changed.

`falsifiable_hypothesis`: with w_h=50, body_z deviations into the danger zone [0.20, 0.25] will incur per-step penalties of 1.0–2.5 (exceeding the ~0.8 forward reward), forcing the policy to learn height maintenance as a hard constraint; this should reduce the termination rate from 19/20 toward ≤10/20, increase average episode length, and raise the score.

`expected_next_round`: termination rate should drop substantially; episode_length should increase beyond 500; score should improve above 0 (become positive); stability_penalty magnitude may increase initially as the penalty now captures more height deviations, but should decrease as the policy learns to stay in bounds.

`main_risk`: the 5× stronger height penalty could make the agent overly conservative, reducing forward velocity to avoid any height deviation entirely; if score improves but forward_velocity drops sharply, next round may need to scale w_fwd upward or narrow the penalty's lower margin.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from the result state (next_obs)
    body_z = next_obs[0]
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # Quaternion-based body uprightness: z-component of body's local up in world frame
    body_up_z = qw * qw - qx * qx - qy * qy + qz * qz
    body_up_z = max(-1.0, min(1.0, body_up_z))

    # Weights
    w_fwd = 1.0
    w_lat = 0.5
    w_ang = 0.001
    w_h = 50.0
    w_orient = 0.5

    # 1. Main learning signal: forward velocity reward
    forward_reward = w_fwd * body_x_vel

    # 2. Stability constraint (penalty for dangerous/inefficient deviations)
    lateral_penalty = w_lat * abs(body_y_vel)
    angular_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)
    height_penalty = w_h * (
        max(0.0, 0.25 - body_z) + max(0.0, body_z - 0.95)
    )
    stability_penalty = -(lateral_penalty + angular_penalty + height_penalty)

    # 3. Orientation reward: reward staying upright to prevent collapse
    orientation_reward = w_orient * body_up_z

    total_reward = forward_reward + stability_penalty + orientation_reward

    components = {
        "forward_velocity_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "orientation_reward": orientation_reward
    }

    return float(total_reward), components
```
