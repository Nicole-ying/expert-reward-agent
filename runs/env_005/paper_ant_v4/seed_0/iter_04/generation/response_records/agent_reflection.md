# Response Record

1. **evidence**: The policy had 18/20 terminated episodes (length 128), with height_penalty dominating at 54.6% magnitude share (mean 121.2) and velocity_reward only 28.1 (15.7%), while score remained at 24.1 — far below the target.
2. **behavior_diagnosis**: The agent frequently falls (high termination rate), cannot maintain stable height, and generates very little forward motion; heavy height and upright penalties suppress exploration of effective gaits.
3. **signal_completeness**: All necessary signal categories exist (forward progress, height, upright, lateral, angular), but the height penalty's scale is currently overwhelming the progress signal.
4. **selected_level**: Level 1 — the height_penalty component has a reasonable mathematical form (quadratic, signed), but its extreme coefficient (10) makes it too strong relative to other components, poisoning the reward signal.
5. **selected_intervention**: Reduce the height_penalty coefficient from 10.0 to 2.0, keeping the quadratic form and all other components unchanged.
6. **falsifiable_hypothesis**: A softer height penalty will let the agent tolerate minor height deviations while still discouraging extremes, thereby freeing exploration toward forward velocity and reducing early falls, which should increase score.
7. **expected_next_round**: height_penalty magnitude share should drop (e.g., below 30%), velocity_reward share should rise, episode length may increase, and the terminal/failure rate should decrease; score should improve notably.
8. **main_risk**: The reduced penalty might allow the agent to operate near the 0.2 lower bound, increasing early termination if the safety margin becomes insufficient, potentially lowering score rather than raising it.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== Core Forward Velocity Reward ==========
    forward_velocity = next_obs[13]
    velocity_reward = 1.0 * forward_velocity

    # ========== Height Safety Penalty (relaxed coefficient) ==========
    body_height = next_obs[0]
    height_error = body_height - 0.6
    height_penalty = 2.0 * (height_error ** 2)

    # ========== Upright Pose Penalty ==========
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)
    upright_error = 1.0 - body_up_z
    upright_penalty = 0.15 * (upright_error ** 2)

    # ========== Lateral Drift Penalty ==========
    lateral_velocity = abs(next_obs[14])
    lateral_penalty = 0.1 * (lateral_velocity ** 2)

    # ========== Angular Velocity Penalty ==========
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    angular_penalty = 0.05 * (roll_vel ** 2 + pitch_vel ** 2)

    # ========== Soft Health Gate for Extreme Heights ==========
    health_factor = 1.0
    if body_height < 0.5:
        health_factor = max(0.0, (body_height - 0.2) / 0.3)
    elif body_height > 0.85:
        health_factor = max(0.0, (1.0 - body_height) / 0.15)

    velocity_reward = velocity_reward * health_factor

    # ========== Total Reward Assembly ==========
    total_reward = (
        velocity_reward
        - height_penalty
        - upright_penalty
        - lateral_penalty
        - angular_penalty
    )

    # ========== Component Dictionary ==========
    components = {
        'velocity_reward': velocity_reward,
        'height_penalty': height_penalty,
        'upright_penalty': upright_penalty,
        'lateral_penalty': lateral_penalty,
        'angular_penalty': angular_penalty
    }

    return float(total_reward), components
```
