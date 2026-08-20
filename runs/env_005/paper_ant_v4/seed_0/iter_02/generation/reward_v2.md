```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== Core Forward Velocity Reward (mandatory) ==========
    forward_velocity = next_obs[13]
    velocity_reward = 1.0 * forward_velocity

    # ========== Height Safety Penalty (mandatory) ==========
    body_height = next_obs[0]
    height_error = body_height - 0.6
    height_penalty = 10.0 * (height_error ** 2)   # 系数从 0.2 提升到 10.0

    # ========== Upright Pose Penalty (mandatory) ==========
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)
    upright_error = 1.0 - body_up_z
    upright_penalty = 0.15 * (upright_error ** 2)

    # ========== Lateral Drift Penalty (conditional) ==========
    lateral_velocity = abs(next_obs[14])
    lateral_penalty = 0.1 * (lateral_velocity ** 2)

    # ========== Angular Velocity Penalty (conditional) ==========
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    angular_penalty = 0.05 * (roll_vel ** 2 + pitch_vel ** 2)

    # ========== Soft Health Gate for Extreme Heights ==========
    health_factor = 1.0
    if body_height < 0.35:
        health_factor = max(0.0, (body_height - 0.2) / 0.15)
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