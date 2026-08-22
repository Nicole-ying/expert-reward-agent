# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Indices for next_obs
    BODY_X_VEL = 13
    BODY_Y_VEL = 14
    ROLL_VEL = 16
    PITCH_VEL = 17

    # Quaternion (w, x, y, z)
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    body_z = next_obs[0]
    forward_vel = next_obs[BODY_X_VEL]
    lateral_vel = next_obs[BODY_Y_VEL]
    roll_vel = next_obs[ROLL_VEL]
    pitch_vel = next_obs[PITCH_VEL]

    # Weights
    w_fwd = 1.0
    w_orient = 0.3
    w_lat = 0.2
    w_ang = 0.05
    w_height = 2.0

    # Boundary-hinge height penalty: safe zone [0.3, 0.9]
    height_too_low = max(0.0, 0.3 - body_z)
    height_too_high = max(0.0, body_z - 0.9)
    height_boundary_penalty = -w_height * (height_too_low + height_too_high)

    # Reward components
    forward_reward = w_fwd * forward_vel
    orientation_reward = w_orient * body_up_z
    lateral_penalty = -w_lat * abs(lateral_vel)
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    total_reward = (
        forward_reward
        + orientation_reward
        + lateral_penalty
        + angular_penalty
        + height_boundary_penalty
    )

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "height_boundary_penalty": height_boundary_penalty,
    }
    return float(total_reward), components
```
