def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component 1: forward speed tracking (main learning signal)
    # Encourage body_x_velocity to stay close to a target speed.
    # ------------------------------------------------------------
    target_speed = 1.5  # m/s, modest running pace
    speed_error = abs(next_obs[13] - target_speed)
    forward_tracking = -10.0 * speed_error

    # ------------------------------------------------------------
    # Component 2: stability cost bundle
    # Penalise deviations that lead to falls or inefficient gaits.
    # ------------------------------------------------------------
    # height: target 0.6 (middle of healthy range 0.2-1.0)
    height_error = next_obs[0] - 0.6
    height_cost = 5.0 * (height_error ** 2)

    # orientation: penalise tilt using the first two quaternion
    # components (x, y); they are near zero when the robot is upright.
    tilt_cost = 2.0 * (next_obs[2] ** 2 + next_obs[3] ** 2)

    # lateral velocity: suppress side‑ways motion
    lateral_cost = 1.0 * abs(next_obs[14])

    # angular velocity (roll + pitch only)
    angular_cost = 0.1 * (abs(next_obs[16]) + abs(next_obs[17]))

    stability_cost = height_cost + tilt_cost + lateral_cost + angular_cost

    # ------------------------------------------------------------
    # Total reward
    # ------------------------------------------------------------
    total_reward = forward_tracking - stability_cost

    components = {
        "forward_tracking": forward_tracking,
        "stability_cost": -stability_cost
    }

    return float(total_reward), components