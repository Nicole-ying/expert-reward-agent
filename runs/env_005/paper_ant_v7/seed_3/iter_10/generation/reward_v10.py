def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]

    # Body uprightness (1.0 = perfectly upright, 0.0 = tilted)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    body_up_z = max(0.0, min(1.0, body_up_z))

    # Forward progress (bounded, only positive velocity)
    vx = max(0.0, body_x_vel)
    forward_reward = vx / (1.0 + vx)          # bounded in [0, 1)

    # Height safety gate
    low_gate  = max(0.0, min(1.0, (body_z - 0.2) / 0.1))   # 0 at 0.2, 1 at 0.3
    high_gate = max(0.0, min(1.0, (1.0 - body_z) / 0.1))   # 0 at 1.0, 1 at 0.9
    height_gate = low_gate * high_gate                     # 1 inside safe zone

    # Lateral stability gate (replaces lateral_penalty)
    # Gate decays smoothly with absolute lateral velocity, never goes negative
    lateral_gate = 2.718281828 ** (-abs(body_y_vel) / 0.5)  # ~1 at vy=0, ~0.37 at vy=0.5

    # Upright posture penalty (preserved but will be dominated by forward)
    upright_penalty = (1.0 - body_up_z)**2

    # Weights
    w_forward  = 1.0
    w_upright  = 5.0

    total_reward = (w_forward * height_gate * lateral_gate * forward_reward
                    - w_upright * upright_penalty)

    components = {
        "gated_forward":   w_forward * height_gate * lateral_gate * forward_reward,
        "upright_penalty": w_upright * upright_penalty,
        "lateral_gate":    lateral_gate   # factor, for monitoring
    }

    return float(total_reward), components