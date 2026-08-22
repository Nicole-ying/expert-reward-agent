def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]

    # Body uprightness (1.0 = perfectly upright, 0.0 = tilted)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    # Guard against tiny numerical overshoot
    body_up_z = max(0.0, min(1.0, body_up_z))

    # ---------- Forward progress (bounded, only positive velocity) ----------
    vx = max(0.0, body_x_vel)
    forward_reward = vx / (1.0 + vx)          # bounded in [0, 1)

    # ---------- Height safety gate (replaces independent height penalty) ----------
    # Safe zone: [0.3, 0.9]; termination boundaries: 0.2 and 1.0
    # Ramp width 0.1 on each side, giving linear decay to 0 at the boundaries
    low_gate  = max(0.0, min(1.0, (body_z - 0.2) / 0.1))   # 0 at 0.2, 1 at 0.3
    high_gate = max(0.0, min(1.0, (1.0 - body_z) / 0.1))   # 0 at 1.0, 1 at 0.9
    height_gate = low_gate * high_gate                     # 1 inside safe zone, decays near boundaries

    # ---------- Upright posture penalty ----------
    upright_penalty = (1.0 - body_up_z)**2

    # ---------- Lateral slip penalty ----------
    lateral_penalty = abs(body_y_vel)

    # Weights
    w_forward  = 1.0
    w_upright  = 5.0
    w_lateral  = 0.5

    total_reward = (w_forward * height_gate * forward_reward
                    - w_upright  * upright_penalty
                    - w_lateral  * lateral_penalty)

    components = {
        "gated_forward":   w_forward * height_gate * forward_reward,
        "upright_penalty": w_upright * upright_penalty,
        "lateral_penalty": w_lateral * lateral_penalty
    }

    return float(total_reward), components