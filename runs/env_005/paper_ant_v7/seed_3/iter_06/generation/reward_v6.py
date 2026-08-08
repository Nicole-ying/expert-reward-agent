def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract observation indices ----------
    body_z_next = next_obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]

    # ---------- forward velocity reward (primary) ----------
    w_fwd = 1.5
    forward_reward = w_fwd * body_x_vel

    # ---------- height safety gate (dual-bound) ----------
    # termination: body_z <= 0.2 or body_z >= 1.0
    # safe zone: [0.35, 0.85] where gate = 1.0
    # gate decays linearly to 0 at boundaries [0.2, 1.0]
    z_low_safe = 0.35
    z_low_dead = 0.2
    z_high_safe = 0.85
    z_high_dead = 1.0

    gate_low = min(1.0, max(0.0, (body_z_next - z_low_dead) / (z_low_safe - z_low_dead)))
    gate_high = min(1.0, max(0.0, (z_high_dead - body_z_next) / (z_high_dead - z_high_safe)))
    height_gate = gate_low * gate_high

    # ---------- gated forward ----------
    gated_forward = forward_reward * height_gate

    # ---------- upright posture reward (light) ----------
    # body_up = 1 - 2*(qx^2 + qy^2), ranges [-1, 1], 1 = perfectly upright
    body_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    w_up = 0.1
    upright_reward = w_up * body_up

    # ---------- lateral drift penalty (light) ----------
    w_lat = 0.3
    lateral_penalty = -w_lat * (body_y_vel ** 2)

    # ---------- total reward ----------
    total_reward = gated_forward + upright_reward + lateral_penalty

    components = {
        "gated_forward": gated_forward,
        "height_gate": height_gate,
        "upright_reward": upright_reward,
        "lateral_penalty": lateral_penalty
    }
    return float(total_reward), components