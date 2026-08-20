def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_w = obs[1]
    quat_x = obs[2]
    quat_y = obs[3]
    quat_z = obs[4]
    body_x_velocity = obs[13]
    body_z_velocity = obs[15]

    # ---------- Derived signals ----------
    # Uprightness: body_up_z = 1.0 - 2*(qx^2 + qy^2), range ~[-1, 1]
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Health gate factors ----------
    # Height factor: healthy in (0.25, 0.95), smooth decay outside
    # Below 0.25: linear from 1.0 at z=0.25 down to 0.0 at z=0.15
    height_low_danger = max(0.0, (0.25 - body_z) / 0.10)  # 0.0 when z>=0.25, 1.0 when z<=0.15
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    # Above 0.95: linear from 1.0 at z=0.95 down to 0.0 at z=1.05
    height_high_danger = max(0.0, (body_z - 0.95) / 0.10)
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # Upright factor: healthy when body_up_z > 0.6, linear decay below
    upright_factor = min(1.0, max(0.0, (body_up_z - 0.2) / 0.4))
    # At body_up_z=0.6: (0.6-0.2)/0.4 = 1.0  → gate=1.0
    # At body_up_z=0.2: (0.2-0.2)/0.4 = 0.0  → gate=0.0
    # Smooth gradient throughout

    # Combined health gate: must satisfy both to earn full forward reward
    health_gate = min(height_factor, upright_factor)

    # ---------- Reward components ----------
    # 1) Forward progress gated by health
    forward_progress = max(0.0, float(body_x_velocity))
    reward_forward = 1.0 * forward_progress * health_gate

    # 2) Uprightness hinge penalty (only active when severely tilted)
    # body_up_z < 0.6: penalty grows linearly as tilt increases
    upright_error = max(0.0, 0.6 - body_up_z)
    reward_upright_penalty = -0.5 * upright_error

    # 3) Vertical velocity penalty (discourage bouncing/launching)
    # Soft hinge: penalize |vz| > 1.5
    vz_magnitude = abs(float(body_z_velocity))
    vz_excess = max(0.0, vz_magnitude - 1.5)
    reward_z_velocity_penalty = -0.2 * vz_excess

    # ---------- Total reward ----------
    total_reward = reward_forward + reward_upright_penalty + reward_z_velocity_penalty

    components = {
        "forward_gated": reward_forward,
        "health_gate": health_gate,
        "upright_penalty": reward_upright_penalty,
        "z_velocity_penalty": reward_z_velocity_penalty
    }

    return float(total_reward), components