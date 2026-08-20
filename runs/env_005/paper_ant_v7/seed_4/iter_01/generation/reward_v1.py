def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract key signals from observation
    body_z = obs[0]                # torso height
    qx = obs[2]                    # quaternion x component
    qy = obs[3]                    # quaternion y component
    body_x_vel = obs[13]           # forward (world‑x) velocity

    # ----- uprightness (projection of body z-axis onto world z-axis) -----
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(0.0, min(1.0, body_up_z))   # numerical safety clamp

    # ----- core learning signal: forward velocity modulated by uprightness -----
    w_fwd = 1.0
    forward_reward = w_fwd * body_x_vel * body_up_z

    # ----- safety hinge penalty on torso height -----
    w_h = 0.5
    low_thresh = 0.35
    high_thresh = 0.85
    low_error = max(0.0, low_thresh - body_z)
    high_error = max(0.0, body_z - high_thresh)
    height_penalty = -w_h * (low_error + high_error)

    total_reward = forward_reward + height_penalty

    components = {
        "forward_upright_product": forward_reward,
        "height_survival_penalty": height_penalty
    }

    return float(total_reward), components