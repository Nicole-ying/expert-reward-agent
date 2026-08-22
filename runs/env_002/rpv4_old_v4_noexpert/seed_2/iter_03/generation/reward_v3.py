def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- forward progress (primary signal) ---
    # Desaturate the normalized speed curve: larger speed_scale pushes the
    # saturation point to higher speeds, keeping gradient alive for faster gaits.
    # forward_weight is raised together so the reward magnitude at current
    # speed (~0.4) stays roughly unchanged.
    forward_weight = 3.0
    speed_scale = 3.0
    forward_speed = 0.5 * (obs[2] + next_obs[2])
    forward_progress_reward = (
        forward_weight
        * forward_speed
        / ((forward_speed ** 2 + speed_scale ** 2) ** 0.5)
    )

    # --- upright stability (health constraint) ---
    # Strengthen the angle penalty: at ~1.0 rad tilt the old weight (1.0) only
    # cost ~0.25/step, less than one forward step, so falls were cheap.
    angle_deadband = 0.5
    angvel_deadband = 1.5
    angle_weight = 5.0
    angvel_weight = 0.05
    eps = 1e-6

    avg_angle = 0.5 * (obs[0] + next_obs[0])
    avg_angvel = 0.5 * (obs[1] + next_obs[1])

    angle_raw = abs(avg_angle) - angle_deadband
    angle_excess = 0.5 * (angle_raw + (angle_raw ** 2 + eps) ** 0.5)
    angle_penalty = angle_weight * angle_excess ** 2

    angvel_raw = abs(avg_angvel) - angvel_deadband
    angvel_excess = 0.5 * (angvel_raw + (angvel_raw ** 2 + eps) ** 0.5)
    angvel_penalty = angvel_weight * angvel_excess ** 2

    upright_stability_penalty = -(angle_penalty + angvel_penalty)

    # energy penalty removed: it shortened episodes (1092 -> 803) and cut
    # forward integration without producing a more efficient gait.

    total_reward = forward_progress_reward + upright_stability_penalty

    components = {
        "forward_progress_reward": float(forward_progress_reward),
        "upright_stability_penalty": float(upright_stability_penalty),
    }

    return float(total_reward), components