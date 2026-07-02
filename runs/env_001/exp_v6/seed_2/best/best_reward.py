def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Stage-weighted: early focus on proximity, late focus on precision landing.
    No leg-contact checks — avoids terminal_success_reward proxy.
    """
    # --- hyperparameters ---
    k_proximity = 5.0          # decay rate for bounded proximity
    w_proximity_early = 3.0    # proximity weight in early training
    w_quality_late = 4.0       # landing quality weight in late training
    w_stability = 0.02         # stability penalty weight (mild, background)

    # thresholds for continuous landing quality factors
    D_near = 0.5               # distance: full quality at 0, zero beyond 0.5
    V_slow = 0.5               # speed: full quality at 0, zero beyond 0.5 m/s
    A_upright = 0.3            # angle: full quality at 0, zero beyond 0.3 rad

    # --- extract state ---
    dx, dy = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    dist = (dx * dx + dy * dy) ** 0.5
    speed = (vx * vx + vy * vy) ** 0.5

    # --- 1. bounded proximity reward (dense, always active) ---
    proximity = 1.0 / (1.0 + k_proximity * dist)

    # --- 2. continuous landing quality (no leg contacts, purely geometric) ---
    near_factor = max(0.0, 1.0 - dist / D_near)
    slow_factor = max(0.0, 1.0 - speed / V_slow)
    upright_factor = max(0.0, 1.0 - abs(angle) / A_upright)
    landing_quality = near_factor * slow_factor * upright_factor

    # --- 3. mild stability penalty (angular velocity only, to avoid spinning) ---
    stability_penalty = -w_stability * abs(ang_vel)

    # --- stage weighting ---
    # early: proximity dominates → agent learns to approach target
    # late:  landing quality dominates → agent learns precision touchdown
    early_w = max(0.0, 1.0 - training_progress)
    late_w = training_progress

    progress_signal = early_w * w_proximity_early * proximity
    quality_signal = late_w * w_quality_late * landing_quality

    total_reward = progress_signal + quality_signal + stability_penalty

    components = {
        "proximity": proximity,
        "landing_quality": landing_quality,
        "stability_penalty": stability_penalty,
        "progress_signal": progress_signal,
        "quality_signal": quality_signal,
    }

    return float(total_reward), components