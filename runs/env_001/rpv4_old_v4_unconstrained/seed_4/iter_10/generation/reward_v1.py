def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observations: [x, y, vx, vy, angle, angular_vel, left_contact, right_contact]
    x, y, vx, vy, angle, av, left_c, right_c = obs
    nx, ny, nvx, nvy, nangle, nav, nleft_c, nright_c = next_obs

    # --- A: progress based on distance squared to target ---
    d_sq_obs = x**2 + y**2
    d_sq_next = nx**2 + ny**2
    progress = 2.0 * (d_sq_obs - d_sq_next)   # positive if moving toward (0,0)

    # --- B: light stability penalty on attitude and velocities ---
    stability = (
        -0.02 * abs(nangle)
        - 0.01 * abs(nav)
        - 0.02 * abs(nvx)
        - 0.01 * abs(nvy)
    )

    # --- C: soft contact bonus (two‑leg landing, near ground, low vertical speed) ---
    proximity = 1.0 / (1.0 + abs(ny))
    low_speed_gate = 1.0 / (1.0 + abs(nvy))
    both_legs = min(nleft_c, nright_c)          # 1.0 only when both legs touch
    contact_bonus = 1.0 * both_legs * proximity * low_speed_gate

    # --- D: tiny fuel/action cost ---
    action_penalty = -0.01 if action != 0 else 0.0

    total_reward = progress + stability + contact_bonus + action_penalty

    components = {
        "progress_delta": progress,
        "stability_penalty": stability,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty
    }
    return total_reward, components