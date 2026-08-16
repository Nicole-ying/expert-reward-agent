def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    w_progress = 1.0       # weight for horizontal distance reduction
    w_descent = 2.0        # weight for height descent reward
    contact_bonus = 50.0   # one-time reward for first dual-leg contact event
    w_action = 0.01        # action sparsity penalty weight

    # -------------------------------
    # 1. Horizontal progress (distance to target, no health gate)
    # -------------------------------
    dist_cur = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = w_progress * (dist_cur - dist_next)

    # -------------------------------
    # 2. Descent reward (altitude reduction only)
    # -------------------------------
    descent_reward = w_descent * max(0.0, obs[1] - next_obs[1])

    # -------------------------------
    # 3. First dual-leg contact event reward
    # -------------------------------
    prev_contact = (obs[6] > 0.5 and obs[7] > 0.5)
    curr_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    contact_reward = 0.0
    if (not prev_contact) and curr_contact:
        contact_reward = contact_bonus

    # -------------------------------
    # 4. Action sparsity penalty
    # -------------------------------
    action_penalty = -w_action * (1.0 if action != 0 else 0.0)

    # -------------------------------
    # Total reward
    # -------------------------------
    total_reward = progress_reward + descent_reward + contact_reward + action_penalty
    components = {
        "progress_reward": progress_reward,
        "descent_reward": descent_reward,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty
    }
    return float(total_reward), components