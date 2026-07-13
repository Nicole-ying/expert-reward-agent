def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox = 10.0      # distance reduction weight
    w_contact = 25.0   # contact bonus weight
    k_vel = 0.5        # velocity gate slope
    k_angle = 2.0      # angle gate slope

    # ---------- 1. distance progress ----------
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = prev_dist - next_dist

    # ---------- 2. multiplicative gate (based on speed and tilt) ----------
    speed = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    angle = abs(next_obs[4])
    gate = 1.0 / (1.0 + k_vel * speed + k_angle * angle)

    # ---------- 3. contact bonus for stable landing ----------
    # both support legs in contact
    contact = next_obs[6] * next_obs[7]

    # ---------- reward composition ----------
    progress_reward = w_prox * progress * gate
    contact_reward = w_contact * contact * gate

    total_reward = progress_reward + contact_reward

    components = {
        'progress': progress_reward,
        'contact_reward': contact_reward,
    }
    return float(total_reward), components