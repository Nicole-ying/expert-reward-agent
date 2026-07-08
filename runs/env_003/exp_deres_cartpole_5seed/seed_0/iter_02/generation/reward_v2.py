def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Survival balance task (Env_003).
    Potential-based shaping with positive bounded potential function.
    
    Change from iter 1: Φ changed from negative quadratic (-x²) to positive
    bounded form 1/(1+x²). This provides a proper "potential well" with global
    maximum at the goal state, ensuring non-vanishing gradient and removing
    the negative bias at optimum.
    """
    GAMMA = 0.99
    ANGLE_LIMIT = 0.20943951
    POS_LIMIT = 2.4

    pos_now, _, angle_now, _ = obs[0], obs[1], obs[2], obs[3]
    pos_next, _, angle_next, _ = next_obs[0], next_obs[1], next_obs[2], next_obs[3]

    # Positive bounded potential: peaks at 1.0 when pole upright & base centered
    # Φ ∈ [1/3, 1], with maximum at the goal → proper potential well
    phi_now = 1.0 / (1.0 + (angle_now / ANGLE_LIMIT) ** 2 + (pos_now / POS_LIMIT) ** 2)
    phi_next = 1.0 / (1.0 + (angle_next / ANGLE_LIMIT) ** 2 + (pos_next / POS_LIMIT) ** 2)

    # Potential-based shaping: F = γ·Φ(s') - Φ(s)
    progress_reward = GAMMA * phi_next - phi_now

    total_reward = progress_reward
    components = {
        "progress_reward": progress_reward
    }

    return float(total_reward), components