def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Survival balance task (Env_003).
    Use potential-based shaping as the main progress signal.
    """
    # Constants
    GAMMA = 0.99
    ANGLE_LIMIT = 0.20943951   # radians, pole falls if exceeded
    POS_LIMIT = 2.4            # base goes out of bounds if exceeded

    # Current and next states
    pos_now, _, angle_now, _ = obs[0], obs[1], obs[2], obs[3]
    pos_next, _, angle_next, _ = next_obs[0], next_obs[1], next_obs[2], next_obs[3]

    # Potential function: encourages pole upright and base centered
    # Phi = - ( (angle/limit)^2 + (position/limit)^2 )
    phi_now = -((angle_now / ANGLE_LIMIT) ** 2 + (pos_now / POS_LIMIT) ** 2)
    phi_next = -((angle_next / ANGLE_LIMIT) ** 2 + (pos_next / POS_LIMIT) ** 2)

    # Shaping reward
    progress_reward = GAMMA * phi_next - phi_now

    total_reward = progress_reward
    components = {
        "progress_reward": progress_reward
    }

    return float(total_reward), components