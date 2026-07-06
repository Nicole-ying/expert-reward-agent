def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Main learning signal: reward forward velocity only when positive.
    # This drives locomotion, while termination from falling or extreme tilt
    # provides implicit negative feedback against instability.
    forward_vel = obs[5]
    forward_reward = max(0.0, forward_vel)
    total_reward = forward_reward
    components = {
        'forward_reward': forward_reward
    }
    return float(total_reward), components