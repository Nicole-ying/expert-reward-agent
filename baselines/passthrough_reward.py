def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """Official reward pass-through — returns the environment's native reward unchanged."""
    return float(original_reward), {"official_reward": float(original_reward)}
