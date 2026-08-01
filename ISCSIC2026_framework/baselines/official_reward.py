def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """Expose the unchanged environment reward through CREATE's reward API."""
    return float(original_reward), {"original": float(original_reward)}
