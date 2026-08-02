def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    return float(original_reward), {"original": float(original_reward)}
