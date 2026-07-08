def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D bipedal locomotion on rough terrain.
    Drives forward progress, maintains stable posture, and encourages energy-efficient motion.
    """
    # ---------- forward progress component ----------
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # ---------- stability penalty component ----------
    hull_angle = next_obs[0]
    vertical_velocity = next_obs[3]
    w_angle = 0.5
    w_vertical = 0.1
    stability_penalty = -w_angle * abs(hull_angle) - w_vertical * abs(vertical_velocity)

    # ---------- energy efficiency component ----------
    w_energy = 0.05
    energy_penalty = -w_energy * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- total reward ----------
    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty
    }

    return float(total_reward), components