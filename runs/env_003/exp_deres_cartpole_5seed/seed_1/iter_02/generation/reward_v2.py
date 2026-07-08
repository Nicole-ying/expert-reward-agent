def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward.
    Uses:
    - progress_reward: dense improvement towards stable upright and center.
    - stability_penalty: light penalty on linear/angular speeds.
    
    Iter 2 change: reduced stability_penalty weights 10x (0.01→0.001)
    because penalty ratio_to_progress was -3.16, drowning out the progress signal.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.1

    # Reduced 10x from 0.01 → 0.001 to fix penalty dominance (ratio was -3.16)
    w_vel = 0.001
    w_angvel = 0.001

    # -- cost helper --
    def cost(o):
        return w_pos * abs(o[0]) + w_angle * abs(o[2])

    # -- main progress reward --
    progress_reward = scale_progress * (cost(obs) - cost(next_obs))

    # -- light stability penalty (only on next step) --
    stability_penalty = -w_vel * abs(next_obs[1]) - w_angvel * abs(next_obs[3])

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components