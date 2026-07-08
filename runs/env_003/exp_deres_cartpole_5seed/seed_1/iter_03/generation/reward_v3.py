def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward — v3.

    Based on iter 2 (score=500, solved).  One change: added a small per-step
    survival_bonus so that the total reward stays slightly positive even when
    the agent is perfectly balanced (previously mean ~ -0.0007).

    Signals:
      - progress_reward: encourages moving towards upright + center.
      - stability_penalty: light damping on velocities to suppress jitter.
      - survival_bonus: constant small positive per step.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.1

    # kept low from iter 2 (0.001) — ratio_to_progress ~ -0.32, healthy
    w_vel = 0.001
    w_angvel = 0.001

    # new in v3: tiny constant to keep total reward positive at steady state
    survival_bonus = 0.005

    # -- cost helper --
    def cost(o):
        return w_pos * abs(o[0]) + w_angle * abs(o[2])

    # -- main progress reward (potential-based shaping) --
    progress_reward = scale_progress * (cost(obs) - cost(next_obs))

    # -- light stability penalty --
    stability_penalty = -w_vel * abs(next_obs[1]) - w_angvel * abs(next_obs[3])

    # -- total --
    total_reward = progress_reward + stability_penalty + survival_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "survival_bonus": survival_bonus,
    }

    return float(total_reward), components