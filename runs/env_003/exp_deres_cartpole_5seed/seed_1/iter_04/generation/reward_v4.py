def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward — v4.

    Diagnosis (iter 3): potential-based progress_reward collapsed to near-zero
    at steady state (mean -0.0003).  survival_bonus (0.005, constant) dominated
    total but provides zero gradient.  Agent solved via env's own 1.0/step,
    not via our shaped reward.

    Change (v4): switch progress_reward from potential-based shaping
      scale * (cost(obs) - cost(next_obs))
    to bounded proximity:
      scale * (w_angle * 1/(1+10*|angle|) + w_pos * 1/(1+5*|pos|))

    Bounded proximity provides gradient everywhere — even at the optimum,
    there is always incentive to get closer.  Naturally bounded in [0,1]
    per factor, preventing numerical issues.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.01   # calibrated for bounded proximity range ~0.07-0.11

    # unchanged from iter 2/3
    w_vel = 0.001
    w_angvel = 0.001
    survival_bonus = 0.005

    # -- bounded proximity progress reward --
    # Each factor ∈ (0, 1]; product with weight gives max w_angle at angle=0
    angle_proximity = 1.0 / (1.0 + 10.0 * abs(next_obs[2]))
    pos_proximity   = 1.0 / (1.0 + 5.0 * abs(next_obs[0]))
    progress_reward = scale_progress * (w_angle * angle_proximity + w_pos * pos_proximity)

    # -- light stability penalty (unchanged) --
    stability_penalty = -w_vel * abs(next_obs[1]) - w_angvel * abs(next_obs[3])

    # -- total --
    total_reward = progress_reward + stability_penalty + survival_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "survival_bonus": survival_bonus,
    }

    return float(total_reward), components