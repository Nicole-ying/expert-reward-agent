def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for Env_003 (CartPole-like survival balance task).

    DIAGNOSIS: Task already solved (500/500, full survival). The only refinement
    is changing stability_penalty from abs() to squared form for smoother gradients
    (abs has discontinuous subgradient at zero). Coefficient adjusted 0.01→0.03
    to maintain similar magnitude.

    - progress_reward: baseline - (w_angle * angle^2 + w_pos * pos^2)
      Encourages keeping the pole near upright and the cart near the center.
    - stability_penalty: light squared penalty on velocities for smooth gradients.
    """
    w_angle = 5.0
    w_pos = 0.1
    baseline = 1.0

    lambda_vel = 0.03
    lambda_angvel = 0.03

    pos = next_obs[0]
    vel = next_obs[1]
    angle = next_obs[2]
    angvel = next_obs[3]

    progress_reward = baseline - (w_angle * (angle ** 2) + w_pos * (pos ** 2))
    stability_penalty = -lambda_vel * (vel ** 2) - lambda_angvel * (angvel ** 2)

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
    }

    return float(total_reward), components