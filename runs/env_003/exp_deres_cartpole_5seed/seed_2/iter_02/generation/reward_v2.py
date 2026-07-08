def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for Env_003 (CartPole-like survival balance task).

    DIAGNOSIS: Agent achieves perfect score (500/500, full 500-step survival).
    No changes needed — reward function is already optimal for this environment.

    - progress_reward: baseline - (w_angle * angle^2 + w_pos * pos^2)
      Encourages keeping the pole near upright and the cart near the center.
    - stability_penalty: negligible background penalty on velocities,
      ratio_to_progress = -0.0037, does not interfere with learning.
    """
    w_angle = 5.0
    w_pos = 0.1
    baseline = 1.0

    lambda_vel = 0.01
    lambda_angvel = 0.01

    pos = next_obs[0]
    vel = next_obs[1]
    angle = next_obs[2]
    angvel = next_obs[3]

    progress_reward = baseline - (w_angle * (angle ** 2) + w_pos * (pos ** 2))
    stability_penalty = -lambda_vel * abs(vel) - lambda_angvel * abs(angvel)

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
    }

    return float(total_reward), components