def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next observation
    forward_vel = next_obs[5]    # forward velocity of the torso
    torso_angle = next_obs[1]    # torso tilt angle
    torso_angvel = next_obs[7]   # torso angular velocity

    # ---------- Main progress signal ----------
    # Encourage sustained forward motion. This is the primary task driver.
    w_forward = 2.0
    forward_reward = w_forward * forward_vel

    # ---------- Stability constraint ----------
    # Penalise large torso tilt and high angular velocity to keep the body upright
    # and avoid rapid tumbling.  Squared terms create a soft, continuous penalty.
    w_angle = 0.5
    w_angvel = 0.3
    upright_penalty = -w_angle * (torso_angle ** 2) - w_angvel * (torso_angvel ** 2)

    # ---------- Total reward ----------
    total_reward = forward_reward + upright_penalty

    components = {
        "forward_reward": forward_reward,
        "upright_penalty": upright_penalty
    }
    return float(total_reward), components