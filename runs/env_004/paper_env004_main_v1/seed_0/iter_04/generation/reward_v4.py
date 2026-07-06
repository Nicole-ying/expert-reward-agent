def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for hopping locomotion: forward progress + vertical push-off + stability.

    Components:
        forward_stability_reward – forward velocity weighted by torso uprightness
        vertical_pushoff         – reward only upward vertical velocity (push-off phase),
                                    gated by uprightness; does not reward downward motion
        stability_penalty        – penalty for torso tilt and angular velocity
    """
    # --- extract relevant signals from next_obs ---
    torso_angle = next_obs[1]          # rad, 0 = upright
    forward_vel = next_obs[5]          # positive = forward
    vertical_vel = next_obs[6]         # vertical velocity (positive = upward)
    torso_ang_vel = next_obs[7]        # rad/s

    # --- shared upright gate ---
    temp = 0.3
    upright_factor = 2.718281828 ** (-abs(torso_angle) / temp)

    # --- forward progress with upright gating ---
    forward_stability_reward = forward_vel * upright_factor

    # --- vertical push-off: reward only the active upward phase of hopping ---
    # max(0, vertical_vel) selects only upward motion (push-off),
    # ignoring the passive downward (landing) phase that gravity handles.
    # This prevents incentivising crash landings that destabilise the agent.
    w_pushoff = 0.15
    vertical_pushoff = w_pushoff * max(0.0, vertical_vel) * upright_factor

    # --- light stability penalty ---
    w_angle = 0.1
    w_ang_vel = 0.01
    stability_penalty = -w_angle * (torso_angle ** 2) - w_ang_vel * (torso_ang_vel ** 2)

    # --- total reward ---
    total_reward = forward_stability_reward + vertical_pushoff + stability_penalty

    components = {
        "forward_stability_reward": forward_stability_reward,
        "vertical_pushoff": vertical_pushoff,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components