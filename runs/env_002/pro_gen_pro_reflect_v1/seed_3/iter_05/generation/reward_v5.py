def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Main learning signal: forward velocity, gated by posture health
    forward_vel = next_obs[2]                 # positive = moving forward
    hull_angle = next_obs[0]                  # radians, 0 = vertical
    health = 1.0 / (1.0 + 10.0 * (hull_angle ** 2))   # posture gate
    forward_progress = 1.0 * forward_vel * health

    # Efficiency: light action cost
    avg_action_power = (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2) / 4.0
    action_cost = -0.05 * avg_action_power

    total_reward = forward_progress + action_cost

    components = {
        'forward_progress': forward_progress,
        'action_cost': action_cost
    }
    return float(total_reward), components