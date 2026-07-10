def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward v1.
    
    Goal: Walk forward as far and as fast as possible while staying upright.
    Components:
    1. forward_reward   - reward for positive horizontal velocity
    2. upright_penalty  - penalty for body tilt (hull_angle)
    3. action_cost      - tiny penalty for joint torque usage (energy efficiency)
    """
    # Extract observation slices
    hull_angle = obs[0]               # body tilt angle (0 = upright)
    horizontal_velocity = obs[2]      # forward velocity (>0 = forward)

    # Main learning signal: reward forward progress
    forward_reward = max(horizontal_velocity, 0.0)    # no reward for moving backward

    # Stability constraint: penalise deviation from upright
    upright_penalty = abs(hull_angle)

    # Energy efficiency: small penalty on squared torques
    action_cost = sum(a * a for a in action)

    # Weights
    w_vel    = 1.0
    w_upright = 0.5
    w_action  = 0.01

    total_reward = w_vel * forward_reward - w_upright * upright_penalty - w_action * action_cost

    components = {
        "forward_reward": forward_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components