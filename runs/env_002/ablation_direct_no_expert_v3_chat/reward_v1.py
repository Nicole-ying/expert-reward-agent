def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant observations
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    horizontal_velocity = obs[2]
    vertical_velocity = obs[3]
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    
    # ========== Component 1: Forward velocity reward (main learning signal) ==========
    # Encourage fast forward movement. Use a linear reward capped at a reasonable speed.
    # This provides a strong, continuous gradient for the primary task.
    target_speed = 2.0  # m/s, reasonable walking speed for a biped
    forward_reward = min(horizontal_velocity / target_speed, 1.0)  # normalized to [0, 1]
    
    # ========== Component 2: Upright posture penalty (stability constraint) ==========
    # Penalize deviation from upright posture. Use a quadratic penalty to strongly discourage
    # falling over while allowing small natural sway during walking.
    max_allowed_angle = 0.5  # radians (~28 degrees)
    angle_penalty = (hull_angle / max_allowed_angle) ** 2  # quadratic penalty
    
    # ========== Component 3: Ground contact bonus (gait quality) ==========
    # Reward alternating ground contact to encourage proper gait cycle.
    # Having at least one foot on ground is essential for stability.
    contact_bonus = 0.5 * (leg1_contact + leg2_contact)  # [0, 1], higher when both feet contact
    
    # ========== Component 4: Energy penalty (efficiency constraint) ==========
    # Penalize large joint torques to discourage wasteful flailing.
    # Use squared action norm for smoothness and energy efficiency.
    energy_penalty = 0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)
    
    # ========== Combine components ==========
    # Weights chosen to balance the components:
    # - forward_reward: primary signal, weight 2.0
    # - angle_penalty: critical for survival, weight 1.0 (subtracted)
    # - contact_bonus: auxiliary signal, weight 0.5
    # - energy_penalty: small penalty, weight 0.01 (already scaled)
    total_reward = 2.0 * forward_reward - 1.0 * angle_penalty + 0.5 * contact_bonus - energy_penalty
    
    components = {
        'forward_reward': forward_reward,
        'angle_penalty': angle_penalty,
        'contact_bonus': contact_bonus,
        'energy_penalty': energy_penalty
    }
    
    return float(total_reward), components