def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1: Materially different approach — a multi-objective proximity reward that directly optimizes
    horizontal/vertical approach + velocity damping near the pad, with a contact-conditioned
    landing proxy. Focuses on continuous gradients instead of multiplicative threshold products.
    """
    
    # ── Extract state ────────────────────────────────────────────────────────
    # Current (for delta computations)
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    left_contact, right_contact = obs[6], obs[7]
    
    # Next state
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # Derived signals
    both_feet_contact = 1.0 if (next_left_contact > 0.5 and next_right_contact > 0.5) else 0.0
    
    # ── Component 1: Multi-objective proximity + velocity damping (main learning signal) ──
    # Penetrate toward target (0,0) in both axes, penalize speed when close
    horizontal_proximity = -abs(next_x_pos)          # closer to 0 is better
    vertical_proximity = -abs(next_y_pos)           # closer to 0 is better (pad height)
    
    # Velocity damping: penalize speed proportional to proximity to target
    horizontal_damping = -abs(next_x_vel) * (1.0 - min(abs(next_x_pos), 1.0))
    vertical_damping = -abs(next_y_vel) * (1.0 - min(abs(next_y_pos), 1.0))
    
    # Combine: main signal drives toward (0,0) while encouraging low speed near target
    approach_reward = 2.0 * horizontal_proximity + 1.0 * vertical_proximity + \
                      3.0 * horizontal_damping + 2.0 * vertical_damping
    
    # ── Component 2: Attitude stability (soft constraint) ─────────────────────
    # Penalize large body angles, encouraging near-upright orientation
    angle_penalty = -abs(next_body_angle)
    
    # ── Component 3: Landing proxy (sparse + continuous) ──────────────────────
    # Only active when both feet are in contact, encourages position/speed within tolerance
    distance_from_center = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    speed_magnitude = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    
    # Continuous bounty: larger when feet contact and close/stationary near (0,0)
    landing_bounty = both_feet_contact * (2.0 - distance_from_center - speed_magnitude)
    landing_bounty = max(0.0, landing_bounty) * 5.0  # scale up meaningful values
    
    # ── Total reward ──────────────────────────────────────────────────────────
    total_reward = approach_reward + angle_penalty + landing_bounty
    
    # ── Components dict (only top-level additive terms) ───────────────────────
    components = {
        'approach_reward': approach_reward,
        'angle_penalty': angle_penalty,
        'landing_bounty': landing_bounty
    }
    
    return float(total_reward), components