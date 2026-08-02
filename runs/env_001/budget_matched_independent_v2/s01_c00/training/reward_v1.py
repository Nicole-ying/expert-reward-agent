def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs (post-action state)
    x_pos = next_obs[0]       # horizontal position relative to target
    y_pos = next_obs[1]       # vertical position relative to pad height
    x_vel = next_obs[2]       # horizontal velocity
    y_vel = next_obs[3]       # vertical velocity
    angle = next_obs[4]       # body orientation angle
    ang_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]  # left support contact flag (0 or 1)
    right_contact = next_obs[7] # right support contact flag (0 or 1)

    # Distance to target (Euclidean distance in position space)
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5

    # Speed magnitude
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5

    # --- Reward components ---

    # 1. Proximity reward: encourage getting close to target
    # Exponential decay with distance, scaled to be in [0, 1]
    proximity_reward = 2.718281828 ** (-2.0 * distance)

    # 2. Velocity penalty: penalize high speed, especially when close to target
    # Use a distance-weighted penalty: when far, allow some speed; when close, penalize more
    velocity_penalty = -0.5 * speed * (1.0 + 2.0 * (2.718281828 ** (-1.0 * distance)))

    # 3. Orientation reward: encourage upright orientation (angle near 0)
    # Penalize deviation from vertical, using cosine-like smooth penalty
    orientation_penalty = -0.3 * (angle ** 2)

    # 4. Angular velocity penalty: discourage spinning
    angular_penalty = -0.2 * (ang_vel ** 2)

    # 5. Contact bonus: reward stable contact with both supports on the pad
    # Both contacts active indicates settled state
    contact_bonus = 0.5 * (left_contact * right_contact)

    # 6. Action penalty: penalize engine usage to encourage fuel efficiency
    # action 0 = no engine (no penalty), actions 1-3 = engine firing
    action_penalty = -0.1 if action != 0 else 0.0

    # 7. Landing completion bonus: large reward when settled on target with both contacts
    # Condition: very close to target, low speed, both contacts active
    is_settled = (distance < 0.1) and (speed < 0.1) and (left_contact > 0.5) and (right_contact > 0.5)
    completion_bonus = 10.0 if is_settled else 0.0

    # Sum all components
    total_reward = (proximity_reward + velocity_penalty + orientation_penalty +
                    angular_penalty + contact_bonus + action_penalty + completion_bonus)

    components = {
        "proximity_reward": proximity_reward,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "angular_penalty": angular_penalty,
        "contact_bonus": contact_bonus,
        "action_penalty": action_penalty,
        "completion_bonus": completion_bonus,
    }

    return float(total_reward), components