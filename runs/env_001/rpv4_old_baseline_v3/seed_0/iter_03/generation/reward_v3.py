def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: exponential proximity attraction as primary signal,
    strong potential-based approach shaping, proximity-gated velocity damping,
    relaxed orientation penalty, and moderate contact/landing bonuses.
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # 1. Exponential proximity: primary continuous attractor toward pad center
    # Range: ~1.5 at pad (dist=0), ~0.55 at dist=2.5, ~0.07 at dist=7.5, ~0.0005 at dist=20
    w_proximity = 1.5
    dist_scale = 2.5
    proximity_reward = w_proximity * (2.718281828 ** (-dist / dist_scale))

    # 2. Potential-based approach: strong directional shaping
    # Rewards every step that reduces distance to pad
    w_approach = 20.0
    approach_reward = w_approach * (dist_prev - dist)

    # 3. Proximity-gated velocity damping: penalise speed only near the pad
    # Far away: allow free movement for efficient descent. Near pad: enforce soft landing.
    near_pad = 2.718281828 ** (-dist / 1.5)
    w_vel_near = 1.0
    w_vel_far = 0.02
    vel_penalty = -(w_vel_near * near_pad + w_vel_far) * (abs(vx) + abs(vy))

    # 4. Orientation cost: relaxed to allow necessary tilting for horizontal control
    w_angle = 0.3
    w_angvel = 0.05
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 5. Per-foot contact bonus: moderate reward for ground contact
    w_contact = 0.5
    contact_bonus = w_contact * (left_contact + right_contact)

    # 6. Landing quality: bonus for stable two-foot contact
    both_feet = left_contact * right_contact
    w_landing = 2.0
    landing_quality = w_landing * both_feet / (1.0 + abs(angle) + speed)

    # 7. Fuel efficiency: light penalty for engine use
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + proximity_reward + vel_penalty +
        orientation_cost + contact_bonus + landing_quality + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "proximity_reward": proximity_reward,
        "vel_penalty": vel_penalty,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components