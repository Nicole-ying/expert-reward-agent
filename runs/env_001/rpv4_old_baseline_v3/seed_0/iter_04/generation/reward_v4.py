def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: potential-based approach shaping as primary driver,
    one-shot landing success bonus for terminal incentive, proximity-gated
    velocity damping, and supporting contact/landing-quality signals.
    No continuous proximity reward — prevents hovering exploit.
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_left = obs[6]
    prev_right = obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # 1. Potential-based approach shaping — primary driver toward the pad.
    # Bounded: total per episode = w_approach * initial_distance. Cannot be exploited.
    w_approach = 10.0
    approach_reward = w_approach * (dist_prev - dist)

    # 2. One-shot landing success bonus.
    # Fires only on the transition from not-landed to landed, near the pad,
    # with low speed and upright attitude. Gives a large terminal incentive
    # that exceeds any possible cumulative reward from hovering.
    prev_both = prev_left * prev_right
    now_both = left_contact * right_contact
    just_landed = float(now_both > 0.5 and prev_both < 0.5)
    near_pad = float(dist < 0.5)
    soft_landing = float(speed < 1.5)
    upright = float(abs(angle) < 0.5)
    w_success = 200.0
    success_bonus = w_success * just_landed * near_pad * soft_landing * upright

    # 3. Landing quality — continuous reward while settled on both feet.
    # Rewards stability: low speed + upright + both feet down.
    w_landing = 5.0
    landing_quality = w_landing * (left_contact * right_contact) / (1.0 + abs(angle) + speed)

    # 4. Proximity-gated velocity damping.
    # Strong near the pad (enforces soft landing), negligible far away.
    near_gate = 1.0 / (1.0 + 5.0 * dist)
    w_vel = 0.5
    vel_penalty = -w_vel * near_gate * speed

    # 5. Orientation cost — keep the lander upright.
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * abs(angle) - w_angvel * abs(angvel)

    # 6. Per-foot contact bonus — encourage ground contact.
    w_contact = 1.0
    contact_bonus = w_contact * (left_contact + right_contact)

    # 7. Fuel efficiency — light pressure against unnecessary engine use.
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + success_bonus + landing_quality +
        vel_penalty + orientation_cost + contact_bonus + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "success_bonus": success_bonus,
        "landing_quality": landing_quality,
        "vel_penalty": vel_penalty,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components