def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Balanced lander reward: strong approach + aligned-descent gradients,
    moderate contact/landing bonuses (iter-2 weights), stable costs.
    Hover exploit closed: approach=0 when stationary, aligned_descent=0 when not descending.
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left = next_obs[6]
    right = next_obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ------------------------------------------------------------------
    # 1. Strong potential-based approach — doubled from iter-2 (5.0→10.0)
    #    Only rewards movement toward pad; zero when hovering.
    # ------------------------------------------------------------------
    w_app = 10.0
    approach = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Aligned descent — replaces dead descent_reward
    #    Rewards downward motion, gated by horizontal alignment and speed.
    #    Strongest when: directly above pad (x≈0), moving down slowly.
    #    Near-zero when: far horizontally OR moving fast (plummeting).
    # ------------------------------------------------------------------
    descent_amount = max(0.0, y_prev - y)
    w_desc = 8.0
    aligned_descent = w_desc * descent_amount / (1.0 + abs(x) + speed)

    # ------------------------------------------------------------------
    # 3. Proximity — iter-2 weight (0.05), modest hover-prevention
    # ------------------------------------------------------------------
    w_prox = 0.05
    proximity = w_prox / (1.0 + dist)

    # ------------------------------------------------------------------
    # 4. Per-foot contact bonus — iter-2 weight (0.5), not dominant
    # ------------------------------------------------------------------
    w_contact = 0.5
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 5. Landing quality — iter-2 weight (2.0), not dominant
    # ------------------------------------------------------------------
    both_feet = left * right
    w_land = 2.0
    landing_quality = w_land * both_feet / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 6. Velocity cost — iter-2 weight (0.1)
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 7. Orientation cost — iter-2 weights
    # ------------------------------------------------------------------
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ------------------------------------------------------------------
    # 8. Fuel efficiency — iter-2 weight (0.03)
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach + aligned_descent + proximity + contact_bonus +
        landing_quality + vel_cost + orientation_cost + fuel_cost
    )

    components = {
        "approach": approach,
        "aligned_descent": aligned_descent,
        "proximity": proximity,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "vel_cost": vel_cost,
        "orientation_cost": orientation_cost,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components