def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Two-tier speed penalty + strong approach shaping for 2D lander.

    Replaces the failed speed-corridor with:
    - Gentle universal speed cost (encourages efficient descent)
    - Proximity-gated near-ground damping (enforces soft landing)
    Combined with a boosted potential-based approach reward as the primary
    descent driver, and a large terminal success bonus.
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
    # 1. Potential-based approach shaping — PRIMARY descent driver.
    #    Boosted from 5.0 to 8.0 so it can overcome penalty noise.
    #    Total per episode capped at w_app * initial_distance (unexploitable).
    # ------------------------------------------------------------------
    w_app = 8.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Gentle universal speed cost — discourages reckless speed.
    #    Same form as the successful iteration 2 (score +40).
    #    At speed=2.0 this is only -0.2/step — does not dominate.
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 3. Proximity-gated near-ground speed damping.
    #    gate = 1/(1+2*dist): at dist=0 gate=1.0, at dist=2 gate=0.2,
    #    at dist=5 gate=0.09.  Strong near pad, negligible far away.
    #    This replaces the speed corridor with a one-sided penalty that
    #    only asks the agent to slow down when it matters.
    # ------------------------------------------------------------------
    near_gate = 1.0 / (1.0 + 2.0 * dist)
    w_near = 1.0
    near_speed_penalty = -w_near * near_gate * speed

    # ------------------------------------------------------------------
    # 4. Horizontal drift penalty — always undesirable.
    # ------------------------------------------------------------------
    w_drift = 0.4
    drift_penalty = -w_drift * abs(vx)

    # ------------------------------------------------------------------
    # 5. Orientation cost — keep the lander upright.
    # ------------------------------------------------------------------
    w_angle = 0.8
    w_angvel = 0.3
    orientation_cost = -w_angle * abs(angle) - w_angvel * abs(angvel)

    # ------------------------------------------------------------------
    # 6. Per-foot contact bonus — encourage ground contact.
    # ------------------------------------------------------------------
    w_contact = 2.0
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 7. Landing quality — continuous bonus while both feet are down.
    # ------------------------------------------------------------------
    both = left * right
    w_land = 6.0
    landing_quality = w_land * both / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 8. Terminal success bonus — one-shot, quality-scaled.
    #    Wider near_pad (1.5 vs old 1.2) and bigger weight (200 vs 120).
    #    Makes landing discovery easier and more rewarding.
    # ------------------------------------------------------------------
    prev_both = obs[6] * obs[7]
    now_both = left * right
    just_landed = float(now_both > 0.5 and prev_both < 0.5)
    near_pad = float(dist < 1.5)
    land_quality = 1.0 / (1.0 + abs(angle) + speed)
    w_success = 200.0
    success_bonus = w_success * just_landed * near_pad * land_quality

    # ------------------------------------------------------------------
    # 9. Fuel efficiency — light pressure against unnecessary engine use.
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + vel_cost + near_speed_penalty +
        drift_penalty + orientation_cost + contact_bonus +
        landing_quality + success_bonus + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "vel_cost": vel_cost,
        "near_speed_penalty": near_speed_penalty,
        "drift_penalty": drift_penalty,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "success_bonus": success_bonus,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components