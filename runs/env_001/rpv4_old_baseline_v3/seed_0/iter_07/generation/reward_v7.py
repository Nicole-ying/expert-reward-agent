def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Return to iteration-2 architecture (score +40, 850-step survival) with
    enhancements: terminal success bonus, boosted contact/landing rewards.

    Core principle: positive shaping drives descent; penalties are kept
    gentle so the agent survives long enough to discover landing.
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
    # 1. Potential-based approach — primary descent driver.
    #    Bounded total = w_app * initial_distance.  Unexploitable.
    #    Weight 5.0 (same as iteration 2, proven safe).
    # ------------------------------------------------------------------
    w_app = 5.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Small proximity reward — continuous positive signal near pad.
    #    w_prox / (1+dist):  at dist=0 → 0.05,  at dist=5 → 0.008.
    #    Same formula as iteration 2 where it contributed +33 over 850 steps.
    #    NOT the exponential form from iteration 3 that caused hovering.
    # ------------------------------------------------------------------
    w_prox = 0.05
    proximity = w_prox / (1.0 + dist)

    # ------------------------------------------------------------------
    # 3. Gentle universal speed cost — discourages reckless speed.
    #    Same as iteration 2: -0.1 * speed.  At speed=3.0 this is -0.3/step.
    #    Survived 850 steps with this in iteration 2.
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 4. Quadratic orientation cost — keep lander upright.
    #    angle**2 is gentler than abs(angle) for small tilts, still
    #    punishes large deviations.  Same as iteration 2.
    # ------------------------------------------------------------------
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ------------------------------------------------------------------
    # 5. Per-foot contact bonus — encourage ground contact.
    #    Boosted to 1.0/foot (from 0.5 in iteration 2).
    # ------------------------------------------------------------------
    w_contact = 1.0
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 6. Landing quality — continuous bonus while both feet are down
    #    and the lander is stable.  Boosted to 5.0 (from 2.0 in iter 2).
    #    Perfect landing (angle=0, speed=0): 5.0/step.
    # ------------------------------------------------------------------
    both = left * right
    w_land = 5.0
    landing_quality = w_land * both / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 7. Terminal success bonus — one-shot, quality-scaled.
    #    Fires on the transition to both-feet-down, near the pad.
    #    Wider radius (1.5) for easier discovery; continuous quality
    #    scaling so imperfect landings get partial credit.
    # ------------------------------------------------------------------
    prev_both = obs[6] * obs[7]
    now_both = left * right
    just_landed = float(now_both > 0.5 and prev_both < 0.5)
    near_pad = float(dist < 1.5)
    land_quality = 1.0 / (1.0 + abs(angle) + speed)
    w_success = 200.0
    success_bonus = w_success * just_landed * near_pad * land_quality

    # ------------------------------------------------------------------
    # 8. Fuel efficiency — light pressure against unnecessary engine use.
    #    Same as iteration 2.
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + proximity + vel_cost + orientation_cost +
        contact_bonus + landing_quality + success_bonus + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "proximity": proximity,
        "vel_cost": vel_cost,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "success_bonus": success_bonus,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components