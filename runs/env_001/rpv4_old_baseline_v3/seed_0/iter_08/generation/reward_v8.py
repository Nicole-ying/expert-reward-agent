def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Iteration-2 core architecture (proven: score +40→+88, 850-step survival)
    + main-engine deceleration bonus (direct incentive to slow descent)
    + crash penalty (detect hard ground contact)
    - NO success_bonus (lottery destabilizer removed)
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left = next_obs[6]
    right = next_obs[7]

    vy_prev = obs[3]
    prev_left = obs[6]
    prev_right = obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5

    # ------------------------------------------------------------------
    # 1. Potential-based approach — ITERATION 2 WEIGHT (proven safe)
    # ------------------------------------------------------------------
    w_app = 5.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Proximity — ITERATION 2 WEIGHT
    # ------------------------------------------------------------------
    w_prox = 0.05
    proximity = w_prox / (1.0 + dist)

    # ------------------------------------------------------------------
    # 3. Velocity cost — ITERATION 2 WEIGHT
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 4. Orientation cost — ITERATION 2 WEIGHTS
    # ------------------------------------------------------------------
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ------------------------------------------------------------------
    # 5. Per-foot contact bonus — ITERATION 2 WEIGHT
    # ------------------------------------------------------------------
    w_contact = 0.5
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 6. Landing quality — ITERATION 2 WEIGHT
    # ------------------------------------------------------------------
    both_feet = left * right
    w_land = 2.0
    landing_quality = w_land * both_feet / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 7. Fuel efficiency — ITERATION 2 WEIGHT
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    # ------------------------------------------------------------------
    # 8. Main-engine deceleration bonus (NEW — directly rewards slowing descent)
    #    vy - vy_prev > 0  means vertical speed became less negative (decelerating).
    #    Only rewards when action==2 (main engine).  This gives the agent a clear
    #    gradient to discover that main engine controls descent speed.
    # ------------------------------------------------------------------
    decel = vy - vy_prev
    using_main = float(action == 2)
    w_main_decel = 2.0
    main_decel_bonus = w_main_decel * max(0.0, decel) * using_main

    # ------------------------------------------------------------------
    # 9. Crash penalty (NEW — penalises first ground contact at high speed)
    #    Uses pre-contact speed so the penalty reflects impact velocity,
    #    not post-collision-resolved speed.
    # ------------------------------------------------------------------
    prev_any_contact = float(prev_left + prev_right > 0.5)
    cur_any_contact = float(left + right > 0.5)
    just_made_contact = float(cur_any_contact > 0.5 and prev_any_contact < 0.5)
    impact_speed = max(0.0, prev_speed - 1.0)
    w_crash = 25.0
    crash_penalty = -w_crash * just_made_contact * impact_speed

    total_reward = (
        approach_reward + proximity + vel_cost + orientation_cost +
        contact_bonus + landing_quality + fuel_cost +
        main_decel_bonus + crash_penalty
    )

    components = {
        "approach_reward": approach_reward,
        "proximity": proximity,
        "vel_cost": vel_cost,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "fuel_cost": fuel_cost,
        "main_decel_bonus": main_decel_bonus,
        "crash_penalty": crash_penalty,
    }
    return float(total_reward), components