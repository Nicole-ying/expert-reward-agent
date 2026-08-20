```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Velocity-corridor reward for 2D lander.

    Core idea: instead of a narrow proximity-gated velocity penalty, shape the
    entire descent with a *target-speed corridor*.  Far from the pad, faster
    descent is rewarded; near the pad, only slow speeds are rewarded.  This
    creates a continuous, learnable gradient from "descending" through
    "decelerating" to "soft contact."

    Additional signals: potential-based approach, broadened terminal success,
    horizontal drift penalty, orientation cost, contact/landing quality, fuel.
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
    # 1. Potential-based approach shaping
    #    Bounded total = w_app * initial_distance.  Cannot be exploited by
    #    oscillating because going away gives symmetric negative reward.
    # ------------------------------------------------------------------
    w_app = 5.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Speed corridor — the central innovation
    #    target_speed(d) = v_max * (1 - exp(-d / d_scale))
    #    At d=0:  target=0       → must be nearly stationary at the pad
    #    At d=2:  target≈1.65    → moderate speed
    #    At d=6:  target≈2.82    → faster descent OK far away
    #    At d=15: target≈2.99    → essentially unconstrained
    #
    #    Penalising |speed - target_speed| teaches *progressive deceleration*
    #    across the whole descent, not just the last 0.5 units.
    # ------------------------------------------------------------------
    v_max = 3.0
    d_scale = 2.5
    target_speed = v_max * (1.0 - 2.718281828 ** (-dist / d_scale))
    speed_error = abs(speed - target_speed)
    w_speed = 1.2
    speed_shaping = -w_speed * speed_error

    # ------------------------------------------------------------------
    # 3. Horizontal drift penalty — always undesirable
    #    Separated from vertical so the corridor only governs descent rate,
    #    while lateral motion is penalised uniformly.
    # ------------------------------------------------------------------
    w_drift = 0.4
    drift_penalty = -w_drift * abs(vx)

    # ------------------------------------------------------------------
    # 4. Orientation cost — keep the lander upright
    # ------------------------------------------------------------------
    w_angle = 0.8
    w_angvel = 0.2
    orientation_cost = -w_angle * abs(angle) - w_angvel * abs(angvel)

    # ------------------------------------------------------------------
    # 5. Per-foot contact bonus — encourage ground contact
    # ------------------------------------------------------------------
    w_contact = 2.0
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 6. Landing quality — continuous bonus while both feet are down
    # ------------------------------------------------------------------
    both = left * right
    w_land = 5.0
    landing_quality = w_land * both / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 7. Terminal success bonus — broadened and quality-scaled
    #    Fires on the transition to both-feet-down, with a wider near_pad
    #    radius (1.2 instead of 0.5) and continuous quality scaling so the
    #    agent gets partial credit for imperfect landings and can improve.
    # ------------------------------------------------------------------
    prev_both = obs[6] * obs[7]
    now_both = left * right
    just_landed = float(now_both > 0.5 and prev_both < 0.5)
    near_pad = float(dist < 1.2)
    land_quality = 1.0 / (1.0 + abs(angle) + speed)
    w_success = 120.0
    success_bonus = w_success * just_landed * near_pad * land_quality

    # ------------------------------------------------------------------
    # 8. Fuel efficiency
    # ------------------------------------------------------------------
    w_fuel = 0.04
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + speed_shaping + drift_penalty +
        orientation_cost + contact_bonus + landing_quality +
        success_bonus + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "speed_shaping": speed_shaping,
        "drift_penalty": drift_penalty,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "success_bonus": success_bonus,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components
```