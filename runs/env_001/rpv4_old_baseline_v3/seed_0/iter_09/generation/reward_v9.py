def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Iter-2 skeleton restored (proven: score +40, 6/20 landings).
    + controlled-descent incentive (speed-gated to prevent plummeting)
    + boosted contact/landing weights
    - crash_penalty removed (poisons ground-contact exploration)
    - main_decel_bonus removed (rewards hovering, anti-landing)
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
    # 1. Potential-based approach — ITER 2 WEIGHT (proven safe)
    # ------------------------------------------------------------------
    w_app = 5.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Proximity — ITER 2 WEIGHT
    # ------------------------------------------------------------------
    w_prox = 0.05
    proximity = w_prox / (1.0 + dist)

    # ------------------------------------------------------------------
    # 3. Controlled-descent incentive (NEW — replaces harmful main_decel_bonus)
    #    Rewards descending (y decreasing), gated by speed so fast plummeting
    #    gives much less reward than slow, controlled descent.
    #    At speed=0: full weight; at speed=3: weight/4; at speed=10: weight/11.
    # ------------------------------------------------------------------
    descent = max(0.0, y_prev - y)
    w_desc = 3.0
    descent_reward = w_desc * descent / (1.0 + speed)

    # ------------------------------------------------------------------
    # 4. Velocity cost — ITER 2 WEIGHT
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 5. Orientation cost — ITER 2 WEIGHTS
    # ------------------------------------------------------------------
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ------------------------------------------------------------------
    # 6. Per-foot contact bonus — BOOSTED from iter-2's 0.5 → 2.0
    # ------------------------------------------------------------------
    w_contact = 2.0
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 7. Landing quality — BOOSTED from iter-2's 2.0 → 5.0
    # ------------------------------------------------------------------
    both_feet = left * right
    w_land = 5.0
    landing_quality = w_land * both_feet / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 8. Fuel efficiency — ITER 2 WEIGHT
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + proximity + descent_reward + vel_cost +
        orientation_cost + contact_bonus + landing_quality + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "proximity": proximity,
        "descent_reward": descent_reward,
        "vel_cost": vel_cost,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components