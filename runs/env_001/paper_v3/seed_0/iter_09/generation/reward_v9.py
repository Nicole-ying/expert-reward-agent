def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: penalizes high speed when near target, gated by distance
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: geometric mean of 3 continuous factors (proximity, low speed,
      upright posture) providing dense approach gradient; plus additive contact bonus
      scaled by approach quality to incentivize final touchdown without collapsing
      the approach signal
    """
    # next_obs unpacking
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Euclidean distance to target
    dist = (x**2 + y**2) ** 0.5

    # Component A: distance penalty (core progress signal)
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # Component B: velocity penalty (damped by distance to target)
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: landing proxy with decoupled approach quality and contact bonus
    # Bounded factors for continuous dimensions
    prox_factor = max(0.0, 1.0 - dist / 2.5)
    speed = (vx**2 + vy**2) ** 0.5
    vel_factor = max(0.0, 1.0 - speed / 2.0)
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    # Geometric mean of 3 continuous factors: enforces joint satisfaction
    # of proximity, low speed, and upright posture during approach.
    # Unlike the 4-factor version, this does NOT collapse when airborne
    # (contact_factor was the zero-killer for 94% of steps).
    approach_quality = (prox_factor * vel_factor * angle_factor) ** (1.0 / 3.0)

    # Contact factor: 0.0 (none), 0.5 (single), 1.0 (double)
    contact_factor = (left_contact + right_contact) / 2.0

    # Contact bonus: additive extra reward when touching ground,
    # scaled by approach_quality so contact only pays off when
    # the vehicle is also close, slow, and upright.
    contact_bonus = contact_factor * approach_quality

    w_approach = 5.0
    w_contact = 3.0
    landing_proxy = w_approach * approach_quality + w_contact * contact_bonus

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components