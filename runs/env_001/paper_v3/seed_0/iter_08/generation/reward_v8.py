def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: penalizes high speed when near target, gated by distance
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: geometric mean of bounded factors rewarding simultaneous
      proximity, low speed, upright posture, and ground contact;
      joint satisfaction required (all factors must be good, no compensation)
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

    # Component D: landing quality via geometric mean (joint satisfaction)
    w_landing = 8.0
    # proximity factor: 1 at dist=0, 0 when dist >= 2.5
    prox_factor = max(0.0, 1.0 - dist / 2.5)
    # speed factor: 1 at speed=0, 0 when speed >= 2.0 m/s
    speed = (vx**2 + vy**2) ** 0.5
    vel_factor = max(0.0, 1.0 - speed / 2.0)
    # angle factor: 1 at angle=0, 0 when |angle| >= 0.5 rad
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.5)
    # contact factor: 0.0 (none), 0.5 (single), 1.0 (double contact)
    contact_factor = (left_contact + right_contact) / 2.0

    # Geometric mean enforces joint satisfaction across all four dimensions.
    # If any factor is 0, the entire term is 0 -- no compensation allowed.
    # Using exponent 0.25 (fourth root) prevents the collapse of a raw product
    # while retaining the AND-like pressure on all dimensions.
    landing_quality = (prox_factor * vel_factor * angle_factor * contact_factor) ** 0.25
    landing_proxy = w_landing * landing_quality

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