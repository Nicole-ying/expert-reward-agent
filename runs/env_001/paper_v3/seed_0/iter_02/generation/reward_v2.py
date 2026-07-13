def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: discourages high speed when close to the target
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: sum of bounded factors rewarding simultaneous proximity,
      low speed, upright posture, and ground contact
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
    gate = 1.0 / (1.0 + dist)          # near target -> gate approx 1; far away -> gate approx 0
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: soft landing proxy (sum of bounded factors, non-collapsing joint)
    w_landing = 3.0
    # proximity factor: 1 at dist=0, 0 when dist >= 2.5
    prox_reward = max(0.0, 1.0 - dist / 2.5)
    # speed factor: 1 at speed=0, 0 when speed >= 2.0 m/s
    speed = (vx**2 + vy**2) ** 0.5
    vel_reward = max(0.0, 1.0 - speed / 2.0)
    # angle factor: 1 at angle=0, 0 when |angle| >= 0.5 rad (~28.6 deg)
    angle_reward = max(0.0, 1.0 - abs(angle) / 0.5)
    # contact factor: 0, 0.5, or 1.0 depending on single/double contact
    contact_reward = (left_contact + right_contact) / 2.0
    # sum of four independent bounded factors, averaged
    landing_proxy = w_landing * (prox_reward + vel_reward + angle_reward + contact_reward) / 4.0

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