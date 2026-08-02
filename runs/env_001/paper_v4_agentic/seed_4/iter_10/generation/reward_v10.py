def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- unpack observations -------------------
    x,  y  = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle      = obs[4]
    angvel     = obs[5]
    left_leg   = obs[6]
    right_leg  = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle  = next_obs[4]
    n_angvel = next_obs[5]
    n_left   = next_obs[6]
    n_right  = next_obs[7]

    # ------------------- helper quantities -------------------
    dist      = (x**2  + y**2)  ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 8.0
    w_approach   = 1.0
    w_touchdown  = 10.0

    th_angle  = 0.5
    th_vel    = 1.0
    th_angvel = 2.0

    # brake reward weights/thresholds
    w_brake = 0.5
    descend_threshold = -0.3

    # ------------------- 1. distance-improvement progress signal -------------------
    delta_dist = dist - next_dist   # positive when approaching target

    gate_min = 0.1
    gate_angle  = max(gate_min, 1.0 - abs(n_angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - next_vel_abs   / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(n_angvel)  / th_angvel)
    gate = (gate_angle * gate_vel * gate_angvel) ** (1.0/3.0)

    progress_delta = w_progress * max(0.0, delta_dist) * gate

    # ------------------- 2. landing (approach + touchdown) -------------------
    contact_next = (n_left + n_right) / 2.0

    pos_factor    = max(0.0, 1.0 - abs(nx) / 0.5)
    height_factor = max(0.0, 1.0 - abs(ny) / 0.5)
    vel_factor    = max(0.0, 1.0 - next_vel_abs / 0.5)
    angle_factor  = max(0.0, 1.0 - abs(n_angle) / 0.3)
    angvel_factor = max(0.0, 1.0 - abs(n_angvel) / 0.5)

    touchdown_reward = 0.0
    approach_reward  = 0.0

    if contact_next > 0.1:
        quality = pos_factor * height_factor * vel_factor * angle_factor * angvel_factor * contact_next
        touchdown_reward = w_touchdown * quality
    else:
        dist_factor   = max(0.2, 1.0 - next_dist / 1.5)
        h_factor      = max(0.2, 1.0 - abs(ny) / 1.5)
        vel_factor_a  = max(0.2, 1.0 - next_vel_abs / 0.8)
        angle_factor_a = max(0.2, 1.0 - abs(n_angle) / 0.5)
        angvel_factor_a = max(0.2, 1.0 - abs(n_angvel) / 1.0)
        approach_reward = w_approach * dist_factor * h_factor * vel_factor_a * angle_factor_a * angvel_factor_a

    landing = approach_reward + touchdown_reward

    # ------------------- 3. brake reward (replaces fuel penalty) -------------------
    brake_reward = 0.0
    if action == 2 and nvy < descend_threshold:
        brake_reward = w_brake * (descend_threshold - nvy)

    # ------------------- total reward -------------------
    total_reward = progress_delta + landing + brake_reward

    components = {
        'progress_delta':   progress_delta,
        'landing':          landing,
        'brake_reward':     brake_reward
    }

    return float(total_reward), components