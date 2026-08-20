def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: penalizes high speed when near target, gated by distance
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: potential-based shaping reward that rewards IMPROVEMENT
      in approach quality, not the absolute state. Computed as
      w * (approach_quality(next_obs) - approach_quality(obs)).
      Zero reward for maintaining a constant good state, positive only
      for progress toward the landing configuration.
    """
    # Unpack observations
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]
    prev_angle = obs[4]

    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component A: distance penalty (core progress signal) ---
    dist = (x**2 + y**2) ** 0.5
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # --- Component B: velocity penalty (damped by distance to target) ---
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # --- Component C: orientation stabilization penalty ---
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # --- Component D: landing proxy (potential-based shaping) ---
    # Bounded approach quality function: geometric mean of 3 continuous factors.
    # This is the potential function Phi(state).
    def approach_quality(px, py, pvx, pvy, pangle):
        d = (px**2 + py**2) ** 0.5
        prox = max(0.0, 1.0 - d / 2.5)
        sp = (pvx**2 + pvy**2) ** 0.5
        vel = max(0.0, 1.0 - sp / 2.0)
        ang = max(0.0, 1.0 - abs(pangle) / 0.5)
        # Geometric mean enforces joint satisfaction; zero if any factor is zero
        return (prox * vel * ang) ** (1.0 / 3.0)

    # Potential at previous state (obs) and current state (next_obs)
    phi_prev = approach_quality(prev_x, prev_y, prev_vx, prev_vy, prev_angle)
    phi_curr = approach_quality(x, y, vx, vy, angle)

    # Potential-based shaping: reward = gamma * Phi(next) - Phi(now)
    # Using gamma=1 since environment has no explicit discount in reward design.
    # w_potential controls the scale of the improvement reward.
    w_potential = 8.0
    landing_proxy = w_potential * (phi_curr - phi_prev)

    # --- Total reward ---
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components