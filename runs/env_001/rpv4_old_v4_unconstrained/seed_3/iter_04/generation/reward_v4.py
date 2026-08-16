def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: eliminate bounce-exploitable contact_bonus; replace with per-step settled_reward.
    
    Core changes:
    - REMOVED transition-based contact_bonus (bounce-exploitable).
    - ADDED settled_reward: per-step, both legs must STAY in contact, tight gates.
    - ADDED contact_speed_penalty: speed while touching ground is punished (anti-bounce).
    - REDUCED landing_approach weight 0.6 → 0.20 (anti-hover).
    - ADDED downward-velocity penalty near ground (anti-slam).
    - REBALANCED goal_proximity for cleaner signal separation.
    """

    # --- unpack next_obs ---
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_cont = next_obs[6]
    right_cont = next_obs[7]

    # --- unpack obs (for delta computations) ---
    x_cur = obs[0]
    y_cur = obs[1]

    # --- derived quantities ---
    dist = (x ** 2 + y ** 2) ** 0.5
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    speed_sq = vx ** 2 + vy ** 2
    both_legs = left_cont * right_cont          # 0.0 or 1.0

    # ================================================================
    # 1. goal_proximity — distance penalty + approach incentive
    #    Moderate weights. landing_approach handles fine positioning.
    # ================================================================
    w_dist = 0.06
    proximity_penalty = -w_dist * dist

    w_approach = 0.30
    delta_dist = dist_cur - dist                 # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ================================================================
    # 2. velocity_damping — general speed penalty + anti-slam near ground
    #    w_v increased, plus dedicated downward-velocity penalty.
    # ================================================================
    w_v = 0.28
    k_gate = 5.0
    gate = 1.0 / (1.0 + k_gate * dist)           # ~1 near pad, ~0 far away

    # anti-slam: heavy penalty for downward velocity when very near ground
    w_slam = 1.0
    slam_gate = 1.0 / (1.0 + 20.0 * dist)        # ultra-tight: only |dist|<0.1 matters
    downward_v = max(0.0, -vy)                   # positive = moving downward

    velocity_penalty = (
        -w_v * gate * speed_sq
        - w_slam * slam_gate * (downward_v ** 2)
    )

    # ================================================================
    # 3. orientation_penalty — unchanged structure, slightly higher weights
    # ================================================================
    w_ang = 0.05
    w_angvel = 0.01
    orientation_penalty = -w_ang * abs(angle) - w_angvel * (angvel_nxt := angvel) ** 2

    # ================================================================
    # 4. landing_approach — continuous shaping for pre-contact guidance
    #    Weight reduced 0.6 → 0.20. Temperature tightened slightly.
    #    Provides a broad basin of attraction toward landing configuration.
    # ================================================================
    near_y   = 2.718281828 ** (-abs(y) / 0.22)
    near_x   = 2.718281828 ** (-abs(x) / 0.30)
    slow     = 2.718281828 ** (-speed_sq / 0.45)
    upright  = 2.718281828 ** (-abs(angle) / 0.28)
    landing_approach = 0.20 * near_y * near_x * slow * upright

    # ================================================================
    # 5. settled_reward — per-step reward for STAYING landed properly
    #    Replaces old contact_bonus entirely. Only pays when:
    #    - both legs are ON the ground (sustained, not transition)
    #    - VERY close to (0,0) in position
    #    - VERY low speed
    #    - near-upright orientation
    #    Tight gates mean bouncing → near-zero reward (no re-trigger exploit).
    # ================================================================
    w_settled = 3.5
    s_spatial  = 2.718281828 ** (-(x ** 2 + y ** 2) / 0.04)
    s_slow     = 2.718281828 ** (-speed_sq / 0.03)
    s_upright  = 2.718281828 ** (-abs(angle) / 0.10)
    settled_reward = w_settled * both_legs * s_spatial * s_slow * s_upright

    # ================================================================
    # 6. contact_speed_penalty — anti-bounce: speed while in contact is BAD
    #    Directly punishes the bouncing exploit. Strong enough to matter.
    # ================================================================
    w_contact_spd = 2.0
    contact_speed_penalty = -w_contact_spd * both_legs * speed_sq

    # ================================================================
    # 7. fuel_efficiency — moderate regularisation
    # ================================================================
    w_fuel = 0.006
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ================================================================
    # assemble
    # ================================================================
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + settled_reward
        + contact_speed_penalty
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'settled_reward': settled_reward,
        'contact_speed_penalty': contact_speed_penalty,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components