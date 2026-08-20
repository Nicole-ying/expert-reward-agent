def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact
    right_c = next_obs[7]   # right leg contact

    # ------------------------------------------------------------------
    # Component A - goal proximity (positive, bounded approach reward)
    # ------------------------------------------------------------------
    w_prox = 2.0
    dist = (x**2 + y**2) ** 0.5
    # bounded reward: 1.0 when dist=0, decays to 0 when dist large
    comp_prox = w_prox * (1.0 - dist / (dist + 1.0))

    # ------------------------------------------------------------------
    # Component B - velocity penalty (soft brake)
    # ------------------------------------------------------------------
    w_vel = 0.5
    comp_vel = w_vel * (-vx**2 - vy**2)

    # ------------------------------------------------------------------
    # Component C - angle penalty
    # ------------------------------------------------------------------
    w_ang = 1.0
    comp_angle = w_ang * (-angle**2 - 0.1 * angvel**2)

    # ------------------------------------------------------------------
    # Component D - contact quality (leg contact gated by distance)
    # ------------------------------------------------------------------
    w_contact = 2.0
    gate = 1.0 / (1.0 + dist)   # near pad -> ~1, far away -> small
    leg_contact = left_c * right_c   # bilateral contact only
    comp_contact = w_contact * leg_contact * gate

    # ------------------------------------------------------------------
    # Component E - soft success shaping (sum of linear scores)
    # ------------------------------------------------------------------
    w_soft = 30.0

    tol_x     = 0.1
    tol_y     = 0.1
    tol_vx    = 0.2
    tol_vy    = 0.2
    tol_angle = 0.2

    score_x     = max(0.0, 1.0 - abs(x) / tol_x)
    score_y     = max(0.0, 1.0 - abs(y) / tol_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / tol_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / tol_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / tol_angle)
    score_leg   = (left_c + right_c) * 0.5   # linear in [0,1]

    soft_sum = (score_x + score_y + score_vx + score_vy +
                score_angle + score_leg)
    comp_soft = w_soft * soft_sum

    # ------------------------------------------------------------------
    # Component F - thrust penalty (fuel efficiency)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_soft + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "soft_success": comp_soft,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components