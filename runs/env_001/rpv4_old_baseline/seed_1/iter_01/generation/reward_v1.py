def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar‑lander‑style environment.
    Task family: navigation_goal_reaching, dynamics_subtype: goal_approach_and_soft_contact.
    """

    # ---- unpack observations ----
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    x_vel_new, y_vel_new = next_obs[2], next_obs[3]
    body_angle_new = next_obs[4]
    ang_vel_new = next_obs[5]
    left_contact_new = next_obs[6]
    right_contact_new = next_obs[7]

    # ---- 1. Approach the goal (improvement_delta) ----
    dist_old = (x_old**2 + y_old**2) ** 0.5
    dist_new = (x_new**2 + y_new**2) ** 0.5
    approach_reward = 100.0 * (dist_old - dist_new)   # positive when getting closer

    # ---- 2. Stability penalty (quadratic_penalty) ----
    w_angle = 5.0
    w_angvel = 0.5
    stability_penalty = -w_angle * body_angle_new**2 - w_angvel * ang_vel_new**2

    # ---- 3. Thrust efficiency (discrete action cost) ----
    w_thrust = 0.03
    thrust_cost = -w_thrust if action != 0 else 0.0    # action 0 = no engine

    # ---- 4. Soft‑landing quality (joint_condition_proxy) ----
    contact_both = float(left_contact_new > 0.5 and right_contact_new > 0.5)
    if contact_both > 0.5:
        # how “soft” the landing is
        vel_sum = abs(x_vel_new) + abs(y_vel_new)
        vel_factor = 1.0 / (1.0 + 10.0 * vel_sum)
        ang_factor = 1.0 / (1.0 + 5.0 * abs(ang_vel_new))
        angle_factor = 1.0 / (1.0 + 5.0 * abs(body_angle_new))
        landing_quality = vel_factor * ang_factor * angle_factor
        landing_bonus = 200.0 * landing_quality
    else:
        landing_bonus = 0.0

    # ---- assemble ----
    total_reward = approach_reward + stability_penalty + thrust_cost + landing_bonus
    components = {
        'approach_reward': approach_reward,
        'stability_penalty': stability_penalty,
        'thrust_cost': thrust_cost,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components