# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2 reward: goal proximity (delta of squared distance) + velocity damping (gate by dist)
             + orientation penalty (abs angle + angvel^2)
             + dense landing progress (distance-based + per-leg contact)
    
    obs / next_obs layout: [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    """
    # --- unpack ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement delta on squared distance) ---
    w_x, w_y = 1.0, 2.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity damping (quadratic penalty, gated by distance) ---
    w_v = 0.05
    k_gate = 1.0
    dist_nxt = abs(x_nxt) + abs(y_nxt)
    gate = 1.0 / (1.0 + k_gate * dist_nxt)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation penalty ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. dense landing progress (replaces sparse contact_bonus) ---
    # a) distance-based progress: larger when close to the pad centre
    w_land = 3.0
    eucl_dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    landing_bonus = w_land / (1.0 + 5.0 * eucl_dist_nxt)

    # b) per‑leg contact bonus: encourages any leg touch
    w_leg = 1.0
    leg_bonus = w_leg * (left_cont_nxt + right_cont_nxt)

    contact_bonus = landing_bonus + leg_bonus

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus    # now dense landing signal
    }

    return total_reward, components
```
