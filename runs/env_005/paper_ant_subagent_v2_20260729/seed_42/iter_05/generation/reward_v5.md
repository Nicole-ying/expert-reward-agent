1. evidence: Best score 5.5 after 4 iterations (target 3000), 18/20 episodes truncated (alive), forward_progress dominates at 88.5% signed share but net reward is tiny; descent_early_penalty dead; no joint/yaw signals used.
2. behavior_diagnosis: Agent survives until timeout but moves inefficiently—forward speed is present but offset by constant small lateral and angular disturbances, yielding near-zero net reward; it lacks incentive to push speed or refine gait.
3. signal_completeness: Forward speed, body height, and orientation signals are available; a convex speed attractor is missing, and the old independent penalties failed to stabilise gait because they never forced simultaneous satisfaction of multiple constraints.
4. selected_level: Level 3—same skeleton family (forward + independent penalties) has persisted 4 iterations and best score remains <0.2% of target, fulfilling the REBUILD condition.
5. selected_intervention: Replace all independent constraints with a geometric‑mean health gate (height, upright, lateral, angular including yaw) multiplied by a convex forward‑speed reward (velocity²), plus a tiny action‑cost penalty.
6. falsifiable_hypothesis: The multiplicative gate forces the agent to meet all health conditions before earning substantial forward reward, thereby aligning stability and speed optimisation; this should raise forward_reward’s episode_sum_mean >500, keep len >900, and lift the evaluation score from ~5 to at least 10‑50.
7. expected_next_round: forward_reward active_rate >70%, episode_sum_mean >500, lateral/angular signals indirectly improved through gate, score 10-50, len ≈950.
8. main_risk: The gate product may collapse to zero too early during exploration, causing zero reward throughout an episode and stalling learning; the action penalty is minimal and may not suppress motor jitter if needed.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- observations ---
    body_z = obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]
    roll_vel = obs[16]
    pitch_vel = obs[17]
    yaw_vel = obs[18]

    # --- upright factor ---
    up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    up_z = max(-1.0, min(1.0, up_z))
    if up_z < 0.4:
        up_factor = 0.0
    elif up_z < 0.8:
        up_factor = (up_z - 0.4) / 0.4
    else:
        up_factor = 1.0

    # --- height factor (healthy range: 0.2–1.0) ---
    h_low_safe = 0.35
    h_high_safe = 0.9
    if body_z <= 0.2:
        h_factor = 0.0
    elif body_z < h_low_safe:
        h_factor = (body_z - 0.2) / (h_low_safe - 0.2)
    elif body_z <= h_high_safe:
        h_factor = 1.0
    elif body_z < 1.0:
        h_factor = (1.0 - body_z) / (1.0 - h_high_safe)
    else:
        h_factor = 0.0

    # --- lateral speed factor (0 = perfect, decays with speed) ---
    lat_speed = abs(body_y_vel)
    k_lat = 2.0
    lat_factor = 1.0 / (1.0 + k_lat * lat_speed)

    # --- angular disturbance factor (roll + pitch + yaw) ---
    ang_disturb = abs(roll_vel) + abs(pitch_vel) + abs(yaw_vel)
    k_ang = 0.5
    ang_factor = 1.0 / (1.0 + k_ang * ang_disturb)

    # --- multiplicative health gate (geometric mean) ---
    gate = (h_factor * up_factor * lat_factor * ang_factor) ** 0.25

    # --- convex forward progress ---
    w_forward = 1.0
    forward_reward = w_forward * (max(0.0, body_x_vel) ** 2) * gate

    # --- tiny action cost to discourage useless jitter ---
    action_cost = 0.001 * sum(a * a for a in action)

    total_reward = forward_reward - action_cost

    components = {
        'forward_reward': forward_reward,
        'action_cost': action_cost,
        'height_factor': h_factor,
        'upright_factor': up_factor,
        'lateral_factor': lat_factor,
        'angular_factor': ang_factor,
        'gate': gate
    }

    return float(total_reward), components
```