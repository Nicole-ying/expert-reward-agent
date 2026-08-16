```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander: approach target pad, slow down, and settle
    with both support legs on the platform.

    Components:
      - progress (improvement delta on distance to target)
      - landing_gate_penalty (soft gate penalising speed/angle near platform)
      - contact_quality (bonus for dual leg contact with low speed & tilt)
    """
    # --- 1. Progress: distance reduction (improvement_delta) ---
    x_cur, y_cur = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_cur - dist_next)   # positive when approaching

    # --- 2. Soft landing gate: penalise speed & attitude only near ground ---
    y_threshold = 0.5               # how close to the platform the gate activates
    gate = max(0.0, 1.0 - abs(y_next) / y_threshold)  # 0 when far, ->1 when y≈0

    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    vel_penalty = vx**2 + vy**2
    angle_penalty = angle**2
    ang_vel_penalty = ang_vel**2

    landing_gate_penalty = -gate * (2.0 * vel_penalty + 1.0 * angle_penalty + 0.5 * ang_vel_penalty)

    # --- 3. Contact quality bonus: sparse reward for stable dual-leg contact ---
    left = next_obs[6]
    right = next_obs[7]
    contact_both = left * right               # 0 or 1

    # quality decays towards 0 when speed or tilt are large
    v_sq = vx**2 + vy**2
    angle_sq = angle**2
    v_scale = 0.1          # width of the exponential (tuneable)
    a_scale = 0.1
    quality = (2.718281828 ** (-v_sq / v_scale)) * (2.718281828 ** (-angle_sq / a_scale))

    contact_reward = contact_both * 20.0 * quality

    # --- assemble ---
    total = progress + landing_gate_penalty + contact_reward
    components = {
        'progress': progress,
        'landing_gate_penalty': landing_gate_penalty,
        'contact_quality': contact_reward
    }
    return (float(total), components)
```