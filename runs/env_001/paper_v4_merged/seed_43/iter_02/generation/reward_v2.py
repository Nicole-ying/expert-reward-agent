def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])
    # obs[4] body_angle – kept for potential logging, not used separately here

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    nangle = float(next_obs[4])

    # ---------- 1. Main learning signal: potential-based shaping ----------
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    speed_obs = (vx * vx + vy * vy) ** 0.5
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5

    alpha = 0.5
    potential_obs = -(dist_obs + alpha * speed_obs)
    potential_next = -(dist_next + alpha * speed_next)
    progress_shaping = potential_next - potential_obs

    # ---------- 2. Stability constraint: body angle hinge ----------
    angle_threshold = 0.3
    angle_hinge = -0.5 * max(0.0, abs(nangle) - angle_threshold)

    # ---------- 3. Efficiency bonus: action penalty ----------
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    # ---------- 4. NEW: danger penalty for fatal states ----------
    danger = False
    # horizontal out-of-bounds (viewport edge)
    if abs(nx) > 1.2:
        danger = True
    # body below landing pad level (crash into ground)
    elif ny < -0.2:
        danger = True
    # extreme tilt (flipped over)
    elif abs(nangle) > 0.8:
        danger = True
    # excessive speed (crashed at high velocity)
    elif (nvx * nvx + nvy * nvy) ** 0.5 > 5.0:
        danger = True

    danger_penalty = -1.0 if danger else 0.0

    total_reward = progress_shaping + angle_hinge + action_cost + danger_penalty

    components = {
        "progress_shaping": progress_shaping,
        "angle_hinge": angle_hinge,
        "action_cost": action_cost,
        "danger_penalty": danger_penalty
    }

    return float(total_reward), components