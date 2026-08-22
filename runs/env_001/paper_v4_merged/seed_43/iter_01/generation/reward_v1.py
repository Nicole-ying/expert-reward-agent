def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])
    # obs[4] is current body_angle – not used separately to avoid double penalizing,
    # penalty is applied only on next state

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    nangle = float(next_obs[4])

    # ---------- 1. Main learning signal: potential-based shaping ----------
    # Potential: Phi(s) = -(distance_from_target + alpha * speed)
    # Shaping reward = Phi(s') - Phi(s)
    # This encourages reducing distance AND slowing down when near the target,
    # giving dense gradients every step.
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    speed_obs = (vx * vx + vy * vy) ** 0.5
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5

    alpha = 0.5          # trade-off between position and speed
    potential_obs = -(dist_obs + alpha * speed_obs)
    potential_next = -(dist_next + alpha * speed_next)
    progress_shaping = potential_next - potential_obs

    # ---------- 2. Stability constraint: body angle hinge ----------
    # Penalise only when the tilt exceeds a safe threshold (in radians).
    # Small tilts are unrestricted, promoting exploration without continuous penalty.
    angle_threshold = 0.3       # ~17 degrees
    angle_hinge = -0.5 * max(0.0, abs(nangle) - angle_threshold)

    # ---------- 3. Efficiency bonus: action penalty ----------
    # Discourage unnecessary engine use: any non‑zero action incurs a tiny cost.
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    total_reward = progress_shaping + angle_hinge + action_cost

    components = {
        "progress_shaping": progress_shaping,
        "angle_hinge": angle_hinge,
        "action_cost": action_cost
    }

    return float(total_reward), components