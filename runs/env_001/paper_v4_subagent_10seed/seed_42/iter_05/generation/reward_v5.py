def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Approach: replace safe_progress with unbounded distance progress + velocity alignment.
    Soft-landing and weak orientation penalty unchanged.
    """

    # ── Unpack observations ──────────────────────────────────────────
    px0, py0 = obs[0], obs[1]          # last position
    px1, py1 = next_obs[0], next_obs[1]  # current position
    vx1, vy1 = next_obs[2], next_obs[3]  # current velocity
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ─────────────────────────────────────────────
    dist_prev = (px0**2 + py0**2) ** 0.5
    dist_next = (px1**2 + py1**2) ** 0.5
    speed_sq  = vx1**2 + vy1**2
    speed     = speed_sq ** 0.5

    # ── 1. Approach (distance progress + velocity alignment) ─────────
    # progress: positive when moving toward origin
    raw_progress = dist_prev - dist_next
    progress     = max(0.0, raw_progress)

    # alignment: cosine similarity between velocity and direction-to-target
    if dist_next > 1e-6 and speed > 1e-6:
        # direction to target is (-px1, -py1)
        dot = vx1 * (-px1) + vy1 * (-py1)
        alignment = dot / (dist_next * speed)
    else:
        alignment = 0.0      # at origin or stationary
    alignment = max(0.0, alignment)   # only reward approaching motion

    w_progress  = 2.0
    w_alignment = 1.0
    approach    = w_progress * progress + w_alignment * alignment

    # ── 2. Orientation / stability penalties (unchanged, weak) ──────
    angle_penalty  = -0.01 * (angle1 ** 2)
    angvel_penalty = -0.005 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ── 3. Soft landing guidance (unchanged) ─────────────────────────
    proximity_threshold = 0.3
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor   = 1.0 / (1.0 + 10.0 * speed)
        angle_factor   = 1.0 / (1.0 + 5.0 * (angle1**2))
        soft_landing   = contact_factor * speed_factor * angle_factor
    else:
        soft_landing = 0.0

    # ── Combine ──────────────────────────────────────────────────────
    total_reward = (
        1.0 * approach
        + 1.0 * orientation_penalty
        + 2.0 * soft_landing
    )

    components = {
        "approach":            approach,
        "orientation_penalty": orientation_penalty,
        "soft_landing":        soft_landing,
    }
    return float(total_reward), components