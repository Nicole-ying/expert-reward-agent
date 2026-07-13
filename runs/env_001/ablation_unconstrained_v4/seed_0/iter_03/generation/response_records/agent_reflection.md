# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation to target platform.

    Key change from previous: settle_delta replaces absolute settle_reward.
    Settle quality improvement is rewarded per-step; holding steady gives zero.
    This prevents the "hover and farm" strategy seen with per-step absolute settle.
    """
    # Previous state
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    # Next state (result of action)
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speeds ---
    prev_speed = (pvx ** 2 + pvy ** 2) ** 0.5
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (tighter than before: k=7.0) ---
    k_prox = 7.0
    prev_prox = 1.0 / (1.0 + k_prox * prev_distance)
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # --- Settle quality factors (previous state) ---
    prev_velocity_factor = 1.0 / (1.0 + 3.0 * prev_speed)
    prev_angle_factor = 1.0 / (1.0 + 3.0 * abs(pangle))
    prev_contact_factor = 0.5 * (pleft_contact + pright_contact)
    prev_quality = prev_prox * (prev_velocity_factor + prev_angle_factor + prev_contact_factor)

    # --- Settle quality factors (current state) ---
    curr_velocity_factor = 1.0 / (1.0 + 3.0 * curr_speed)
    curr_angle_factor = 1.0 / (1.0 + 3.0 * abs(nangle))
    curr_contact_factor = 0.5 * (nleft_contact + nright_contact)
    curr_quality = curr_prox * (curr_velocity_factor + curr_angle_factor + curr_contact_factor)

    # ============================================================
    # Component A: progress_delta (primary approach guidance)
    # ============================================================
    w_progress = 3.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (steady baseline pull)
    # ============================================================
    w_dist = 0.5
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: velocity_damping (proximity-gated speed penalty)
    # ============================================================
    w_vel = 1.5
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component D: settle_delta (improvement in landing quality)
    # ============================================================
    w_settle = 8.0
    settle_delta = w_settle * (curr_quality - prev_quality)

    # ============================================================
    # Component E: orientation_penalty (proximity-gated tilt penalty)
    # ============================================================
    w_orient = 0.5
    w_angvel = 0.15
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component F: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.05
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + velocity_damping
        + settle_delta
        + orientation_penalty
        + engine_penalty
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "velocity_damping": float(velocity_damping),
        "settle_delta": float(settle_delta),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), components
```
