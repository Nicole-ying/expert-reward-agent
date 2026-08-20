```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Best v6 base with single Level 1 fix: w_height reduced 1.0 -> 0.3.
    Height penalty-to-progress ratio drops from 0.925 to ~0.28, giving forward
    velocity the dominant gradient while retaining a soft height constraint.
    All other components and weights are preserved from v6.
    """

    # ---------- Observation indices (v6, confirmed by best run) ----------
    BODY_X_VEL = 13
    BODY_Y_VEL = 14
    ROLL_VEL   = 16
    PITCH_VEL  = 17

    # Quaternion in (w, x, y, z) order — validated by best run survival
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    body_z     = next_obs[0]
    forward_vel = next_obs[BODY_X_VEL]
    lateral_vel = next_obs[BODY_Y_VEL]
    roll_vel    = next_obs[ROLL_VEL]
    pitch_vel   = next_obs[PITCH_VEL]

    # ---------- Weights (v6 preserved except w_height) ----------
    w_fwd    = 1.0
    w_orient = 0.3
    w_lat    = 0.2
    w_ang    = 0.05
    w_height = 0.3   # only change: 1.0 -> 0.3 (Level 1 scale fix)

    # ---------- Height deviation penalty ----------
    # Healthy range [0.2, 1.0], center 0.6, half-range 0.4
    # Quadratic: gentle near center, steep near boundaries
    height_deviation = (body_z - 0.6) / 0.4
    height_deviation_penalty = -w_height * (height_deviation ** 2)

    # ---------- Forward progress (main learning signal) ----------
    forward_reward = w_fwd * forward_vel

    # ---------- Orientation: keep body upright ----------
    orientation_reward = w_orient * body_up_z

    # ---------- Lateral drift suppression ----------
    lateral_penalty = -w_lat * abs(lateral_vel)

    # ---------- Angular velocity damping ----------
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    # ---------- Assemble ----------
    total_reward = (
        forward_reward
        + orientation_reward
        + lateral_penalty
        + angular_penalty
        + height_deviation_penalty
    )

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "height_deviation_penalty": height_deviation_penalty,
    }

    return float(total_reward), components
```