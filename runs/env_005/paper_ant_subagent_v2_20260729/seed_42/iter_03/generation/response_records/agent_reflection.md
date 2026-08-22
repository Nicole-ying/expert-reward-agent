# Response Record

1. `evidence`: score=-75.4 with term 12/20, forward_progress 86.4% share but gated; lateral_drift and roll_pitch penalties are small; gate suppresses forward reward and no direct penalty exists for height near failure boundaries.
2. `behavior_diagnosis`: Agent frequently enters unsafe body‑height regions, causing early termination or low forward reward, leading to a net negative evaluation.
3. `signal_completeness`: The height gate only cancels forward reward but adds no gradient to repel dangerous body_z; a height‑boundary penalty component is missing.
4. `selected_level`: Level 2 — signal gap for height penalty confirmed by termination rate and gate suppression; add one new component.
5. `selected_intervention`: Introduce `height_boundary_hinge`: a hinge penalty on body_z outside [0.3, 0.9] with weight 0.8, providing direct negative feedback for height excursions.
6. `falsifiable_hypothesis`: Direct penalty on near‑boundary body_z will push the agent to stay within [0.3,0.9] more often, increasing height‑gate active steps and forward_progress sum, and reducing terminations.
7. `expected_next_round`: `height_boundary_hinge` active_rate > 0 but < 50%, signed_share small negative (<5%), forward_progress episode_sum_mean may rise slightly, terminated rate should drop, score may improve modestly.
8. `main_risk`: The added penalty could shorten episodes if the margin is too narrow, or cause the agent to over‑prioritize height safety at the cost of forward speed.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract necessary observations
    body_z = obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    body_roll_velocity = obs[16]
    body_pitch_velocity = obs[17]

    # Compute body up vector z-component (uprightness)
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(-1.0, min(1.0, body_up_z))  # numerical safety

    # Height gate
    height_low = 0.2
    height_low_margin = 0.15   # 0.2 + 0.15 = 0.35
    height_high = 0.95
    height_high_margin = 0.05  # 0.95 - 0.05 = 0.90

    if body_z < height_low:
        height_gate = 0.0
    elif body_z < height_low + height_low_margin:
        height_gate = (body_z - height_low) / height_low_margin
    elif body_z > height_high:
        height_gate = 0.0
    elif body_z > height_high - height_high_margin:
        height_gate = (height_high - body_z) / height_high_margin
    else:
        height_gate = 1.0

    # Uprightness gate
    upright_threshold = 0.4   # below this -> gate 0
    upright_full = 0.8        # above this -> gate 1
    if body_up_z < upright_threshold:
        upright_gate = 0.0
    elif body_up_z < upright_full:
        upright_gate = (body_up_z - upright_threshold) / (upright_full - upright_threshold)
    else:
        upright_gate = 1.0

    gate = height_gate * upright_gate

    # Forward progress (main signal)
    w_forward = 1.0
    forward_progress = w_forward * max(0.0, body_x_velocity) * gate

    # Lateral drift penalty (hinge)
    w_lat = 0.5
    lateral_drift_penalty = -w_lat * max(0.0, abs(body_y_velocity) - 0.3)

    # Roll/pitch stability hinge
    w_rp = 0.2
    ang_vel_margin = 1.0  # rad/s
    roll_penalty = max(0.0, abs(body_roll_velocity) - ang_vel_margin)
    pitch_penalty = max(0.0, abs(body_pitch_velocity) - ang_vel_margin)
    roll_pitch_penalty = -w_rp * (roll_penalty + pitch_penalty)

    # NEW: Height boundary hinge penalty
    # Safe region: [0.3, 0.9]; linear penalty outside
    low_thresh = 0.3
    high_thresh = 0.9
    w_height = 0.8
    height_violation = 0.0
    if body_z < low_thresh:
        height_violation += low_thresh - body_z
    if body_z > high_thresh:
        height_violation += body_z - high_thresh
    height_boundary_hinge = -w_height * height_violation

    total_reward = (forward_progress +
                    lateral_drift_penalty +
                    roll_pitch_penalty +
                    height_boundary_hinge)

    components = {
        'forward_progress': forward_progress,
        'lateral_drift_hinge': lateral_drift_penalty,
        'roll_pitch_hinge': roll_pitch_penalty,
        'height_boundary_hinge': height_boundary_hinge
    }

    return float(total_reward), components
```
