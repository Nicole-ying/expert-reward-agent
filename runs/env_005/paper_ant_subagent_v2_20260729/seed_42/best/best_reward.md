1. `evidence`：Score dropped from -75.45 to -186.37 after adding height_boundary_hinge; height_boundary_hinge active_rate 17.7% and sum -4.27 shows negligible but harmful effect; terminated rate 6/20 indicates height violations still occur; body_z_velocity (obs[15]) is unused and could provide early descent warning.
2. `behavior_diagnosis`：Agent survives most episodes (len 752) but accumulates negative total reward, suggesting forward progress is often gated out while penalties persist; occasional fatal falls (terminated) are not prevented by post‑hoc height penalty.
3. `signal_completeness`：Missing early warning for dangerous descent; height penalty triggers only after violation, offering no gradient to avoid low height; body_z_velocity is observable and fits this role.
4. `selected_level`：Level 2 structure transformation — replace height_boundary_hinge with descent_early_penalty to supply missing predictive signal and remove conflicting standalone penalty.
5. `selected_intervention`：Remove height_boundary_hinge entirely; add descent_early_penalty that penalizes downward velocity when body_z is below a safe margin (0.5), using obs[15].
6. `falsifiable_hypothesis`：Early descent penalty will reduce terminated episodes and raise score above previous best (-75.45) by teaching the agent to avoid fatal low‑height trajectories without the conflicting post‑hoc penalty that harmed iter3.
7. `expected_next_round`：Score > -75; terminated < 5/20; descent_early_penalty active_rate > 20%; forward_progress per‑step at least 0.9; lateral/roll pitch penalty unchanged or slightly reduced.
8. `main_risk`：The descent penalty could encourage freezing at safe heights, reducing forward progress; careful calibration is needed to keep stepwise penalty below 0.3× forward signal.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- observations ---
    body_z = obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    body_z_velocity = obs[15]
    body_roll_velocity = obs[16]
    body_pitch_velocity = obs[17]

    # === uprightness gate ===
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(-1.0, min(1.0, body_up_z))  # clip

    upright_threshold = 0.4
    upright_full = 0.8
    if body_up_z < upright_threshold:
        upright_gate = 0.0
    elif body_up_z < upright_full:
        upright_gate = (body_up_z - upright_threshold) / (upright_full - upright_threshold)
    else:
        upright_gate = 1.0

    # === height gate (safety multiplier on forward reward) ===
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

    gate = height_gate * upright_gate

    # === forward progress ===
    w_forward = 1.0
    forward_progress = w_forward * max(0.0, body_x_velocity) * gate

    # === lateral drift penalty (hinge) ===
    w_lat = 0.5
    lateral_drift_penalty = -w_lat * max(0.0, abs(body_y_velocity) - 0.3)

    # === roll/pitch stability hinge ===
    w_rp = 0.2
    ang_vel_margin = 1.0  # rad/s
    roll_penalty = max(0.0, abs(body_roll_velocity) - ang_vel_margin)
    pitch_penalty = max(0.0, abs(body_pitch_velocity) - ang_vel_margin)
    roll_pitch_penalty = -w_rp * (roll_penalty + pitch_penalty)

    # === descent early penalty (replaces height_boundary_hinge) ===
    # Penalty applied when falling (body_z_velocity < 0) and already below safe margin
    safe_z_margin = 0.5
    w_descend = 0.15
    descent_early_penalty = 0.0
    if body_z_velocity < 0.0 and body_z < safe_z_margin:
        # Linear penalty proportional to downward speed and proximity to low height
        fall_factor = (safe_z_margin - body_z) / safe_z_margin  # 0 when body_z = 0.5, 1 when body_z = 0
        descent_early_penalty = -w_descend * abs(body_z_velocity) * fall_factor

    total_reward = (forward_progress +
                    lateral_drift_penalty +
                    roll_pitch_penalty +
                    descent_early_penalty)

    components = {
        'forward_progress': forward_progress,
        'lateral_drift_hinge': lateral_drift_penalty,
        'roll_pitch_hinge': roll_pitch_penalty,
        'descent_early_penalty': descent_early_penalty
    }

    return float(total_reward), components
```