1. **evidence**：final score=-222.06, 20/20 terminated, soft_landing/angvel dead, progress minor, failure_penalty rare but huge negative contribution, all prior skeletons showed no meaningful improvement.
2. **behavior_diagnosis**：the policy flies long episodes (mean 148.5) without ever achieving soft landing, then crashes or goes out-of-bounds; the lack of a strong progress gradient and missing success signal prevent any learning toward the task goal.
3. **signal_completeness**：the per-step progress was too weak and gated too aggressively, the success condition was never triggered, and the failure penalty only activated on terminal explosions without guiding avoidance.
4. **selected_level**：Level 3 rebuild — the same skeleton family failed for 3 consecutive rounds and the best score never exceeded -113, with pre‑judgement all ❌.
5. **selected_intervention**：design a new skeleton based on improvement_delta (distance reduction + encouraged descent) as the main progress signal, add separate success_bonus (soft‑landing condition) and failure_penalty (out‑of‑bounds/crash), keep a hinge angle penalty and a light action‑efficiency cost.
6. **falsifiable_hypothesis**：providing dense progress reward combined with a clearly separated terminal success bonus will steer the agent toward safe descent and landing, thereby raising score from ≪-200 to a positive range and making success_bonus actually activate.
7. **expected_next_round**：score should rise substantially, possibly above zero; progress must become the dominant positive component; success_bonus active_rate > 0 and some episodes should succeed; episode length likely shortens.
8. **main_risk**：the success thresholds could be too tight, making the bonus unreachable; the failure detection may fire in ambiguous states and create hesitation.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── thresholds and weights ──
    safe_vy_max   = 0.5
    vy_margin     = 1.0
    angle_thresh  = 0.3
    angle_w       = 0.3

    succ_x        = 0.2
    succ_y        = 0.2
    succ_vel      = 0.3
    succ_angle    = 0.2
    succ_reward   = 15.0

    fail_x        = 2.0
    fail_y        = -0.2
    fail_reward   = -10.0

    w_progress    = 1.0
    w_vert        = 0.5
    eff_w         = 0.02

    # ── unpack ──
    old_x, old_y  = obs[0],   obs[1]
    new_x, new_y  = next_obs[0], next_obs[1]
    new_vx, new_vy = next_obs[2], next_obs[3]
    new_angle = next_obs[4]
    left, right = next_obs[6], next_obs[7]

    # ── success check ──
    any_contact = 1.0 if (left + right) > 0.5 else 0.0
    speed = (new_vx ** 2 + new_vy ** 2) ** 0.5
    if (abs(new_x) < succ_x and abs(new_y) < succ_y and speed < succ_vel
            and abs(new_angle) < succ_angle and any_contact > 0.5):
        success = succ_reward
    else:
        success = 0.0

    # ── failure check (only when not success) ──
    if success == 0.0:
        if abs(new_x) > fail_x or new_y < fail_y:
            failure = fail_reward
        else:
            failure = 0.0
    else:
        failure = 0.0

    # ── progress: improvement delta + descent gate ──
    old_dist = (old_x ** 2 + old_y ** 2) ** 0.5
    new_dist = (new_x ** 2 + new_y ** 2) ** 0.5
    delta_dist = old_dist - new_dist          # positive = closer to origin
    delta_y = old_y - new_y                   # positive = going down

    downward_speed = -new_vy if new_vy < 0.0 else 0.0
    if downward_speed > safe_vy_max:
        overshoot = downward_speed - safe_vy_max
        gate = max(0.0, 1.0 - overshoot / vy_margin)
    else:
        gate = 1.0

    progress = w_progress * (delta_dist + w_vert * max(0.0, delta_y)) * gate

    # ── angle penalty (hinge) ──
    angle_pen = -angle_w * max(0.0, abs(new_angle) - angle_thresh)

    # ── action efficiency ──
    eff = -eff_w * (0.0 if action == 0 else 1.0)

    total = progress + angle_pen + eff + success + failure

    components = {
        'progress': progress,
        'angle_penalty': angle_pen,
        'efficiency': eff,
        'success_bonus': success,
        'failure_penalty': failure
    }
    return float(total), components
```