# Response Record

`evidence`: Current iter 5 scored −123.86 with all 20 episodes terminating early (avg len=68.3) at score < −50, while iter 3 (best, −26.30) survived 951 steps using proximity-based reward. `progress_delta` was 100% active (mean 1.11) and `descent_safety` was entirely absent. The positive progress_delta signal combined with rapid crashing is direct evidence of proxy misalignment — the agent learned to reduce distance by diving into the pad at fatal speed.

`behavior_diagnosis`: The agent dives straight toward the target origin at high speed, reducing `progress_delta` maximally per step, but inevitably crashes because no component penalizes unsafe descent velocity or angle near the ground.

`signal_completeness`: A safe-descent signal is missing. The current reward has proximity guidance (`progress_delta`) and a sparse landing event (`contact_event`, 1.0% active) but nothing that communicates "you must slow down before reaching the ground." Without this, the credit assignment problem is unsolvable — the agent only discovers crashing is bad through sparse external termination, too late to learn a braking policy.

`selected_level`: Level 2 — `proxy_to_completion_alignment`. The `progress_delta` proxy is structurally misaligned with safe landing; simply rescaling it (Level 1) won't fix the incentive to dive. The fix transforms the guidance signal from raw distance reduction to proximity-with-safety-gating and adds a descent-danger penalty that activates continuously before impact.

`selected_intervention`: Remove `progress_delta` and `approach_improvement` entirely. Add three new components: (1) `safe_proximity` — bounded state-based reward `1/(1+dist)` that does not reward the act of moving closer; (2) `descent_safety` — penalty `-2 * total_speed / (1 + y)` active only when falling (`yv < 0`), ramping up as altitude shrinks; (3) `stable_landed` — ongoing reward when both legs touch, altitude is low, speed and angle are good, to incentivize maintaining a landed state instead of hovering.

`falsifiable_hypothesis`: By removing the per-step reward for distance reduction and imposing a penalty proportional to `speed/altitude` during descent, the agent can no longer profit from diving. The only way to accumulate positive reward becomes: approach the pad slowly enough to avoid the descent penalty, make leg contact, and remain stable — which is the actual task.

`expected_next_round`: `descent_safety` active_rate should be high (>80% if the agent descends at all) and episode length should increase beyond 150 steps as the agent learns to brake. `safe_proximity` should replace `progress_delta` as the main guidance component. External score should rise above the current best of −26.30 if the agent learns controlled descent.

`main_risk`: The `descent_safety` penalty, if too strong relative to `safe_proximity`, could make descending at all net-negative, causing the agent to hover high up indefinitely or even ascend to avoid the penalty — a new form of proxy exploitation. The 2.0 coefficient and `1/(1+y)` urgency function are initial guesses that may need tuning.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous state
    prev_lc = obs[6]
    prev_rc = obs[7]

    # Next state
    x = next_obs[0]
    y = next_obs[1]
    xv = next_obs[2]
    yv = next_obs[3]
    ang = next_obs[4]
    lc = next_obs[6]
    rc = next_obs[7]

    # 1. Safe proximity: bounded state reward for being close to pad center
    # Max ~1.0 at origin, decays smoothly — does NOT reward the act of moving
    dist = (x ** 2 + y ** 2) ** 0.5
    safe_proximity = 1.0 / (1.0 + dist)

    # 2. Descent safety: penalize dangerous high-speed descent near ground
    # Only active when falling (yv < 0); urgency grows as altitude shrinks
    descent_safety = 0.0
    if yv < 0.0 and y > 0.02:
        total_speed = (xv ** 2 + yv ** 2) ** 0.5
        height_urgency = 1.0 / (1.0 + y)
        descent_safety = -2.0 * total_speed * height_urgency

    # 3. Touchdown bonus: first moment both legs achieve ground contact
    prev_contacts = prev_lc + prev_rc
    next_contacts = lc + rc
    touchdown_bonus = 0.0
    if next_contacts >= 2.0 and prev_contacts < 2.0:
        speed_quality = max(0.0, 1.0 - (abs(xv) + abs(yv)))
        angle_quality = max(0.0, 1.0 - abs(ang) / 0.5)
        touchdown_bonus = 3.0 * (1.0 + speed_quality * angle_quality)

    # 4. Stable landed: ongoing reward for maintaining a successful landing state
    stable_landed = 0.0
    if lc > 0.5 and rc > 0.5 and y < 0.3:
        speed_score = max(0.0, 1.0 - (abs(xv) + abs(yv)))
        angle_score = max(0.0, 1.0 - abs(ang))
        stable_landed = speed_score * angle_score

    # 5. Fuel penalty: small cost for any engine use
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = safe_proximity + descent_safety + touchdown_bonus + stable_landed + fuel_penalty

    components = {
        "safe_proximity": safe_proximity,
        "descent_safety": descent_safety,
        "touchdown_bonus": touchdown_bonus,
        "stable_landed": stable_landed,
        "fuel_penalty": fuel_penalty
    }

    return (float(total_reward), components)
```
