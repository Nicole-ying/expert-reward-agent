1. evidence: All 20 episodes terminated early (crash/out-of-bounds); progress_shaping positive (1.06 sum) but unable to prevent crashes; angle_hinge dead (0.1% active_rate); landing_contact sparse (3.1%); previous iteration added landing_contact and dropped score to -122.
2. behavior_diagnosis: Agent rushes toward the center due to progress shaping but fails to decelerate when approaching the pad, resulting in high-speed crash or out-of-bounds termination every episode.
3. signal_completeness: Lacks a proximity-based speed constraint; dead angle constraint provides no useful gradient; a soft landing gate on progress is missing.
4. selected_level: Level 2 — structure change: remove zombie angle_hinge and replace with a landing_speed_gate that scales progress_shaping based on speed when close to target.
5. selected_intervention: Delete angle_hinge; add `landing_speed_gate = 1.0 / (1.0 + 5.0 * speed_next * max(0.0, 1.0 - dist_next / 0.5))` and multiply progress_shaping by it. This one-component swap leaves action_cost and landing_contact_reward untouched.
6. falsifiable_hypothesis: By reducing progress reward when speed is high near the pad, the agent will learn to slow down on approach, reducing crash rate and increasing score (less negative).
7. expected_next_round: Score should improve (less negative), len may rise slightly; progress_shaping signed_share may drop but total reward gets closer to zero; terminated rate may stay 100% initially but average episode lifetime could extend.
8. main_risk: If gate is too strong, agent might hover outside the threshold radius and never get close enough to trigger landing, stalling progress.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    left_contact = float(next_obs[6])
    right_contact = float(next_obs[7])

    # ---------- 1. Main learning signal: potential-based shaping ----------
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    speed_obs = (vx * vx + vy * vy) ** 0.5
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5

    alpha = 0.5
    potential_obs = -(dist_obs + alpha * speed_obs)
    potential_next = -(dist_next + alpha * speed_next)
    progress_shaping = potential_next - potential_obs

    # ---------- 2. Landing speed gate (replaces dead angle_hinge) ----------
    # Gate: penalises high speed when close to target (dist < 0.5)
    proximity_factor = max(0.0, 1.0 - dist_next / 0.5)   # 0 at >=0.5, 1 at 0
    # speed_next multiplied by proximity: large number only when fast and close
    speed_cost_input = speed_next * proximity_factor
    landing_speed_gate = 1.0 / (1.0 + 5.0 * speed_cost_input)

    # Apply gate to progress shaping
    shaped_progress = progress_shaping * landing_speed_gate

    # ---------- 3. Efficiency: action penalty ----------
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    # ---------- 4. Landing contact bonus ----------
    contact_sum = left_contact + right_contact
    contact_factor = contact_sum / 2.0
    proximity = max(0.0, 1.0 - dist_next / 0.8)
    landing_contact_reward = 0.2 * contact_factor * proximity

    total_reward = shaped_progress + action_cost + landing_contact_reward

    components = {
        "progress_shaping": progress_shaping,
        "landing_speed_gate": landing_speed_gate,
        "shaped_progress": shaped_progress,
        "action_cost": action_cost,
        "landing_contact_reward": landing_contact_reward
    }

    return float(total_reward), components
```