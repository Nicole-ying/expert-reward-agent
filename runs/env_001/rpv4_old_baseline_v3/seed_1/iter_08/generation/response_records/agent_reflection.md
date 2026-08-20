# Response Record

Good. Now I have full context across all iterations. Here's my final diagnosis and v8 design:

---

## Final Diagnosis Summary

| Component | Problem | Action |
|---|---|---|
| `soft_landing` | **191x exploit**: 25.0/step × 30 settled steps = 750 pts, 94.2% share. Agent abandons progress to farm contact. | **Replace** with one-shot transition bonus |
| `progress_reward` | Drowned from 93%→4.7% share. Coefficient 30x is correct. | **Keep**, will recover when exploit removed |
| `descent_bonus` | Stable ~7.0, all checkpoints. | **Keep** |
| `attitude_penalty` | Stable ~-2.0, working. | **Keep** |
| `crash_penalty` | Dead, never triggered. | **Remove** |

**The fix**: Convert `soft_landing` from a per-step state reward (exploitable) into a **one-shot transition reward** that fires only when the agent first achieves two-leg contact. Use `obs[6]*obs[7]` vs `next_obs[6]*next_obs[7]` to detect the transition. Add a tiny per-step stability nudge (0.3 max) for posture while in contact — too small to exploit.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    obs / next_obs: 8-dim array
    [x_position, y_position, x_velocity, y_velocity, body_angle,
     angular_velocity, left_contact, right_contact]

    v8: Fix soft_landing exploit by converting it into a one-shot transition bonus.
    - Keep v5's 30x Euclidean progress backbone
    - Keep descent_bonus for dense vertical signal
    - Replace per-step soft_landing (25.0/step exploit) with:
      * landing_bonus: one-shot transition reward (~50 max) when both legs first touch
      * stability_reward: tiny per-step posture nudge (0.3 max, not exploitable)
    - Drop dead crash_penalty
    - Drop fuel_cost (v6's magnitude killer)
    """
    ox, oy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]
    o_left, o_right = obs[6], obs[7]
    n_left, n_right = next_obs[6], next_obs[7]

    # ── 1. Progress: Euclidean distance delta (v5's proven backbone) ──
    old_dist = (ox**2 + oy**2) ** 0.5
    new_dist = (nx**2 + ny**2) ** 0.5
    progress_reward = 30.0 * (old_dist - new_dist)

    # ── 2. Descent bonus: reward downward movement, zero for hover/ascent ──
    dy = oy - ny
    descent_bonus = 5.0 * max(0.0, dy)

    # ── 3. Attitude penalty: keep level, suppress spin ──
    attitude_penalty = 0.5 * (n_angle**2) + 0.1 * (n_av**2)

    # ── 4. Landing transition bonus: ONE-SHOT reward for first achieving contact ──
    # Uses obs→next_obs transition: only fires when contact is NEW, not maintained.
    # This eliminates the "sit on pad for 750 points" exploit.
    old_both = o_left * o_right
    new_both = n_left * n_right

    landing_bonus = 0.0
    if new_both > old_both:
        speed = (nvx**2 + nvy**2) ** 0.5
        speed_factor = 1.0 / (1.0 + 5.0 * speed)
        angle_factor = 1.0 / (1.0 + 5.0 * abs(n_angle))
        landing_bonus = 50.0 * speed_factor * angle_factor

    # ── 5. Stability: tiny per-step reward for good contact posture ──
    # At 0.3 max/step, even 30 settled steps = 9 points (not 750).
    stability_reward = 0.0
    if new_both > 0.0:
        speed = (nvx**2 + nvy**2) ** 0.5
        speed_factor = 1.0 / (1.0 + 5.0 * speed)
        angle_factor = 1.0 / (1.0 + 5.0 * abs(n_angle))
        stability_reward = 0.3 * speed_factor * angle_factor

    total_reward = (
        progress_reward
        + descent_bonus
        - attitude_penalty
        + landing_bonus
        + stability_reward
    )

    components = {
        "progress_reward": progress_reward,
        "descent_bonus": descent_bonus,
        "attitude_penalty": -attitude_penalty,
        "landing_bonus": landing_bonus,
        "stability_reward": stability_reward,
    }

    return float(total_reward), components
```
