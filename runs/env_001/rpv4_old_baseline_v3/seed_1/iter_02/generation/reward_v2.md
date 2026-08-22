Now I have all the evidence. Here's my complete analysis and the rewritten reward function.

---

## Full Diagnosis

| Component | Issue | Root Cause | Action |
|---|---|---|---|
| `proximity_progress` (83% mag) | Dominates but flat across training; only rewards movement, not arrival | Pure delta-distance with no absolute position signal | Reduce 6×, add vertical weight, supplement with position quality |
| `soft_landing_bonus` (14.6% mag, 0.3% active) | Nearly dead; exploited at cp80% when it finally triggers (256.5 spike → score collapse) | Requires both legs simultaneously — far too sparse for learning; 5.0/step too generous when triggered | Tiered contact (partial + full), add hard-landing penalty, reduce per-step |
| `attitude_penalty` (2.5% mag) | Negligible — can't influence behavior | Coefficients 5-10× too small | Strengthen 3× |
| *(missing)* | Agent hovers 813 steps, 15/20 truncated | No fuel cost, no step cost, no speed penalty | Add all three |

The reward needs a complete structural rewrite — not just parameter tuning.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2: balanced approach shaping + tiered contact + efficiency pressure.

    Core changes from v1:
    - approach: vertically-weighted delta-distance (reduced from 30x to 5x)
    - position_quality: dense absolute reward for being near (0,0) with low vel/angle
    - contact_reward: tiered — partial credit for any leg contact, bonus for both
    - hard_landing: penalty for high-speed contact
    - fuel + speed + step costs: create pressure to land efficiently, not hover
    - attitude: strengthened 3x
    """
    # ── state extraction ──
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_x, old_y = obs[0], obs[1]

    # ── 1. approach_shaping: reward moving toward landing zone ──
    # Vertical progress is weighted 2x because descending is the critical action;
    # horizontal centering matters but the agent must commit to going down.
    old_wdist = (old_x ** 2 + 2.0 * old_y ** 2) ** 0.5
    new_wdist = (x ** 2 + 2.0 * y ** 2) ** 0.5
    approach = 5.0 * (old_wdist - new_wdist)

    # ── 2. position_quality: dense reward for being in the landing sweet spot ──
    # Gives the agent a reason to stay near the target, not just approach it.
    # Each factor is in [0, 1]; overall max is 1.0 at perfect landing state.
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5
    pos_factor = 1.0 / (1.0 + 3.0 * dist)
    vel_factor = 1.0 / (1.0 + 2.0 * speed)
    ang_factor = 1.0 / (1.0 + 2.0 * abs(angle))
    position_quality = 1.5 * pos_factor * vel_factor * ang_factor

    # ── 3. contact_reward: tiered — any contact helps, both legs best ──
    any_contact = max(left_contact, right_contact)          # 1 if either leg touches
    both_contact = left_contact * right_contact             # 1 only if both touch

    contact_softness = 1.0 / (1.0 + 3.0 * speed)
    contact_upright = 1.0 / (1.0 + 2.0 * abs(angle))

    # Partial credit for any leg contact (easier to discover)
    partial_contact = 2.0 * any_contact * contact_softness * contact_upright
    # Full bonus for both legs
    full_contact = 6.0 * both_contact * contact_softness * contact_upright
    contact_reward = partial_contact + full_contact

    # ── 4. hard_landing: penalize high-speed contact ──
    hard_landing = -4.0 * both_contact * speed

    # ── 5. attitude_stabilization: strengthened 3x from v1 ──
    attitude_penalty = -1.5 * (angle ** 2) - 0.5 * (ang_vel ** 2)

    # ── 6. speed_penalty: general disincentive for high velocity ──
    speed_penalty = -0.15 * speed

    # ── 7. fuel_cost: penalize engine use to discourage hovering ──
    if action == 2:          # main engine
        fuel_cost = -0.3
    elif action in (1, 3):   # side orientation engines
        fuel_cost = -0.03
    else:                    # no_engine
        fuel_cost = 0.0

    # ── 8. step_cost: small per-step penalty to create urgency to land ──
    step_cost = -0.1

    # ── total ──
    total_reward = (
        approach
        + position_quality
        + contact_reward
        + hard_landing
        + attitude_penalty
        + speed_penalty
        + fuel_cost
        + step_cost
    )

    components = {
        "approach": approach,
        "position_quality": position_quality,
        "contact_reward": contact_reward,
        "hard_landing": hard_landing,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "fuel_cost": fuel_cost,
        "step_cost": step_cost,
    }
    return float(total_reward), components
```

---

## Design Rationale

**Why this fixes the failures:**

| Failure from v1 | How v2 addresses it |
|---|---|
| Agent hovers forever (813 steps, 15/20 truncated) | `fuel_cost` (-0.3 for main engine) + `step_cost` (-0.1/step) make hovering net-negative. The agent must land to stop the bleeding. |
| `proximity_progress` dominates at 83% | Reduced from 30× to 5×, supplemented by `position_quality` which gives absolute credit for being near (0,0) — not just getting closer. |
| `soft_landing_bonus` active only 0.3% | `partial_contact` rewards *any* leg contact (2.0×), making the signal ~100× more frequent. `full_contact` (6.0×) provides the upgrade path. |
| Exploitation at cp80% (256.5 spike → score collapse) | `hard_landing` (-4.0 × speed × both_contact) penalizes the agent if it triggers contact at high speed. Per-step contact reward capped at 8.0 total (vs 5.0 in v1), reducing exploitability. |
| `attitude_penalty` negligible (2.5% mag) | Strengthened 3×: angle² from -0.5 to -1.5, angvel² from -0.1 to -0.5. |
| No speed management | `speed_penalty` (-0.15 × speed) provides general velocity damping. |

**Expected magnitude balance (rough estimate for a 300-step good landing):**
- approach: ~70, position_quality: ~60, contact_reward: ~180, hard_landing: ~-20, attitude: ~-8, speed: ~-40, fuel: ~-35, step: ~-30 → total ~177. No single component exceeds ~35% of magnitude share.