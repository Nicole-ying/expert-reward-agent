Now let me systematically analyze each component before writing code.

## Component-by-Component Diagnosis

### 1. `fuel_cost` — **THE KILLER** (magnitude_share=63.7%, ep_sum=-42.72)

This is the dominant negative force. At 0.05 per engine step × ~854 active steps across 1000, it overwhelms everything. Checkpoint trend confirms: fuel_cost grows monotonically from -1.29 (cp10%) to -43.67 (cp100%). The agent fires engines more as training progresses (active rate 34% → 87%), but the penalty grows faster than progress_reward can compensate. **This single component explains the score collapse from v5's 146 to v6's -53.**

**Verdict: REMOVE entirely.** v5 proved you don't need it to score 146.

### 2. `progress_reward` — Working but too weak vs fuel_cost (ep_sum=13.28, magnitude_share=34.5%)

The potential-based formulation (`-ndist - 0.2*nspeed_sq`) at 15× scale is functional but half v5's 30×. In v5, pure Euclidean delta at 30× delivered ep_sum=22.0 and score=146. The velocity-aware potential in v6 is clever but underpowered relative to the fuel_cost drain. Checkpoint trend shows it stable at 12-19 throughout — it works, just can't overcome the negative.

**Verdict: Revert to v5's simpler 30× Euclidean progress.** It's proven.

### 3. `contact_reward` — Dead at eval, was alive mid-training (cp20%=39.9, cp90%=69.3, cp100%=0.0)

Fascinating pattern: the agent *did* learn to contact the pad mid-training, but the behavior was extinguished by cp100%. Why? Because fuel_cost kept rising and the agent discovered a better local optimum: hover without contacting (collect progress_reward, pay fuel_cost, but avoid the risk of crashing). Once contact_reward went to 0 at final eval, the agent had fully abandoned landing.

**Verdict: Replace with a contact-conditioned soft_landing bonus that's strong enough to overcome any negative drift.**

### 4. `landing_bonus` — Dead at eval, spiked at cp90% (166.05!)

The massive spike at cp90% shows the agent briefly discovered stable landing, earning 166 in bonus — but lost it by cp100% (0.0). The 3.0 bonus is too sparse and too small to stabilize the behavior. When the agent wanders away from the landing configuration, there's no gradient to pull it back.

**Verdict: Replace with a continuous soft_landing reward that provides gradient even for partial contact.**

### 5. `attitude_penalty` — Stable, mild, not harmful (ep_sum=-1.21)

Always active, small magnitude. The coefficients are reasonable. No changes needed.

### 6. `crash_penalty` — Dead after cp10% (ep_sum=0.0, active=0.0%)

The `speed > 1.0` threshold is too low to trigger — the agent never hits the ground fast enough. Not harmful but also not helping.

**Verdict: Keep but raise threshold to 2.0 to only catch genuinely dangerous impacts.**

---

## Strategy for v7

1. **Drop `fuel_cost` entirely** — the single biggest regression from v5→v6
2. **Revert `progress_reward` to v5's 30× Euclidean delta** — simpler, proven
3. **Keep `descent_bonus`** (from v4) — dense vertical progress signal, no local optimum
4. **Replace `contact_reward` + `landing_bonus` with a continuous `soft_landing`** — gradient-rich, rewards partial contact
5. **Keep `attitude_penalty`** as-is
6. **Keep `crash_penalty` with raised threshold** as a safety guard

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    obs / next_obs: 8-dim array
    [x_position, y_position, x_velocity, y_velocity, body_angle,
     angular_velocity, left_contact, right_contact]

    v7 design:
    - Restore v5's proven 30x Euclidean progress as the approach engine
    - Add descent_bonus for dense vertical progress (no hovering local optimum)
    - Replace sparse contact/landing with continuous soft_landing gradient
    - Drop fuel_cost entirely (v6's 63.7% magnitude killer)
    - Keep mild attitude penalty and crash guard
    """
    ox, oy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ── 1. Progress: Euclidean distance delta (v5's proven backbone) ──
    old_dist = (ox**2 + oy**2) ** 0.5
    new_dist = (nx**2 + ny**2) ** 0.5
    progress_reward = 30.0 * (old_dist - new_dist)

    # ── 2. Descent bonus: reward downward movement ──
    # Dense per-step signal. Hovering or ascending gives zero — no exploit.
    dy = oy - ny
    descent_bonus = 5.0 * max(0.0, dy)

    # ── 3. Attitude penalty: keep it level ──
    attitude_penalty = 0.5 * (n_angle**2) + 0.1 * (n_av**2)

    # ── 4. Soft landing: continuous gradient for contact + low speed + upright ──
    # Uses multiplication: each factor (contact, speed, angle) gates the reward,
    # providing gradient even for partial fulfillment
    both_legs = n_left * n_right
    speed = (nvx**2 + nvy**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 5.0 * speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(n_angle))
    soft_landing = 25.0 * both_legs * speed_factor * angle_factor

    # ── 5. Crash guard: penalize high-speed ground contact ──
    any_contact = (n_left + n_right) > 0.0
    crash_penalty = speed if (any_contact and speed > 2.0) else 0.0

    total_reward = (
        progress_reward
        + descent_bonus
        - attitude_penalty
        + soft_landing
        - crash_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "descent_bonus": descent_bonus,
        "attitude_penalty": -attitude_penalty,
        "soft_landing": soft_landing,
        "crash_penalty": -crash_penalty,
    }

    return float(total_reward), components
```