# Response Record

Now I have the full picture. Let me do the systematic component analysis.

---

## Component-by-Component Diagnosis (v8, current iteration)

### 1. `progress_reward` — **THE HIDDEN KILLER** (mag=66.9%, ep=33.78, active=100%)

Formula: `30.0 * (old_dist - new_dist)` — pure Euclidean delta.

This is the critical regression from v5. v5 used a **potential-based** formula `15.0 * ((-ndist - 0.2*nspeed_sq) - (-dist - 0.2*speed_sq))` that simultaneously rewards approach AND penalizes high speed. The v8 formula rewards ANY movement toward (0,0) at ANY speed — it literally pays the agent to crash at terminal velocity. The checkpoint trend is flat (~33.7 across all checkpoints), confirming this is just background noise, not a learning signal.

**Action: Restore v5's potential-based formula exactly.**

### 2. `descent_bonus` — **REDUNDANT ACCELERANT** (mag=13.8%, ep=7.17, active=94.3%)

Formula: `5.0 * max(0.0, dy)`. Flat trend across all checkpoints (~7.17). This adds fuel to the "dive bomb" fire — it's redundant with progress (going toward origin IS going down) and further incentivizes high-speed crashing.

**Action: Drop entirely.** The potential function covers vertical progress.

### 3. `attitude_penalty` — **FINE BUT SLIGHTLY WEAK** (mag=2.9%, ep=-1.53, active=100%)

Formula: `0.5 * angle^2 + 0.1 * av^2`. v5 used `0.15 * av^2` — slightly stronger angular velocity damping. Stable across checkpoints, no exploitation.

**Action: Restore v5's coefficient (0.15 on av^2).**

### 4. `landing_bonus` — **UNREACHABLE** (mag=16.2%, ep=8.46, active=1.1%)

One-shot transition bonus. Fires on only 1.1% of steps because the agent crashes before achieving gentle contact. The bonus itself is well-structured, but the agent never reaches the state where it can fire.

**Action: Keep structure, reduce magnitude to ~20, will become reachable once potential backbone slows the agent.**

### 5. `stability_reward` — **DEAD** (mag=0.2%, ep=0.09, active=1.5%)

Flat at ~0.1 across all checkpoints. Too small to matter, fires too rarely.

**Action: Drop entirely.**

---

## Root Cause Summary

v5 (score=146, len=964) worked because the potential function `-dist - 0.2*speed²` creates a **natural speed limit**: accelerating toward the goal makes the potential WORSE. This is proper reward shaping — it doesn't distort the optimal policy.

v8 (score=-122, len=68) fails because `30*(old_dist - new_dist)` + `5*dy` pays the agent to crash. All 20 episodes terminate early. The agent learned exactly what we taught it: go down fast.

**Fix: Restore v5's potential backbone, drop descent_bonus, keep landing_bonus (now reachable), add crash_penalty as guard.**

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v9: Restore v5's potential-based progress backbone.
    - Potential = -distance - 0.2 * speed^2: rewards approach AND penalizes speed
    - Drop descent_bonus (redundant, encouraged crashing)
    - Keep one-shot landing_bonus (now reachable since agent slows down)
    - Add crash_penalty for high-speed ground contact as safety guard
    - Drop stability_reward (dead component)
    """
    ox, oy = obs[0], obs[1]
    ovx, ovy = obs[2], obs[3]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]
    o_left, o_right = obs[6], obs[7]
    n_left, n_right = next_obs[6], next_obs[7]

    # ── 1. Potential-based progress (v5's proven backbone) ──
    # Higher potential = closer + slower. Agent is rewarded for improving potential.
    old_dist = (ox**2 + oy**2) ** 0.5
    new_dist = (nx**2 + ny**2) ** 0.5
    old_speed_sq = ovx**2 + ovy**2
    new_speed_sq = nvx**2 + nvy**2

    pot = -new_dist - 0.2 * new_speed_sq
    prev_pot = -old_dist - 0.2 * old_speed_sq
    progress_reward = 15.0 * (pot - prev_pot)

    # ── 2. Attitude penalty (v5's coefficients) ──
    attitude_penalty = 0.5 * (n_angle ** 2) + 0.15 * (n_av ** 2)

    # ── 3. One-shot landing bonus: first transition to stable two-leg contact ──
    o_both = (o_left > 0.5 and o_right > 0.5)
    n_both = (n_left > 0.5 and n_right > 0.5)

    landing_bonus = 0.0
    if n_both and not o_both:
        speed = (nvx**2 + nvy**2) ** 0.5
        # Higher bonus for gentler touchdown
        landing_bonus = 20.0 * (1.0 / (1.0 + 3.0 * speed)) * (1.0 / (1.0 + 3.0 * abs(n_angle)))

    # ── 4. Crash penalty: quadratic penalty for high-speed ground contact ──
    any_contact = (n_left + n_right) > 0.0
    speed = (nvx**2 + nvy**2) ** 0.5
    crash_penalty = 0.0
    if any_contact and speed > 1.0:
        crash_penalty = 2.0 * (speed ** 2)

    total_reward = (
        progress_reward
        - attitude_penalty
        + landing_bonus
        - crash_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": -attitude_penalty,
        "landing_bonus": landing_bonus,
        "crash_penalty": -crash_penalty,
    }

    return float(total_reward), components
```
