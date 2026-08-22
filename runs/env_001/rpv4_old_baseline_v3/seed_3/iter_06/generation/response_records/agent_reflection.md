# Response Record

Now I have a complete picture. Here's my analysis:

---

## Component Analysis

### 1. `touchdown_reward` — **EXPLOITED. Must restructure.**

| Evidence | Value |
|---|---|
| magnitude_share | **99.2%** |
| ep_sum_mean at cp100% | **2287.5** (≈ 45 touchdowns × 50) |
| active_rate | 7.9% and growing |

**Diagnosis**: The checkpoint trend reveals the exploit pattern clearly. At cp30%, touchdown_reward = 0 (agent hadn't learned to land). At cp40%, it jumps to 1765 — agent discovered the touchdown bonus. By cp100%, it's at 2287.5, meaning the agent is triggering ~45 touchdowns per episode by **bouncing on/off the platform**. Each bounce cycle: lift off → lose contact → land again → collect another 50. This is the classic "reward hacking" pattern where the lump-sum bonus dwarfs all other signals (99.2% magnitude share). The external score stagnates at 157.5 (cp100%) precisely because bouncing undermines the true objective of stable landing.

### 2. `velocity_penalty` — **DEAD. Replace mechanism.**

| Evidence | Value |
|---|---|
| magnitude_share | **0.0%** |
| active_rate | 3.3% and flat |
| ep_sum_mean | -0.197 |

**Diagnosis**: The proximity gate (`dist < 0.5`) is far too narrow. The agent rarely spends time in this zone except during the brief touchdown instants. Combined with generous `safe_vx=0.2, safe_vy=0.3` thresholds and the tiny `w_vel_penalty=2.0`, this component contributes nothing. In iter_1 it had 100% active_rate (ungated) but was abandoned. The gating killed it.

### 3. `progress_reward` — **Stable but irrelevant.**

| Evidence | Value |
|---|---|
| magnitude_share | 0.8% |
| active_rate | 98.8% |
| ep_sum_mean | ~12 |

**Diagnosis**: Structurally fine — always active, provides consistent directional signal. But at 0.8% magnitude share, it can't compete with the touchdown exploit. Once the touchdown exploit is fixed, this component will regain proportional influence.

---

## Root Cause

The 50-point repeatable touchdown bonus creates a **bouncing exploit**. The agent discovered that bouncing earns more reward than stable landing. All other components are irrelevant because they collectively contribute <1% of the reward signal.

## Intervention: Restructure from lump-sum to per-step contact reward

The core fix: replace the repeatable 50-point transition bonus with a **per-step sustained-contact reward** plus a tiny transition bonus. This makes staying on the platform strictly more rewarding than bouncing (bouncing loses contact-reward during airborne steps).

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    xv, yv = obs[2], obs[3]
    angle = obs[4]
    lc, rc = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nlc, nrc = next_obs[6], next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # ======= A. Progress: delta-distance toward target =======
    w_progress = 8.0
    progress_reward = w_progress * (dist - next_dist)

    # ======= B. Velocity penalty: proximity-gated speed cost =======
    # Active within 2.0m of target, ramps up as agent gets closer
    vel_proximity = max(0.0, 1.0 - next_dist / 2.0)
    next_speed = (nxv**2 + nyv**2) ** 0.5
    w_vel = 2.0
    velocity_penalty = -w_vel * vel_proximity * next_speed

    # ======= C. Stable contact: per-step reward for sustained platform contact =======
    # Only rewards double-contact when near the target (prevents fake landings far away)
    on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    w_contact = 5.0
    stable_contact_reward = w_contact if on_platform else 0.0

    # ======= D. Touchdown transition: small one-time signal for achieving contact =======
    prev_on_platform = (lc > 0.5 and rc > 0.5 and dist < 0.5)
    next_on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    touchdown_bonus = 0.0
    if not prev_on_platform and next_on_platform:
        soft_landing = abs(nxv) < 0.5 and abs(nyv) < 0.4 and abs(nangle) < 0.3
        touchdown_bonus = 8.0 if soft_landing else 3.0

    # ======= E. Orientation penalty: discourage tilt =======
    w_orient = 0.3
    orientation_penalty = -w_orient * abs(nangle)

    # ======= F. Fuel efficiency: small cost for engine use =======
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = (
        progress_reward
        + velocity_penalty
        + stable_contact_reward
        + touchdown_bonus
        + orientation_penalty
        + fuel_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "stable_contact_reward": stable_contact_reward,
        "touchdown_bonus": touchdown_bonus,
        "orientation_penalty": orientation_penalty,
        "fuel_penalty": fuel_penalty,
    }
    return float(total_reward), components
```

## Why this fixes the exploit

| Behavior | Old reward (bouncing) | New reward (stable) |
|---|---|---|
| Bounce cycle: 10 steps airborne, 1 step landing | 10×0 + 1×50 = 50 | 10×0 + 1×3 = 3 (hard) or 8 (soft) |
| Stable: 10 steps on platform | 0 (no transition) | 10×5 = 50 |

Under the new function, **stable contact pays 50 over 10 steps while a bounce cycle pays at most 8**. The agent is strongly incentivized to stay put. The velocity penalty also now has teeth — with `vel_proximity` active within 2.0m (vs old 0.5m) and direct speed penalty, the agent must learn to approach slowly.
