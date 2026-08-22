# Response Record

Now I have a complete picture. Let me analyze each component systematically:

---

## Component Analysis

### Step 1: Score assessment — Rule (1) triggered
**Score = -46.32 (negative), checkpoints both at 0.00.** The current reward function is fundamentally broken. The agent learned nothing — it just hovers to collect `proximity` reward. I must rewrite the main signal framework.

### Step 2: Per-component diagnosis

| Component | Ep Sum | Magnitude Share | Active Rate | Diagnosis |
|---|---|---|---|---|
| `proximity` | 307.99 | **97.2%** | 100% | **ROOT CAUSE.** `0.5/(1+curr_dist)` pays ~0.3/step regardless of behavior. Agent collects 308 points just by existing for 1000 steps. This is a participation trophy — it completely masks all other signals. Must be removed. |
| `progress` | 4.67 | 1.7% | 100% | Trivially small. `5.0*(prev_dist-curr_dist)` gives tiny reward even when moving. Drowned out by proximity. |
| `vel_penalty` | -1.86 | 0.6% | 100% | Trivially small. |
| `angle_penalty` | -1.47 | 0.5% | 100% | Trivially small. |
| `angvel_penalty` | -0.03 | 0.0% | 100% | Negligible. |
| `contact_reward` | **0.00** | 0.0% | **0%** | **Agent never touches ground.** Coefficients (0.5/0.5/1.0) are far too weak to incentivize landing. |
| `success_bonus` | **0.00** | 0.0% | **0%** | Never fires — agent never lands. |

### Step 3: Cross-reference with history
- **Iter 3 (best, score=52.52, 16/20 landed):** Used `distance_penalty=-0.5*curr_dist` (ep_sum=-129) + per-step `contact_reward=10/10/30` (ep_sum=6100, 97.8% share). The contact reward was too dominant, but the structure *worked* — agent actually landed.
- **Iter 4→5:** Over-corrected contact dominance by slashing coefficients to 0.5/0.5/1.0, then switched to proximity-based signal. Both changes destroyed landing behavior.
- **Proximity warning:** It jumped from 0→308 but that's because it's a *constant per-step reward*, not a learned behavior. The mathematical form `0.5/(1+curr_dist)` is the problem, not the coefficient.

### Step 4: Intervention plan
**Return to iter 3's proven skeleton (distance_penalty + contact incentive) but fix the contact dominance by using contact ONSET (one-time bonus) instead of per-step accumulation.** This guarantees contact cannot dominate magnitude share because it fires at most once per leg.

---

## New Reward Design

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    px, py = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_left = obs[6]
    prev_right = obs[7]

    curr_dist = (px**2 + py**2)**0.5

    # Core drive toward origin (proven in iter 3, ~-130/ep)
    distance_penalty = -0.5 * curr_dist

    # One-time contact onset bonus — fires only on transition 0→1
    left_onset = 1.0 if (left_contact > 0.5 and prev_left < 0.5) else 0.0
    right_onset = 1.0 if (right_contact > 0.5 and prev_right < 0.5) else 0.0
    contact_reward = 40.0 * left_onset + 40.0 * right_onset + 80.0 * (left_onset * right_onset)

    # Per-step stable-landing bonus (small, only during successful touchdown)
    success_bonus = 2.0 if (left_contact and right_contact and curr_dist < 0.3 and abs(angle) < 0.2) else 0.0

    # Keep upright
    angle_penalty = -0.5 * (angle ** 2)

    # Moderate speed penalty
    vel_penalty = -0.1 * (vx ** 2 + vy ** 2)

    # Smooth rotation
    angvel_penalty = -0.05 * (ang_vel ** 2)

    total = (distance_penalty + contact_reward + success_bonus +
             angle_penalty + vel_penalty + angvel_penalty)

    components = {
        "distance_penalty": distance_penalty,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "vel_penalty": vel_penalty,
        "angvel_penalty": angvel_penalty
    }
    return float(total), components
```
