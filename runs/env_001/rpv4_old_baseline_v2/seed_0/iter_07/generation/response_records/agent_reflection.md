# Response Record

## Analysis of Existing Components and Necessary Changes

**Overall score** is deeply negative (–117.7) with all episodes terminating early (<150 steps, score < –50). According to rule (1), when the score stays far below zero, the whole reward framework must be rewritten using a fundamentally different main signal. The current reward fails to lead the agent to a successful soft landing; instead, the agent crashes quickly. We will redesign the central signal around a proper soft‑landing bonus and fix the dominating velocity penalty.

**Per‑component analysis:**

1. **vel_penalty** (magnitude share 60%, mean –23.0, active only near ground)  
   This component overwhelmingly dominates the reward. The 0.4 coefficient applied to velocity near the ground produces a large negative signal that overwhelms progress.  
   *Decision: scale it down aggressively and keep the ground‑activation (so fast flight at altitude is not penalised), changing only its magnitude.*

2. **dist_reward** (mean +8.9, active 100%)  
   Potential‑shaping towards the target (0,0) is appropriate. Slightly raising the coefficient (e.g., from 8.0 to 10.0) will make approach more attractive.

3. **contact_bonus** (active only 2.9%, mean 3.4, sparse and strict)  
   The tight condition (both legs down, low height, small angle and velocities) almost never triggers. The agent needs a dense touch reward to discover ground contact, plus a large sparse success bonus.  
   *Decision: split into `touch_bonus` (reward per touching leg) and `success_bonus` (large one‑off reward when a stable landing is achieved).*

4. **angle_penalty** (mean –1.3, active 100%)  
   Small but always present. Slightly reduce its coefficient so it does not add unnecessary negative total.

5. **step_penalty** (mean –1.37, active 100%)  
   Fine as a gentle urge to finish quickly. Keep as is.

6. **fuel_penalty** (mean –0.033, active 4.8%)  
   Negligible. Keep a tiny penalty to discourage unnecessary engine use, but do not increase it.

With these changes the agent should be guided by a strong positional incentive, a speed penalty that only asks for deceleration near the pad, a dense contact signal, and a large terminal success reward capable of lifting the total return well above zero.

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # 1. Shaped distance-to-target reward (stronger)
    d_prev = (x * x + y * y) ** 0.5
    d_curr = (nx * nx + ny * ny) ** 0.5
    dist_reward = 10.0 * (d_prev - d_curr)

    # 2. Soft-landing velocity penalty – only active near the ground
    ground_factor = 1.0 if ny < 1.5 else 0.0
    vel_penalty = -0.1 * (abs(nvx) + abs(nvy)) * ground_factor

    # 3. Gentle angle penalty (always active)
    angle_penalty = -0.08 * abs(nangle)

    # 4. Light step penalty to encourage speed
    step_penalty = -0.02

    # 5. Tiny fuel penalty for any non-idle action
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 6. Dense touch bonus – reward every leg contact with the pad
    touch_bonus = 0.4 * (nlc + nrc)

    # 7. Sparse success bonus for a stable two‑legged landing
    success = (
        nlc > 0.5 and nrc > 0.5 and
        ny < 0.3 and
        abs(nangle) < 0.3 and
        abs(nvx) < 0.2 and abs(nvy) < 0.2
    )
    success_bonus = 100.0 if success else 0.0

    total = (
        dist_reward
        + vel_penalty
        + angle_penalty
        + step_penalty
        + fuel_penalty
        + touch_bonus
        + success_bonus
    )

    components = {
        "dist_reward": dist_reward,
        "vel_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "step_penalty": step_penalty,
        "fuel_penalty": fuel_penalty,
        "touch_bonus": touch_bonus,
        "success_bonus": success_bonus,
    }
    return float(total), components
```
