Now let me systematically analyze each component of the current reward, then rewrite.

---

## Component-by-Component Analysis

### (1) Score assessment — **fundamentally broken, full rewrite required**

Score = **-143.49**, far below target of 200. The agent went from iter 1 (score=-17.9, len=1000, 0 terminated — hovering forever) to iter 2 (score=-143.49, len=590, 19/20 terminated — crashing early). The reward function made things **worse**. Rule (1) applies: full rewrite with a different signal framework.

### (2) Individual component diagnosis

| Component | ep_sum | Issue | Verdict |
|---|---|---|---|
| `altitude_proximity` | +61.64 (46%) | Dominant positive but rewards just being near ground regardless of quality | Merge into unified proximity |
| `horizontal_centering` | +25.13 (19%) | Gated by height, moderate gradient | Merge into unified proximity |
| `engine_cost` | -21.98 (17%) | Pure penalty, accumulated over 590 steps | Keep but milder |
| `step_cost` | -17.72 (13%) | **Pure penalty with zero behavioral gradient** — punishes time but provides no useful signal | **REMOVE** |
| `speed_penalty` | -4.11 (3%) | Too weak to prevent hard landings | Replace with product-form quality |
| `descent_reward` | +2.86 (2%) | **Critically weak** — only ~0.95 total descent units over entire episode | **10x multiplier** |
| `attitude_penalty` | -0.14 (0.1%) | Negligible — agent maintains good attitude | Fold into landing quality |
| `contact_bonus` | **0.00 (0%)** | **NEVER triggers** — agent never achieves two-leg contact. This is the root failure | Keep but surround with gradient |

### (3) Root cause

The agent learned to terminate (crash) quickly but never achieved a proper two-legged landing. The step_cost and engine_cost punish it for the 590 steps it takes to crash. The descent_reward is too weak to drive purposeful descent. The contact_bonus has no gradient leading up to it — the agent never discovers it.

### (4) Intervention plan

**New skeleton**: Phase-structured reward matching the `goal_approach_and_soft_contact` dynamics:
1. **Strong descent incentive** — drive purposeful downward motion
2. **Unified proximity reward** — replace altitude+horizontal with one distance-based signal
3. **Product-form landing quality** — near_ground × speed_quality × attitude_quality, provides gradient toward soft landing
4. **Large contact bonus** — simplified, bigger, but scaled by quality
5. **Mild engine cost** — kept but reduced
6. **REMOVE step_cost** — no behavioral value

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- A: Descent Incentive (strong, per-step downward progress) ----------
    descent = y_prev - y
    descent_reward = 10.0 * max(0.0, descent)

    # ---------- B: Target Proximity (unified position gradient toward (0,0)) ----------
    dist = (x * x + y * y) ** 0.5
    proximity_reward = 2.0 / (1.0 + dist)

    # ---------- C: Landing Quality (product of soft-landing conditions) ----------
    near_ground = 1.0 / (1.0 + 5.0 * abs(y))
    speed_quality = 1.0 / (1.0 + 3.0 * (vx * vx + vy * vy))
    attitude_quality = 1.0 / (1.0 + 2.0 * angle * angle)
    landing_quality = near_ground * speed_quality * attitude_quality

    # ---------- D: Contact Completion Bonus (large, one-time, quality-scaled) ----------
    contact = left_contact * right_contact
    contact_bonus = 60.0 * contact * speed_quality * attitude_quality

    # ---------- E: Engine Efficiency (mild) ----------
    if action == 2:
        engine_cost = -0.08
    elif action == 1 or action == 3:
        engine_cost = -0.03
    else:
        engine_cost = 0.0

    total_reward = (
        descent_reward
        + proximity_reward
        + landing_quality
        + contact_bonus
        + engine_cost
    )

    components = {
        'descent_reward': descent_reward,
        'proximity_reward': proximity_reward,
        'landing_quality': landing_quality,
        'contact_bonus': contact_bonus,
        'engine_cost': engine_cost,
    }

    return float(total_reward), components
```