# Analysis of Existing Reward Components

The current score is deeply negative (–105.8) despite about half the episodes receiving the 200‑point success bonus. The data clearly point to a single culprit: the `shaping_reward` produces massive negative totals (up to several hundred points) on the episodes that crash or drift away. This dominates the final score and makes the reward function fundamentally broken for learning. Below is the component‑by‑component analysis following the prescribed reflection order.

1. **success_bonus**  
   - `magnitude_share` 97.8% – expected for a sparse terminal reward of 200.  
   - `active_rate` 0.6% shows it fires only on successful landing steps.  
   - The bonus itself is not the source of the negative score; rather, the processes that happen *before* failure dig a hole that even 200 points cannot fill.  

2. **shaping_reward**  
   - `episode_sum_mean` is only +1.2, but this average hides extreme negative values on failing episodes.  
   - The formula `potential = –(1.0·dist + 0.5·|vx|+|vy| + 0.5·|angle|)` and the difference \( V_{next} – V_{current} \) produce **unbounded negative steps** whenever the agent accelerates away or falls over.  
   - Over ~100 steps these negatives accumulate to hundreds of points of penalty, crashing the total return.  
   - **Conclusion: the root cause of the –105.8 score is this component. It must be removed completely and replaced by a positive‑only shaping scheme.**

3. **contact_bonus**  
   - Very small (+0.36 mean), active only 2.3% of steps. It cannot compensate for the large negative shaping.  
   - It can be kept but should be augmented by other positive signals.

4. **fuel_penalty**  
   - Negligible (–0.32 mean) and not harmful, though it adds a tiny constant drag. Keeping it is acceptable.

Because the overall score has been stuck deeply negative, rule (1) applies: **the reward function is wrong at its core and must be rewritten with a completely different main signal framework.** The new framework will use only **non‑negative dense rewards** (proximity, low speed, upright angle, foot contact) so that failing episodes never suffer huge penalties, while the sparse `success_bonus` remains the dominant driver toward the target score of 200.

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # Distance to the pad centre
    dist = (nx * nx + ny * ny) ** 0.5
    # Total linear speed
    speed = abs(nvx) + abs(nvy)
    # Absolute tilt angle
    tilt = abs(nangle)

    # 1. Proximity reward: always >= 0, highest when on the pad
    proximity_reward = 1.0 / (1.0 + dist)

    # 2. Speed reward: encourages gentle touchdown
    speed_reward = 1.0 / (1.0 + speed)

    # 3. Orientation reward: favours staying upright
    angle_reward = 1.0 / (1.0 + 5.0 * tilt)

    # 4. Contact bonus: tiny reward for each leg touching the ground
    contact_reward = 0.2 * (nlc + nrc)

    # 5. Fuel penalty: minuscule cost to promote efficiency
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 6. Sparse success bonus – unchanged loose condition from previous function
    success = (
        nlc > 0.5 and nrc > 0.5 and
        ny < 0.3 and
        tilt < 0.3 and
        abs(nvx) < 0.2 and abs(nvy) < 0.2
    )
    success_bonus = 200.0 if success else 0.0

    total = proximity_reward + speed_reward + angle_reward + contact_reward + fuel_penalty + success_bonus

    components = {
        "proximity_reward": proximity_reward,
        "speed_reward": speed_reward,
        "angle_reward": angle_reward,
        "contact_reward": contact_reward,
        "fuel_penalty": fuel_penalty,
        "success_bonus": success_bonus,
    }
    return float(total), components
```