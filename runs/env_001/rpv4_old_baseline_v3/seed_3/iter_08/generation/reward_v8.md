Now I have a complete picture. Let me do the systematic analysis.

---

## Component Analysis

### (1) proximity_reward — **ROOT CAUSE: hovering trap via unbounded accumulation**

- Formula: `2.5 / (1.0 + next_dist)` — an **absolute** per-step reward, not potential-based
- ep_sum_mean: 1734.5, **84.5% magnitude share**, 100% active
- Checkpoint trend: 114 → 179 → 1184 → … → 1824 (steady monotonic growth throughout training)
- **Diagnosis**: When the agent reaches dist≈0, it receives ~2.5/step indefinitely. Over 934-step episodes, this accumulates to ~1700+, completely dwarfing every other signal. The agent has no incentive to land — hovering yields nearly the same per-step reward as landing (2.5 vs 4.0 from contact_reward), without risking episode termination. The checkpoint data confirms this: contact_reward was **zero** until cp100%, meaning the agent spent 90% of training just learning to hover close.
- **Intervention**: Convert to **potential-based difference form**: `C * (1/(1+next_dist) - 1/(1+dist))`. This bounds total accumulation to `C * (1 - 1/(1+initial_dist))` and gives **zero** reward when hovering at constant distance. No more hovering exploit.

### (2) contact_reward — Too weak to overcome hovering inertia

- ep_sum_mean: 253.4, 12.4% magnitude, 6.8% active
- Checkpoint: 0 until cp100%, then sudden jump to 303
- **Diagnosis**: At 4.0/step vs proximity's 2.5/step, the marginal gain from landing is only +1.5/step. Combined with the risk of episode termination upon landing, this is not enough incentive. The agent discovers landing very late and only in a minority of episodes.
- **Intervention**: Boost to **20.0/step** — once proximity no longer rewards hovering, contact must provide a clear, dominant incentive to land.

### (3) touchdown_bonus — Negligible one-time signal

- ep_sum_mean: 20.0, 1.0% magnitude, 0.1% active
- **Diagnosis**: A one-time 20.0 bonus is too small relative to the cumulative reward landscape. Even when triggered, it's a drop in the bucket.
- **Intervention**: Increase to **50.0** to provide a salient "you did it" spike upon first landing.

### (4) progress_reward — Fine, supplementary shaping

- ep_sum_mean: 5.18, 0.3% magnitude, 99.8% active
- Checkpoint: improved from -21 (cp10%) to +5 (cp100%)
- **Diagnosis**: Working as intended — provides a weak but consistent gradient during approach. Keep, slightly reduce coefficient to avoid double-counting with the new proximity difference signal.

### (5) speed_penalty — Fine but weak

- ep_sum_mean: -14.47, 0.7% magnitude, 6.8% active (only on platform)
- **Diagnosis**: Only active during landing, so it doesn't interfere with approach. Increase to **-5.0 * speed** to discourage sliding off the platform once landed.

### (6) orientation_penalty — Stable, keep

- Stable at ~-5/episode, small. No change needed.

### (7) fuel_cost — Stable, keep

- Stable at ~-18 to -20/episode. Mild regularization. Keep.

---

Now the code:

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nlc, nrc = next_obs[6], next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # Platform detection
    prev_on_platform = (obs[6] > 0.5 and obs[7] > 0.5 and dist < 0.5)
    on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    next_speed = (nxv**2 + nyv**2) ** 0.5

    # === A. Potential-based proximity: difference form prevents accumulation ===
    # Total contribution over full episode bounded to ~8 * (1 - 1/(1+init_dist))
    # Gives zero when hovering at constant distance — no more hovering exploit.
    proximity_reward = 8.0 * (1.0 / (1.0 + next_dist) - 1.0 / (1.0 + dist))

    # === B. Progress: linear distance reduction, supplementary approach signal ===
    progress_reward = 2.0 * (dist - next_dist)

    # === C. Touchdown: large one-time bonus for first platform contact ===
    touchdown_bonus = 50.0 if (not prev_on_platform and on_platform) else 0.0

    # === D. Sustained contact: strong per-step incentive to stay on platform ===
    contact_reward = 20.0 if on_platform else 0.0

    # === E. Speed penalty on platform: discourage sliding after landing ===
    speed_penalty = -5.0 * next_speed if on_platform else 0.0

    # === F. Orientation penalty: mild tilt discouragement ===
    orientation_penalty = -0.2 * abs(nangle)

    # === G. Fuel cost: light regularization for engine use ===
    fuel_cost = -0.02 if action != 0 else 0.0

    total = (proximity_reward + progress_reward + touchdown_bonus +
             contact_reward + speed_penalty + orientation_penalty + fuel_cost)

    components = {
        "proximity_reward": proximity_reward,
        "progress_reward": progress_reward,
        "touchdown_bonus": touchdown_bonus,
        "contact_reward": contact_reward,
        "speed_penalty": speed_penalty,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
    }
    return float(total), components
```