Now I have a complete picture. Let me do the systematic component analysis before writing code.

---

## Component Analysis (Iteration 6)

### 1. `progress_reward` — Weak, net negative signal
- ep_sum_mean: -5.03 (trend: -25.6 → -5.2, improving but still negative)
- magnitude_share: 35.2%, active: 100%
- **Diagnosis**: Agent moves AWAY from target on net. The progress signal alone is too weak to overcome other penalties pulling the agent away. Needs a complementary *positional* reward.

### 2. `velocity_penalty` — Creates approach barrier
- ep_sum_mean: -17.28, magnitude_share: 36.0%, active: 32.7%
- **Diagnosis**: Proximity-gated (within 2m). Whenever the agent gets close to the target, it's punished for having speed. The agent learns to *avoid the target*. This is the **root cause** of why the agent never reaches the platform. Compare iter 5: velocity_penalty was -0.14 with 3.3% active — almost nonexistent.

### 3. `orientation_penalty` — Mild, flat
- ep_sum_mean: -7.34, magnitude_share: 15.3%, active: 100%
- Flat trend across checkpoints. Not harmful, not helpful. Keep but reduce weight.

### 4. `fuel_penalty` — Mild drag
- ep_sum_mean: -6.50, magnitude_share: 13.5%, active: 57.4%
- Flat trend. Fine as a regularizer.

### 5. `stable_contact_reward` — **Dead** (active_rate: 0.0%)
- Agent never achieves double-contact near target. Cannot be the problem — it's a consequence of the approach barrier.

### 6. `touchdown_bonus` — **Dead** (active_rate: 0.0%)
- Same: agent never reaches the platform, so transition never fires.

### Cross-iteration comparison
Iteration 5 (score **+171.40**, the best) proved the agent CAN reach the platform:
- It used a simple per-step touchdown reward (exploited by bouncing: 2287 ep_sum, 7.9% active_rate)
- Progress was positive (+12.0)
- Velocity penalty was negligible (-0.14)
- The exploit was real, but the approach gradient *worked*

Iteration 6 "fixed" the exploit by adding transition detection, but also kept the proximity-gated velocity penalty, which created a **wall of pain** — agent is punished for approaching.

### Intervention plan
1. **Remove proximity-gated velocity penalty entirely** — it's the approach barrier
2. **Add proximity reward** (`1/(1+dist)`) — positive everywhere, strongest near target
3. **Keep transition-based touchdown** (non-exploitable, large one-time bonus)
4. **Add sustained contact reward** — per-step reward on platform
5. **Speed penalty only ON platform** — prevents sliding after landing
6. **Keep mild orientation + fuel penalties**

---

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

    # === A. Proximity: dense positive shaping toward target ===
    # Always active, grows smoothly as agent approaches.
    proximity_reward = 2.5 / (1.0 + next_dist)

    # === B. Progress: bonus for reducing distance this step ===
    progress_reward = 4.0 * (dist - next_dist)

    # === C. Touchdown: one-time transition bonus for first platform contact ===
    prev_on_platform = (lc > 0.5 and rc > 0.5 and dist < 0.5)
    on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    touchdown_bonus = 0.0
    if not prev_on_platform and on_platform:
        touchdown_bonus = 20.0

    # === D. Sustained contact: per-step reward while on platform ===
    contact_reward = 4.0 if on_platform else 0.0

    # === E. Speed penalty: only active while ON platform (discourage sliding) ===
    next_speed = (nxv**2 + nyv**2) ** 0.5
    speed_penalty = -1.5 * next_speed if on_platform else 0.0

    # === F. Orientation penalty: discourage tilt ===
    orientation_penalty = -0.15 * abs(nangle)

    # === G. Fuel cost: mild regularization for engine use ===
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