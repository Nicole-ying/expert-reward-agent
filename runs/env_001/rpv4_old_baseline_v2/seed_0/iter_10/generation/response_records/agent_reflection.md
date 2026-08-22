# Response Record

## Reflection on Current Reward Components

### Overall score assessment
The evaluation score is strongly negative (`-63.45`) with all episodes truncated (length 1000) and the agent **never reaching the pad** (`contact_reward` and `success_bonus` both zero). This means the current reward fundamentally fails to promote the primary goal (reaching and softly landing on the pad). According to rule (1) we must rewrite the reward from a different signal framework.

### Per‑component analysis

| Component | Issue | Action |
|-----------|-------|--------|
| `speed_reward` (38% magnitude share) | Strongly rewards low absolute velocity at all times. The agent can collect this by doing nothing or hovering, without ever approaching the pad. | **Remove** – it incentivises stalling behaviour that opposes the goal of “as fast as possible”. |
| `angle_reward` (38% magnitude share) | Same problem: rewards near‑zero tilt everywhere. The agent keeps itself upright in mid‑air rather than attempting a landing. | **Remove** – only the final landing attitude matters. |
| `proximity_reward` (24%) | Gives a mild signal toward the pad centre, but its effect is completely swamped by the cheap “stay still” rewards from `speed_reward` and `angle_reward`. | **Keep** but reshape to provide stronger gradient when close. Use `10/(0.5+dist)` so the maximum rises from 1 to 20, making proximity the dominant dense signal. |
| `contact_reward` (0% active) | Almost no incentive to touch the ground (only 0.2 per leg). The agent never learns to make contact. | **Scale up drastically** → `5.0` per leg. This makes landing a high‑value event while keeping it dense. |
| `fuel_penalty` | Works correctly and is tiny, but alone cannot shape landing. | **Keep**, slightly scale to `‑0.05` per non‑zero action to encourage efficiency without preventing exploration. |
| `success_bonus` (0% active) | A huge sparse bonus (200) that is never triggered because the dense rewards trap the agent in hovering. A sparse success bonus is forbidden in the code constraints anyway. | **Remove** (also rule‑bound). Replace with a dense **`settle_reward`** that fires every step once both legs touch AND speeds/tilt are low – this gives strong, immediate reinforcement for a soft landing. |

### Missing ingredient
The agent needs a direct incentive to **reduce distance**, especially early in training when the proximity signal is weak. I introduce a **`progress_reward`**: a small bonus proportional to the reduction in Euclidean distance from one step to the next. This makes every action that moves toward the pad rewarding, encouraging exploration instead of freezing.

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # Distance to pad centre (current step, previous step)
    new_dist = (nx * nx + ny * ny) ** 0.5
    old_dist = (x * x + y * y) ** 0.5

    # 1. Dense proximity – strongly rewards being on/near the pad
    proximity_reward = 10.0 / (0.5 + new_dist)

    # 2. Contact – large reward for each leg touching the ground
    contact_reward = 5.0 * (nlc + nrc)

    # 3. Settle – continuous high reward once soft‑landing conditions hold
    settled = (
        nlc > 0.5 and nrc > 0.5 and
        abs(nvx) < 0.2 and abs(nvy) < 0.2 and
        abs(nangle) < 0.3
    )
    settle_reward = 10.0 if settled else 0.0

    # 4. Progress – small incentive to move toward the pad every step
    progress_reward = 0.1 * (old_dist - new_dist)

    # 5. Fuel penalty – gentle pressure not to fire engines unnecessarily
    fuel_penalty = -0.05 if action != 0 else 0.0

    total = proximity_reward + contact_reward + settle_reward + progress_reward + fuel_penalty

    components = {
        "proximity_reward": proximity_reward,
        "contact_reward": contact_reward,
        "settle_reward": settle_reward,
        "progress_reward": progress_reward,
        "fuel_penalty": fuel_penalty,
    }
    return float(total), components
```
