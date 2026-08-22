# Analysis of existing components

Given the final score of -68.15 (target +200), the current reward function is fundamentally misaligned with the true task. Stability reward dominates learning (81.7% magnitude share) but only activates 5.5% of the time – the agent learns to trigger a single-foot contact, collect the massive positive reward, and then stagnate there instead of achieving a proper two-footed, low-speed, upright landing. The descent and proximity signals are drowned out, so the agent never learns a controlled trajectory to the platform center. According to rule (1) we must replace the main signal entirely.

- **descent_reward** – tiny magnitude (1.6%), active 89% but far too weak to guide landing.
- **proximity_penalty** – modest negative signal, active always, but overshadowed by stability.
- **stability_reward** – too permissive (any contact), too large, and encourages early termination without a true landing. Must be replaced with a two-footed, high-quality landing bonus.
- **fuel_penalty** / **time_penalty** – sensible, but negligible impact; can be kept with minor tuning.

**New framework**  
Shape the entire approach with a potential-based distance reward that encourages moving toward the platform center, enforce smoothness with speed and posture penalties, and provide a large per-step bonus only when *both* legs are firmly on the platform and the vehicle is near upright with near-zero velocity. This make the optimal state continually rewarding, so the agent learns to seek and maintain it.

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next state
    old_x, old_y = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Approach reward: distance-based shaping toward target (0,0) ----
    old_dist = (old_x ** 2 + old_y ** 2) ** 0.5
    new_dist = (x ** 2 + y ** 2) ** 0.5
    approach_reward = (old_dist - new_dist) * 5.0   # positive when moving closer

    # ---- Smoothness penalty: discourage high speed and rotation ----
    speed_penalty = -0.2 * (vx ** 2 + vy ** 2) - 0.1 * (angle ** 2 + angvel ** 2)

    # ---- Landing reward: both legs on platform and vehicle stable ----
    if left_contact > 0.5 and right_contact > 0.5:
        # Quality of touchdown: near upright, negligible velocity
        landing_quality = 10.0 - 15.0 * angle ** 2 - 3.0 * vx ** 2 - 3.0 * vy ** 2 - 3.0 * angvel ** 2
        landing_reward = max(0.0, landing_quality)
    else:
        landing_reward = 0.0

    # ---- Fuel penalty: discourage unnecessary engine use ----
    fuel_penalty = -0.05 if action in [1, 2, 3] else 0.0

    # ---- Small per-step penalty to prevent lingering ----
    time_penalty = -0.01

    total = approach_reward + speed_penalty + landing_reward + fuel_penalty + time_penalty

    components = {
        "approach_reward": approach_reward,
        "speed_penalty": speed_penalty,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty,
        "time_penalty": time_penalty
    }
    return float(total), components
```