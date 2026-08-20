## Reflection

### (1) Score assessment — **fundamentally broken, needs full rewrite**

Score = **-211.11**, target = 200, gap = 411. This is the **worst** of all 3 iterations. The agent never achieves a single both-leg contact (contact_bonus = 0 across all checkpoints, all iterations). All "improvement" from iter 1 (-17.9) to iter 3 (-211) is learning to **crash faster**, not land.

The root cause is clear: `proximity_reward = 2.0/(1+dist)` dominates at 82.5% magnitude share (ep_sum=540), creating a perverse **"hover and collect"** incentive. The agent learns to loiter at moderate distance (~1.44 units away) forever, collecting ~0.82/step. Since landing terminates the episode and stops this income stream, the agent actively avoids landing.

### (2) Component analysis

| Component | Share | Active | Diagnosis | Verdict |
|---|---|---|---|---|
| `proximity_reward` | 82.5% | 100% | Per-step living reward. **THE root cause.** Agent exploits this by hovering. | **Replace entirely** with potential-based shaping |
| `descent_reward` | 0.2% | 50.5% | Drowned out. Agent goes UP half the time! Coefficient 10× too small relative to proximity. | **Multiply coefficient, keep** |
| `landing_quality` | 11.6% | 100% | Per-step product form. Never reaches high values because agent stays far from ground. | **Replace** with harsh near-ground velocity/attitude penalties |
| `contact_bonus` | 0% | 0% | `left*right` product never fires. No gradient to help agent discover landing. | **Rewrite**: use `left+right` for partial credit, much larger bonus |
| `engine_cost` | -5.7% | 91% | Reasonable magnitude. | **Keep**, minor tweak |

### (3) Flat-lining check

- `contact_bonus`: 0.0 across ALL iterations and ALL checkpoints. Completely flat. The agent has NEVER experienced this signal.
- `descent_reward` at 40% checkpoint: 0.348 (1.8% active). At 80%: 2.568 (63.5% active). Some learning, but ep_sum still negligible.

### (4) Magnitude imbalance

`proximity_reward` at 82.5% magnitude share vs `descent_reward` at 0.2%. This is a 400:1 ratio. The agent is effectively single-objective: maximize proximity_reward. All other signals are noise.

### (5) Intervention strategy

**Complete rewrite** with these principles:

1. **No per-step "living" rewards**. Replace proximity with potential-based shaping: `reward = coeff * (prev_dist - curr_dist)`. Moving toward target gives positive reward; staying still gives zero. This removes the hover incentive entirely.

2. **Strong descent incentive**. `20 * max(0, y_prev - y)` — make descending the primary per-step reward.

3. **Harsh near-ground velocity/attitude penalties**. Gate by height so they only activate near the platform, forcing slow, controlled final approach.

4. **Discoverable contact gradient**. Use `left_contact + right_contact` instead of `left * right`. This gives 50% reward for one-leg contact, creating a gradient from 0→1→2 legs that the agent can follow.

5. **Massive landing bonus**. 80 per leg × quality factor, giving the agent a clear target to optimize toward.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- A: Approach (potential-based, no living reward) ----------
    # Reward MOVING toward (0,0), not BEING near (0,0).
    # Zero reward for hovering — removes the "hover and collect" exploit.
    prev_dist = (x_prev * x_prev + y_prev * y_prev) ** 0.5
    curr_dist = (x * x + y * y) ** 0.5
    approach_reward = 5.0 * (prev_dist - curr_dist)

    # ---------- B: Descent (strong, per-step downward progress) ----------
    descent = y_prev - y
    descent_reward = 20.0 * max(0.0, descent)

    # ---------- C: Soft Landing Velocity Penalty (gated by height) ----------
    near_ground = 1.0 / (1.0 + 3.0 * abs(y))
    vel_penalty = -5.0 * near_ground * (vx * vx + vy * vy)

    # ---------- D: Attitude Penalty (gated by height) ----------
    attitude_penalty = -2.0 * near_ground * (angle * angle + 0.3 * angvel * angvel)

    # ---------- E: Contact Reward (discoverable gradient via sum, not product) ----------
    contact_sum = left_contact + right_contact
    contact_quality = 1.0 / (1.0 + 5.0 * (vx * vx + vy * vy) + 5.0 * angle * angle + 3.0 * abs(x))
    contact_reward = 80.0 * contact_sum * contact_quality

    # ---------- F: Engine Cost (mild) ----------
    if action == 2:
        engine_cost = -0.05
    elif action == 1 or action == 3:
        engine_cost = -0.02
    else:
        engine_cost = 0.0

    total_reward = (
        approach_reward
        + descent_reward
        + vel_penalty
        + attitude_penalty
        + contact_reward
        + engine_cost
    )

    components = {
        'approach_reward': approach_reward,
        'descent_reward': descent_reward,
        'vel_penalty': vel_penalty,
        'attitude_penalty': attitude_penalty,
        'contact_reward': contact_reward,
        'engine_cost': engine_cost,
    }

    return float(total_reward), components
```