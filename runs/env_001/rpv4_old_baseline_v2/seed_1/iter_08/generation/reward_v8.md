## Component Analysis

### 1. `contact_reward` — **Rule (3) triggered: 91.6% magnitude share is extreme**

The history shows a clear pattern: when contact_reward works (iter_03 at 97.8%, iter_07 at 91.6%), the agent learns. But the signal completely dominates — over 90% of the reward budget comes from one component. The agent is essentially being told "touch the ground, stay there, ignore everything else." Distance, velocity, angle — all are noise relative to the contact signal. Per Rule (3), this **must** be scaled down.

I'll reduce from `5/5/15` to `2/2/6` (2.5× reduction). This preserves the per-step structure that was the key breakthrough from iter_06→iter_07, while bringing the magnitude share from ~92% down to an estimated ~78-82%. The distance_penalty and other shaping terms will naturally gain relative influence.

### 2. `angvel_penalty` — **Rule (2): flat at 0.0% magnitude share across all iterations**

This component has never exceeded 1.8% share. It contributes essentially zero signal. But per the "one component at a time" rule, I must address the dominant contact_reward first. The angvel_penalty can be revisited next iteration if it remains flat.

### 3. `distance_penalty` (5.1%), `angle_penalty` (0.2%), `vel_penalty` (0.3%), `success_bonus` (2.8%)

All healthy but currently dwarfed by contact_reward. They will naturally gain influence after the contact scaling. No changes needed.

### 4. Checkpoint scores at 0.0

The agent learns nothing until the last 20% of training because contact_reward is the only signal that matters, and it's only discovered late through random exploration. After scaling contact down, the always-active distance_penalty provides a consistent gradient throughout training, which should improve intermediate checkpoints.

---

**Change:** Reduce `contact_reward` coefficients from `5.0/5.0/15.0` → `2.0/2.0/6.0`.

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

    # Per-step contact reward — scaled down 2.5x from iter 7 to reduce 91.6% dominance
    # Gives steady incentive to MAINTAIN contact, not just initiate it
    contact_reward = 2.0 * left_contact + 2.0 * right_contact + 6.0 * (left_contact * right_contact)

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