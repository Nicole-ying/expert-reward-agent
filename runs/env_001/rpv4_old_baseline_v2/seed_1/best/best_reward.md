## Component Analysis

### 1. `contact_reward` (onset-based, 62.8% mag share, 2.1% active, ep_sum=76)

This is **the root cause of failure**. The onset-based design (`left_onset`/`right_onset` firing only on 0→1 transitions) gives the agent a one-time bonus of ~76 per episode, then provides *zero* per-step incentive to maintain contact. The agent touches down briefly, collects the onset, then drifts/crashes — hence all 20 episodes terminate at ~75 steps. 

**Evidence**: Iter 3 used per-step contact (`10*l + 10*r + 30*both = 50/step`) and achieved score=52.52 with 29.5% active rate and 507-step episodes. Iter 6 (onset-based) collapsed to score=-47.43 with 2.1% active rate and 75-step episodes. The per-step contact reward was the key driver that kept the agent on the pad.

**Fix**: Restore per-step contact reward. This is the ONE component change needed.

### 2. `distance_penalty` (-0.5 × dist, 28.5% mag share, 100% active)

Steady, proven in iter 3/6. No change needed.

### 3. `vel_penalty` (-0.1 × (vx²+vy²), 6.8% share)

Reasonable magnitude, always active. Keep.

### 4. `success_bonus` (2.0 conditional, 1.2% share, 0.9% active)

Recently discovered (jumped from 0.2→1.4). Per history warning: **L1 tuning only, do not change form**. Keep as-is.

### 5. `angle_penalty` / `angvel_penalty` (<1% share each)

Minor shaping, keep unchanged.

---

**Conclusion**: Rule (1) applies — score is negative and checkpoints are zero. The entire reward collapsed because of one broken component. I restore per-step `contact_reward` (the proven iter 3 pattern, scaled down to reduce dominance from 97.8% to a healthier range). All other components stay exactly as in iter 6.

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

    # Per-step contact reward — restored from iter 3 pattern, scaled down to reduce dominance
    # Gives steady incentive to MAINTAIN contact, not just initiate it
    contact_reward = 5.0 * left_contact + 5.0 * right_contact + 15.0 * (left_contact * right_contact)

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