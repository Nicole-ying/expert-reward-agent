Now I have the full picture. Let me write a comprehensive analysis and then the new reward.

---

## Component Analysis

### 1. `contact_reward` — **DOMINANT but lottery (69.5% magnitude, 3.1% active)**
The checkpoint trend tells the story: 0 → 125 → 272 → 164 → 232 → **364** → **0** → 89. This oscillates wildly because contact is discovered by accident, creates a huge gradient spike (8.0 per leg × both legs × several steps = massive), then the policy overfits and forgets. The reward is 40-80× larger per firing step than any dense signal. **Verdict: the primary cause of instability. Must be drastically reduced.**

### 2. `landing_bonus` — **Same lottery pattern (14.2%, 1.9% active)**
Checkpoint: 0 → 67 → 150 → 3 → 33 → 130 → **0** → 37. Same wild swings. The product `lc_n * rc_n * near_origin` means it only fires when both legs touch near origin — even rarer than contact_reward. **Verdict: reduce heavily.**

### 3. `progress` — **The only dense driver, but PATHETICALLY weak (3.5%, 100% active)**
4.18 total episode sum with coefficient 10.0 means the agent reduces distance to origin by only **0.42 units** over the entire episode. It's barely moving toward the target. The signal exists everywhere but is drowned out by penalties. **Verdict: must be 3-5× stronger and unshackled from competing negatives.**

### 4. `descent_reward` — **Active but negligible (0.8%, 95.6% active)**
2.45 total. The triple-product gate (`x_near * descending * speed_ok`) means it's almost always tiny even when active. **Verdict: increase coefficient, remove speed_ok gate.**

### 5. `fuel_cost` + `step_cost` — **Steady drain (-10.3% combined, 97-100% active)**
Together they drain ~33 points per episode. This directly counteracts progress (4.18) and descent (2.45). The agent is punished for every step and every engine use, making it net-negative unless it wins the contact lottery. **Verdict: remove step_cost entirely, make fuel_cost symbolic.**

### 6. `x_bonus` — **NEGATIVE! (-0.6%, 100% active)**
-1.89 means the agent moves AWAY from x=0 on net. The signal is too weak to guide horizontal centering. **Verdict: remove; progress already covers this.**

### 7. `orient_cost` — **Reasonable regularizer (1.2%, 100% active)**
-3.78 total. Fine as a mild regularizer. Keep similar.

### 8. `slam_cost` — **DEAD (0%, 0% active)**
Threshold-based penalty that never fires. The -0.5 vy threshold is never exceeded in the evaluated trajectories. **Verdict: replace with continuous speed penalty near ground.**

---

### Root Cause

The reward has a **catastrophic scale mismatch**: sparse bonuses contribute ~268 when they fire (which is rare and unstable), while dense guidance signals contribute only ~7 total. The persistent penalties (~33) ensure the agent is deeply negative unless it wins the contact lottery. This creates a high-variance, unstable learning signal where the policy oscillates between "lucky landing" and "lost."

### Strategy for v4

**Make dense progress the absolute king** (target 200-300/episode). Reduce sparse bonuses to mild supplements (target ~10-20). Eliminate drain penalties. Use continuous speed penalty near ground to shape soft landing.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: Dense-dominated reward. Progress toward (0,0) is the main driver (~200-300/ep).
    Sparse contact/landing bonuses are mild supplements (~10-20/ep).
    No per-step drain. Speed penalty near ground for soft landing.
    """
    x_c, y_c = obs[0], obs[1]
    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n = next_obs[4]
    angvel_n = next_obs[5]
    lc_n = next_obs[6]
    rc_n = next_obs[7]

    # --- 1. progress: THE main signal. Strong potential-based gradient toward (0,0) ---
    # Telescoping sum = 30 * (dist_start - dist_end). From dist~10 to 0: ~300 reward.
    dist_c = (x_c ** 2 + y_c ** 2) ** 0.5
    dist_n = (x_n ** 2 + y_n ** 2) ** 0.5
    progress = 30.0 * (dist_c - dist_n)

    # --- 2. descent_reward: extra push to go DOWN when horizontally centered ---
    # No speed gate — speed is handled separately. Just reward descending.
    x_centered = max(0.0, 1.0 - abs(x_n) / 2.0)
    descending = max(0.0, y_c - y_n)
    descent_reward = 8.0 * x_centered * descending

    # --- 3. speed_cost: penalize high speed near ground (squared, so severe only when fast) ---
    # This is the soft-landing shaper. Strong above speed~1.0, negligible below ~0.3.
    near_ground = max(0.0, 1.0 - abs(y_n) / 1.5)
    speed_sq = vx_n ** 2 + vy_n ** 2
    speed_cost = -4.0 * near_ground * speed_sq

    # --- 4. orient_cost: mild attitude regularizer ---
    orient_cost = -0.1 * abs(angle_n) - 0.02 * (angvel_n ** 2)

    # --- 5. contact_reward: mild supplement for leg contact ---
    contact_reward = 1.5 * (lc_n + rc_n)

    # --- 6. landing_bonus: small bonus for both legs on pad near origin ---
    near_origin = max(0.0, 1.0 - (x_n ** 2 + y_n ** 2) ** 0.5 / 0.4)
    landing_bonus = 3.0 * lc_n * rc_n * near_origin

    # --- 7. fuel_cost: symbolic only, don't discourage necessary engine use ---
    if action == 2:
        fuel_cost = -0.01
    elif action in (1, 3):
        fuel_cost = -0.005
    else:
        fuel_cost = 0.0

    # --- assemble ---
    total_reward = (
        progress
        + descent_reward
        + speed_cost
        + orient_cost
        + contact_reward
        + landing_bonus
        + fuel_cost
    )

    components = {
        'progress': progress,
        'descent_reward': descent_reward,
        'speed_cost': speed_cost,
        'orient_cost': orient_cost,
        'contact_reward': contact_reward,
        'landing_bonus': landing_bonus,
        'fuel_cost': fuel_cost,
    }

    return total_reward, components
```