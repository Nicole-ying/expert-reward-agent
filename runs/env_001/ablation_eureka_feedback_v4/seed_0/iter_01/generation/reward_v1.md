# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs
    x = next_obs[0]            # horizontal position relative to landing pad
    y = next_obs[1]            # vertical position relative to pad
    vx = next_obs[2]           # horizontal velocity
    vy = next_obs[3]           # vertical velocity
    angle = next_obs[4]        # body orientation angle
    ang_vel = next_obs[5]      # angular velocity
    left_contact = next_obs[6] # left leg contact flag (0/1)
    right_contact = next_obs[7]# right leg contact flag (0/1)

    # --- Component 1: position proximity (main learning signal) ---
    distance = (x**2 + y**2) ** 0.5
    proximity = -1.0 * distance   # negative Euclidean distance, encouraging approach

    # --- Component 2: movement stability penalty (soft landing assistance) ---
    # Quadratic penalties on linear velocity, angular velocity, and body angle.
    vel_penalty = -0.05 * (vx**2 + vy**2)
    ang_penalty = -0.02 * (ang_vel**2)
    angle_penalty = -0.5 * (angle**2)
    stability = vel_penalty + ang_penalty + angle_penalty

    # --- Component 3: leg contact reward (encouraging safe settlement) ---
    contact = 0.5 * (left_contact + right_contact)

    # Total reward
    total = proximity + stability + contact

    components = {
        "proximity": proximity,
        "stability": stability,
        "contact": contact
    }
    return float(total), components
```

# reward_v1 设计说明

## 1. Task Profile & Dynamics Subtype
- **Task family**: `navigation_goal_reaching`
- **Dynamics subtype**: `goal_approach_and_soft_contact`
- **Core objectives**: reach the center landing pad (relatively positioned at (0,0)) and settle stably with low velocity, upright orientation, and both legs in contact with the pad.

## 2. Selected Reward Roles
From `reward_role_decomposition`, the following roles were selected for v1:
- **position_approach** (mandatory): drives the lander toward the target pad.
- **soft_landing** (mandatory): encourages low speeds, low angular velocity, small orientation error, and leg contact.
- **fuel_efficiency** and **time_penalty** are excluded in v1 to avoid hindering early exploration and task learning.

## 3. Role‑to‑Signal Mapping and Formula Operators
| Role              | Usable signals                                        | Chosen formula operator(s)            | Rationale |
|-------------------|-------------------------------------------------------|---------------------------------------|-----------|
| position_approach | `next_obs[0]`, `next_obs[1]` (relative position)      | `dense_state_signal` (penalty, convex via sqrt distance) | Negative Euclidean distance provides per‑step gradients encouraging movement toward the target. |
| soft_landing      | `next_obs[2]`, `next_obs[3]`, `next_obs[5]`, `next_obs[4]`, `next_obs[6]`, `next_obs[7]` | `quadratic_penalty` (velocities, angular vel, angle) + `dense_state_signal` (positive linear) for contacts | Quadratic penalties smoothly penalise high speeds and misalignment; additive contact reward incentivises leg touchdown. No heavy gating is used to keep v1 simple. |

- **proximity** → `dense_state_signal` penalty: `-w * distance` (distance is `(x²+y²)^0.5`).
- **stability** → three `quadratic_penalty` terms summed into one component:
  - `-0.05 * (vx² + vy²)` – linear velocity suppression.
  - `-0.02 * (ang_vel²)` – angular velocity suppression.
  - `-0.5 * (angle²)` – body angle suppression.
- **contact** → `dense_state_signal` positive linear: `0.5 * (left_contact + right_contact)`, giving up to +1.0 when both legs touch the pad.

## 4. Excluded Roles and Why
- `explicit_success_bonus` – excluded because `explicit_success_flag_available=false` (info is empty). No reliable success flag exists; constructing a proxy could produce false positives.
- `fuel_efficiency` / `time_penalty` – left for future iterations. v1 focuses purely on learning the approach‑and‑land behaviour without penalising exploration or causing early collisions just to minimise steps.

## 5. Why No terminal_success_reward / terminal_failure_penalty
Both are absent because the environment provides no explicit success or failure flags (info is empty). Using inferred termination conditions would be unreliable and violate the signal‑availability constraint.

## 6. Responsibilities Deferred to Later Iterations
- Fuel/energy efficiency (action penalty).
- Time‑optimality (per‑step constant penalty).
- Curriculum weighting or soft health gates – may be introduced after analysing early training failure modes.
- More refined joint‑condition proxy for landing quality (e.g., product of proximity, speed, and contact factors) could be added once the agent reliably achieves basic landing.

## 7. Post‑Training Failure Modes to Watch
- **high‑velocity impact**: agent rushes toward target and crashes – proximity dominates too strongly; consider decreasing `proximity` weight or increasing `stability` weights.
- **hovering / refusal to descend**: `stability` penalties suppress movement too much, causing the lander to stay high; lower `stability` weights or increase `proximity`.
- **one‑leg landing**: agent lands but only one leg contacts, failing to settle; `contact` reward already promotes bilateral contact, but may need a multiplier if it persists.
- **excessive tipping**: large body angles during descent; `angle_penalty` weight may need adjustment.
- **stand‑still near target**: very slow approach but not finishing; adding a small time penalty or adjusting contact threshold may be needed later.

All weights have been chosen based on typical scale estimates (coordinates ~[-1,1], velocities ~[-3,3], angle ~[-π/2,π/2]). They can be tuned after observing actual episode statistics.