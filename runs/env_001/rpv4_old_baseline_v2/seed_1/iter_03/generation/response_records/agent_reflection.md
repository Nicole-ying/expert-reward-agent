# Response Record

## Analysis of Current Reward Components

### 1. `vel_penalty` = -0.2*(vx²+vy²)
- **magnitude_share**: 44.6% — completely dominates the reward signal.
- **episode_sum_mean**: -5.96 — a large negative every episode.
- **effect**: The agent is heavily penalised for any speed. It quickly learns that the safest move is to stay still, which prevents it from approaching the target and touching the landing pad. This single component makes progress impossible.
- **Decision**: Drastically reduce its weight so that it can no longer drown out the approach signal.

### 2. `progress` = 2.0*(prev_dist - curr_dist)
- **magnitude_share**: 21.0%, positive 11%. It is the only component encouraging approach to the centre.
- **Problem**: Being a difference, its magnitude is small and noisy. It fails to overcome the large velocity penalty, so the agent never learns to reliably reduce distance.
- **Decision**: Because the score is strongly negative and the agent never learns to move, the main signal must be replaced by a robust, level‑based distance penalty that does not depend on step‑to‑step differences.

### 3. `angle_penalty` = -0.5*(angle²)
- **magnitude_share**: 17.3%, mean -2.30. The penalty is reasonable, but currently it cannot offset the velocity penalty. The upright attitude objective is still useful; keep but possibly scale.

### 4. `contact_reward` = 1.0*left + 1.0*right + 3.0*left*right
- **active_rate**: 0.1% — almost never triggered, because the agent rarely reaches the pad.
- **episode_sum_mean**: +2.05 — when it does occur, it gives a small positive.
- **Decision**: The magnitude must be increased drastically to make landing attractive once the agent starts getting close. A larger contact reward also creates a strong final‑approach incentive.

### 5. `angvel_penalty` = -0.1*(ang_vel²)
- **magnitude_share**: 1.8%. Negligible and has no harmful dominance. Keep unchanged.

**Overall:** The score is far below zero, and during training checkpoints it was flat 0.00 (indicating the reward did not guide the agent at all). This triggers rule (1): the whole reward must be rewritten around a different core signal. I will replace `progress` with a direct distance penalty (`-w_dist * current_distance`) to create a continuous, dense gradient toward the target, and severely reduce `vel_penalty` so that the agent is willing to move.

## Revised Reward Function
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state
    px, py, vx, vy, angle, ang_vel = next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4], next_obs[5]
    left_contact, right_contact = next_obs[6], next_obs[7]

    # ---- New core signal: distance penalty ----
    # Agent is heavily pushed to minimise its distance to the centre (0,0)
    curr_dist = (px ** 2 + py ** 2) ** 0.5
    distance_penalty = -0.5 * curr_dist

    # ---- Attitude (keep upright) ----
    angle_penalty = -0.5 * (angle ** 2)

    # ---- Rotation smoothness ----
    angvel_penalty = -0.1 * (ang_vel ** 2)

    # ---- Linear speed: now a gentle penalty (was 0.2, dwarfed everything) ----
    vel_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # ---- Landing contact: much stronger incentive ----
    contact_reward = 10.0 * left_contact + 10.0 * right_contact + 30.0 * left_contact * right_contact

    total = distance_penalty + angle_penalty + angvel_penalty + vel_penalty + contact_reward
    components = {
        "distance_penalty": distance_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "vel_penalty": vel_penalty,
        "contact_reward": contact_reward
    }
    return float(total), components
```
