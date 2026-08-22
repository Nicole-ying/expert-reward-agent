## Analysis of existing reward components

### 1) `landing_reward`
- **Evidence**: `magnitude_share = 92.3%`, `active_rate = 44.7%`, episode sum mean = 794.9.
- **Interpretation**: This single component completely dominates the reward signal. Because it fires only after leg contact (active rate < 50 %), the agent receives almost no useful feedback during the long flight phase. The huge magnitude relative to all other terms means the RL objective is essentially “try to trigger the landing bonus”, but without a dense guiding signal the agent cannot discover how to reach the pad.
- **Verdict**: Must be drastically scaled down (reduce coefficient by at least a factor of 5–10) so that other components can shape behaviour.

### 2) `dist_penalty`
- **Evidence**: `magnitude_share = 4.2%`, active 100 %, flat trajectory (checkpoints at 40 % and 80 % both gave score 0).
- **Interpretation**: A constant negative penalty proportional to distance provides no positive gradient toward the target—it only makes every step mildly painful. The penalty scale (-0.2) is too weak to overcome the overwhelming `landing_reward` and does not reward *progress*; it rewards staying close without any incentive to get closer. This explains why learning completely stalls.
- **Verdict**: Should be replaced by a **progress-based reward** (rewarding reduction in distance) that gives a clear, immediate signal for moving toward the pad.

### 3) `vel_penalty` and `ang_penalty`
- **Evidence**: Both active 100 %, very small magnitude shares (1.7 %, 1.1 %). No sign of learning.
- **Interpretation**: Penalising velocity and angle at every step is counterproductive during the approach phase (the agent *needs* to move fast and may tilt). The current scale is too small to matter anyway. A better approach is to make these penalties conditional on being near/on the landing pad, so they promote a soft landing rather than hindering the whole trajectory.
- **Verdict**: Keep them but reduce their coefficients and, if possible, gate them by proximity to the pad. For simplicity, I will lower the coefficients substantially so they only prevent extreme values without slowing down the approach.

### 4) `fuel_penalty`
- **Evidence**: Tiny magnitude share (0.7 %), 83.6 % active.
- **Interpretation**: The penalty is reasonable but entirely invisible because of the dominating landing reward. Once the rest of the function is balanced, it will serve its purpose as a soft incentive for fuel efficiency.
- **Verdict**: Keep unchanged.

### Overall assessment
The reward function fails fundamentally: there is **no reward for making progress toward the target**, the only positive component fires late and dominates everything, and all other terms are either static penalties or negligible. The checkpoints at 40 % and 80 % both yield a score of 0.0, confirming that no learning occurs during training. According to Rule (1) (“score near zero or negative → rewrite the whole reward with a different framework”), a complete rewrite is necessary. I will introduce a progress reward as the main driver, drastically shrink the landing bonus, and lower the velocity/angle penalties.

---

## Revised reward function

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Progress reward: reward moving closer to the target ---
    old_dist = abs(obs[0]) + abs(obs[1])
    new_dist = abs(next_obs[0]) + abs(next_obs[1])
    progress = old_dist - new_dist  # positive when distance shrinks
    progress_reward = 0.5 * progress

    # --- Landing / contact reward (greatly reduced scale) ---
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    contact_factor = (left_contact + right_contact) / 2.0

    vel_abs = abs(next_obs[2]) + abs(next_obs[3])
    angle_abs = abs(next_obs[4])
    # stability is high when velocity and angle are small
    stability = 1.0 / (1.0 + vel_abs + 5.0 * angle_abs)
    landing_reward = 1.0 * contact_factor * stability   # old was 5.0

    # --- Light penalties for orientation and speed (conditioned softly) ---
    # Keep small throughout so they don't hinder approach but suppress extremes.
    ang_penalty = -0.05 * angle_abs       # old was -0.5 * |angle|
    vel_penalty = -0.01 * vel_abs         # old was -0.1 * (|x_vel|+|y_vel|)

    # --- Fuel efficiency ---
    fuel_penalty = -0.02 * (1.0 if action != 0 else 0.0)

    total_reward = progress_reward + landing_reward + ang_penalty + vel_penalty + fuel_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "ang_penalty": ang_penalty,
        "vel_penalty": vel_penalty,
        "fuel_penalty": fuel_penalty,
    }
    return float(total_reward), components
```