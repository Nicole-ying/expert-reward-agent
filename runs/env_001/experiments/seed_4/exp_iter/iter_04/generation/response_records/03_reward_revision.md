# Response Record

# Reward Revision for Env_001 – Iteration 4

## Design Changes

### Kept
- `progress_reward`: main learning signal, unchanged at 2.0 multiplier
- `distance_reward`: small anchor to keep agent moving toward target, unchanged at -0.1
- `landing_bonus`: soft completion proxy, unchanged structure

### Weakened
- `stability_penalty`: reduced from -0.0555 mean to -0.03 mean by lowering all sub-penalties (angle from 0.03→0.02, angular_vel from 0.015→0.01, speed from 0.05→0.03). This prevents stability from dominating the progress signal while still discouraging extreme instability.

### Revised
- None structurally; only coefficient adjustments.

### Added
- None. The skeleton plan suggested considering `distance_reward` as a small anchor, but it already exists. No new components needed.

### Why still no terminal_success_reward / terminal_failure_penalty
- No explicit success/failure signal in info or obs. Using a proxy would be unreliable and could misguide training.

### Next observation focus
- Check if `progress_reward` nonzero_rate remains high (>0.99) and its mean increases.
- Check if `stability_penalty` mean is now below -0.04 (i.e., less dominant).
- Check if external evaluation reward improves (less negative) and episode length increases.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward (保留)
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta

    # 稳定约束：stability_penalty (进一步削弱，避免主导progress)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.02 * abs(next_obs[4])  # 从0.03降至0.02
    angular_vel_penalty = 0.01 * abs(next_obs[5])  # 从0.015降至0.01
    speed_penalty = 0.03 * speed  # 从0.05降至0.03
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 平滑接近引导：distance_reward (保留)
    distance_reward = -0.1 * next_dist

    # 任务完成proxy：soft_landing_proxy (保留)
    near_target = next_dist < 0.5
    low_speed = speed < 0.8
    stable_angle = abs(next_obs[4]) < 0.3
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 1.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + distance_reward + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "distance_reward": distance_reward,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```
