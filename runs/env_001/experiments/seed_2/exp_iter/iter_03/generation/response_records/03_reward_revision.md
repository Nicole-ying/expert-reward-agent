# Response Record

# Reward Revision Analysis

## Key Evidence from Iteration Context

1. **External evaluation**: -101.26, episode length 72.2 → early failure/crash
2. **Component evidence**: progress_reward mean 0.161 (positive but weak), stability_penalty mean -0.117 (moderate penalty)
3. **Diagnosis**: reward does not guide stable control before termination
4. **Next action**: add smoother approach/landing guidance

## Revision Plan

### Keep
- `progress_delta_reward` (main learning signal, working but weak)
- `distance_reward` (small anchor, needed)

### Revise
- `stability_penalty`: reduce weight further to avoid discouraging exploration
- `landing_bonus`: make it smoother and more accessible (lower thresholds, continuous instead of binary)

### Consider Add
- Add a **smooth velocity shaping** component: reward reducing speed near target (continuous, not binary)
- Add a **contact stability** component: small reward for maintaining both contacts when near target

### Still Defer
- terminal_success_reward, terminal_failure_penalty (no explicit signals)
- energy_penalty (removed last round, no evidence to add back)
- time_penalty (no evidence needed)

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward (保留，权重不变)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定/安全约束：stability_penalty (削弱权重)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.1 * abs(next_body_angle)      # 从0.2降至0.1
    angular_vel_penalty = 0.05 * abs(next_ang_vel)  # 从0.1降至0.05
    speed_penalty = 0.05 * speed                     # 从0.1降至0.05
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 平滑接近引导：smooth_approach_shaping (新增)
    # 当接近目标时，奖励低速和稳定姿态（连续函数，非二值）
    near_target_factor = max(0.0, 1.0 - next_dist / 2.0)  # 距离越近，因子越大
    speed_comfort = max(0.0, 1.0 - speed / 1.0)           # 速度越低，奖励越高
    angle_comfort = max(0.0, 1.0 - abs(next_body_angle) / 0.5)  # 姿态越正，奖励越高
    approach_shaping = 2.0 * near_target_factor * (0.5 * speed_comfort + 0.5 * angle_comfort)
    
    # 4. 着陆代理：soft_landing_proxy (改为连续奖励，降低门槛)
    # 当接近目标、低速、姿态稳定且双接触时给予平滑奖励
    near_target = next_dist < 1.0       # 从0.8放宽至1.0
    low_speed = speed < 0.8             # 从0.5放宽至0.8
    stable_angle = abs(next_body_angle) < 0.5  # 从0.3放宽至0.5
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    landing_bonus = 0.5 if (near_target and low_speed and stable_angle and both_contact) else 0.0  # 从1.0降至0.5
    
    # 5. 距离锚点：distance_reward (保留)
    distance_reward = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + approach_shaping + landing_bonus + distance_reward
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "approach_shaping": approach_shaping,
        "landing_bonus": landing_bonus,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

## Design Explanation

### Compared to previous round:
- **Kept**: `progress_delta_reward` (main learning signal, weight unchanged), `distance_reward` (small anchor)
- **Weakened**: `stability_penalty` weights reduced by 50% (angle: 0.2→0.1, angular_vel: 0.1→0.05, speed: 0.1→0.05) to avoid discouraging exploration
- **Modified**: `landing_bonus` threshold relaxed (dist<1.0, speed<0.8, angle<0.5) and reward reduced (1.0→0.5) to make it more accessible but less dominant
- **Added**: `smooth_approach_shaping` - continuous reward that smoothly increases as agent gets closer to target with low speed and stable angle. This provides a gradient toward stable landing without requiring exact conditions.

### Why no terminal rewards:
- Still no explicit success/failure signals in info
- Terminal rewards would be sparse and not help with the early failure problem

### Next training focus:
- Monitor if `approach_shaping` increases episode length
- Check if `stability_penalty` reduction leads to more stable behavior or more crashes
- Verify that `progress_reward` remains the dominant signal (should be ~0.16 vs approach_shaping ~0.5-1.0)
