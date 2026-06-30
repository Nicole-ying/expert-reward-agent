# Response Record

# Reward Revision Analysis

## Diagnosis from Evidence

- **progress_reward** mean = 0.031, nonzero_rate ≈ 100% → main driving signal is active but very small in magnitude
- **stability_penalty** mean = -0.003, abs_mean = 0.003 → negligible impact, not dominating
- **external_score** = 219.68, episode_length = 511 → agent is learning but slowly; progress signal may need amplification
- **landing_shaping_reward** and **approach_reward** are present but their contribution is unclear from evidence

## Revision Plan

### Keep
- `progress_delta_reward` — main driving signal, keep structure but increase base scale
- `stability_penalty` — very small, not harmful, keep as is
- `landing_shaping_reward` — keep, but reduce `both_contact` weight since contact flags are unreliable
- `approach_reward` — keep, but reduce weight to avoid competing with progress
- `distance_anchor` — keep as small anchor

### Revise
- **progress_reward scale**: increase from 10.0 to 15.0 to amplify the main learning signal
- **landing_shaping_reward**: reduce `both_contact` weight from 0.25 to 0.10 (contact flags are noisy)
- **approach_reward**: reduce from 2.0 to 1.0 to prevent it from overshadowing progress

### Weaken
- None needed; stability penalty is already negligible

### Consider Add
- None; current components are sufficient, just need better balance

### Still Defer
- terminal_success_reward, terminal_failure_penalty — no explicit success/failure signal
- energy_penalty, time_penalty — no evidence they are needed

## Next Observation
- Monitor progress_reward mean: target > 0.05 after revision
- Check if episode length decreases (faster landing) without sacrificing stability

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward (保留，增加基础缩放)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_scale = 1.0 + 0.5 * training_progress
    progress_reward = 15.0 * progress_scale * progress_delta  # 从10.0增加到15.0
    
    # 2. 稳定约束：stability_penalty (保留，权重不变)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.015 * abs(next_body_angle)
    angular_vel_penalty = -0.005 * abs(next_angular_vel)
    speed_penalty = -0.002 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. 着陆质量塑造：landing_shaping_reward (保留，降低contact权重)
    near_target = max(0.0, 1.0 - next_dist / 0.5)
    low_speed = max(0.0, 1.0 - speed / 0.3)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.2)
    both_contact = 1.0 if (next_left_contact > 0.5) and (next_right_contact > 0.5) else 0.0
    
    landing_shaping_reward = 0.5 * near_target + 0.25 * low_speed + 0.25 * stable_angle + 0.10 * both_contact  # contact从0.25降到0.10
    
    # 4. 接近奖励：approach_reward (保留，降低权重)
    approach_reward = 1.0 * max(0.0, 1.0 - next_dist / 1.0)  # 从2.0降到1.0
    
    # 5. 小权重距离锚点：distance_anchor (保留)
    distance_anchor = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_shaping_reward + approach_reward + distance_anchor
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "approach_reward": approach_reward,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
