# 上一轮奖励函数代码（该轮得分: -108.748188）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Parameters
    gamma = 0.995
    w_v = 0.02  # velocity penalty weight
    w_a = 0.1   # angle penalty weight
    w_w = 0.05  # angular velocity penalty weight
    landing_bonus = 2.0
    # Thresholds for soft landing proxy
    dist_thresh = 0.3
    speed_thresh_x = 0.1
    speed_thresh_y = 0.15
    angle_thresh = 0.15

    # 1. Potential based shaping:
    #    Phi(s) = - distance_to_target
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    phi_obs = -dist_obs
    phi_next = -dist_next
    potential_shaping = gamma * phi_next - phi_obs  # = -gamma*dist_next + dist_obs

    # 2. Stability penalty (encourages low speed, small angle, low angular velocity)
    stability_penalty = - (
        w_v * (abs(next_obs[2]) + abs(next_obs[3])) +
        w_a * abs(next_obs[4]) +
        w_w * abs(next_obs[5])
    )

    # 3. Soft landing proxy (indicates successful touchdown without explicit success flag)
    soft_landing_proxy = 0.0
    if (dist_next < dist_thresh
            and abs(next_obs[2]) < speed_thresh_x
            and abs(next_obs[3]) < speed_thresh_y
            and abs(next_obs[4]) < angle_thresh
            and next_obs[6] == 1.0
            and next_obs[7] == 1.0):
        soft_landing_proxy = landing_bonus

    total_reward = potential_shaping + stability_penalty + soft_landing_proxy

    components = {
        "potential_shaping": potential_shaping,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-108.748188, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| potential_shaping | 0.020808 | 0.021185 | 0.999999 | 0.020808 |
| soft_landing_proxy | 0.009501 | 0.009501 | 0.004750 | 0.009501 |
| stability_penalty | -0.031485 | 0.031485 | 1.000000 | -0.031485 |
| total_reward | -0.001176 | 0.021585 | 1.000000 | -0.001176 |
| generated_reward | -0.001176 | 0.021585 | 1.000000 | -0.001176 |
| original_env_reward | -1.529567 | 2.385718 | 1.000000 | -1.529567 |

## Distribution
- score: mean=-108.748188, min=-120.796346, max=-95.059093
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -108.61 | -108.61 | 0.00 | 72.00 | distance_penalty=-0.049 progress_delta_reward=0.016 soft_landing_bonus=0.001 stability_penalty=-0.082 | new_best |
| 2 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -98.04 | -98.04 | 0.00 | 72.30 | distance_penalty=-0.005 progress_delta_reward=0.016 soft_landing_bonus=0.009 stability_penalty=-0.009 | new_best |
| 3 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -12.26 | -12.26 | 0.00 | 1000.00 | approach_bonus=2.711 distance_penalty=-0.003 progress_delta_reward=0.002 stability_penalty=-0.005 | new_best |
| 4 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -60.67 | -12.26 | -48.41 | 85.30 | approach_bonus=0.005 distance_penalty=-0.004 progress_delta_reward=0.013 stability_penalty=-0.007 | no_meaningful_improvement |
| 5 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -110.31 | -12.26 | -98.05 | 72.00 | approach_bonus=0.009 distance_penalty=-0.005 progress_delta_reward=0.016 stability_penalty=-0.008 | no_meaningful_improvement |
| 6 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -55.92 | -12.26 | -43.65 | 955.10 | approach_bonus=3.129 distance_penalty=-0.002 progress_delta_reward=0.002 stability_penalty=-0.007 | unsolved_stagnation_fresh_restart |
| 7 | potential_shaping + soft_landing_proxy + stability_penalty | -108.75 | -108.75 | 0.00 | 72.00 | potential_shaping=0.021 soft_landing_proxy=0.010 stability_penalty=-0.031 | new_best |
