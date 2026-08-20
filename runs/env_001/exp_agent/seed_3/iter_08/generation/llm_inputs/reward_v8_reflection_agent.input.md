# 上一轮奖励函数代码（该轮得分: -112.636699）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主信号：到原点的负距离（目标位置为 (0,0)）
    dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    distance_reward = -1.0 * dist

    # 稳定约束：惩罚速度、倾斜角和角速度的绝对值
    vel_penalty = 0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.2 * abs(next_obs[4])
    angvel_penalty = 0.05 * abs(next_obs[5])
    stability_penalty = -vel_penalty - angle_penalty - angvel_penalty

    # 软着陆代理：当满足安全着陆条件时给予一次性正奖励
    landing_bonus = 0.0
    if (dist < 0.1 and 
        abs(next_obs[2]) < 0.2 and abs(next_obs[3]) < 0.2 and 
        abs(next_obs[4]) < 0.05 and abs(next_obs[5]) < 0.1 and 
        next_obs[6] == 1.0 and next_obs[7] == 1.0):
        landing_bonus = 5.0  # 一次性的正奖励

    total_reward = distance_reward + stability_penalty + landing_bonus

    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'landing_bonus': landing_bonus,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-112.636699, len=70.500000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| distance_reward | -0.972025 | 0.972025 | 1.000000 | -0.972025 |
| landing_bonus | 0.009277 | 0.009277 | 0.001855 | 0.009277 |
| stability_penalty | -0.144418 | 0.144418 | 1.000000 | -0.144418 |
| total_reward | -1.107166 | 1.125475 | 1.000000 | -1.107166 |
| generated_reward | -1.107166 | 1.125475 | 1.000000 | -1.107166 |
| original_env_reward | -1.632276 | 2.447112 | 1.000000 | -1.632276 |

## Distribution
- score: mean=-112.636699, min=-125.170388, max=-95.059093
- episode_length: mean=70.500000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + soft_landing_proxy + stability_penalty | -109.27 | -109.27 | 0.00 | 70.60 | progress_delta=0.016 soft_landing_proxy=0.001 stability_penalty=-0.147 | new_best |
| 2 | progress_delta + soft_landing_proxy + stability_penalty | 148.74 | 148.74 | 0.00 | 745.50 | progress_delta=0.011 soft_landing_proxy=0.017 stability_penalty=-0.012 | new_best |
| 3 | progress_delta + soft_landing_proxy + stability_penalty | 197.22 | 197.22 | 0.00 | 796.30 | progress_delta=0.004 soft_landing_proxy=0.192 stability_penalty=-0.006 | new_best |
| 4 | progress_delta + soft_landing_proxy + stability_penalty | -111.55 | 197.22 | -308.77 | 70.60 | progress_delta=0.016 soft_landing_proxy=0.001 stability_penalty=-0.011 | no_meaningful_improvement |
| 5 | progress_delta + soft_landing_proxy + stability_penalty | -110.74 | 197.22 | -307.96 | 70.60 | progress_delta=0.016 soft_landing_proxy=0.001 stability_penalty=-0.011 | no_meaningful_improvement |
| 6 | progress_delta + soft_landing_proxy + stability_penalty | 142.40 | 197.22 | -54.81 | 1000.00 | progress_delta=0.002 soft_landing_proxy=0.330 stability_penalty=-0.004 | unsolved_stagnation_fresh_restart |
| 7 | distance_reward + landing_bonus + stability_penalty | -112.64 | -112.64 | 0.00 | 70.50 | distance_reward=-0.972 landing_bonus=0.009 stability_penalty=-0.144 | new_best |
