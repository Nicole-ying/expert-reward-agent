# 上一轮奖励函数代码（该轮得分: -2.337403）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步距离和下一步距离
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 主信号1：progress_delta——替代 distance_reward
    # 靠近原点 → 正奖励，远离 → 负惩罚，提供密集的方向性梯度
    # 系数 5.0：远距离时主导（d=2→1.5，delta=0.5，reward=2.5）
    progress_delta_reward = 5.0 * (old_dist - new_dist)

    # 主信号2：proximity_bonus——保持不变
    # 近距离时提供强吸引力（d=0 时=2.0），远距离时弱但始终为正
    proximity_bonus = 2.0 / (1.0 + 5.0 * new_dist)

    # 稳定约束——保持上一轮系数不变
    vel_penalty = 0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.2 * abs(next_obs[4])
    angvel_penalty = 0.05 * abs(next_obs[5])
    stability_penalty = -vel_penalty - angle_penalty - angvel_penalty

    total_reward = progress_delta_reward + proximity_bonus + stability_penalty

    components = {
        'progress_delta_reward': progress_delta_reward,
        'proximity_bonus': proximity_bonus,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 到原点 (0,0) 的距离
    dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 主信号1：负距离——提供指向原点的密集梯度
    distance_reward = -1.0 * dist

    # 主信号2：连续接近奖励——替代稀疏的 landing_bonus
    # 数学形态：bounded 在 [0, 2]，dist=0 时=2.0，dist=1 时≈0.33
    # 每一步都有正向信号，越近越强，解决 nonzero_rate=0.19% 的问题
    proximity_bonus = 2.0 / (1.0 + 5.0 * dist)

    # 稳定约束——保持上一轮系数不变，单独归因本轮改动
    vel_penalty = 0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.2 * abs(next_obs[4])
    angvel_penalty = 0.05 * abs(next_obs[5])
    stability_penalty = -vel_penalty - angle_penalty - angvel_penalty

    total_reward = distance_reward + proximity_bonus + stability_penalty

    components = {
        'distance_reward': distance_reward,
        'proximity_bonus': proximity_bonus,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-2.337403, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.009756 | 0.014119 | 0.999996 | 1.000000 |
| proximity_bonus | 1.077045 | 1.077045 | 1.000000 | 110.396211 |
| stability_penalty | -0.055291 | 0.055291 | 1.000000 | -5.667329 |
| total_reward | 1.031509 | 1.031700 | 1.000000 | 105.728882 |
| generated_reward | 1.031509 | 1.031700 | 1.000000 | 105.728882 |
| original_env_reward | -0.083474 | 2.886906 | 1.000000 | -8.556000 |

## Distribution
- score: mean=-2.337403, min=-34.641733, max=27.386870
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
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
| 8 | distance_reward + proximity_bonus + stability_penalty | 94.80 | 94.80 | 0.00 | 795.90 | distance_reward=-0.552 proximity_bonus=0.843 stability_penalty=-0.087 | new_best |
| 9 | progress_delta_reward + proximity_bonus + stability_penalty | -2.34 | 94.80 | -97.14 | 1000.00 | progress_delta_reward=0.010 proximity_bonus=1.077 stability_penalty=-0.055 | no_meaningful_improvement |
