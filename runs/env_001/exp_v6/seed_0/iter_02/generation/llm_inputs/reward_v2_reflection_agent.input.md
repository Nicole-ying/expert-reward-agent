# 上一轮奖励函数代码（该轮得分: -112.368158）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，因为 obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = dist - next_dist  # 每一步靠近目标获得正奖励

    # 2. 稳定性惩罚（基于速度、姿态和角速度）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.01 * (vel_x + vel_y) - 0.01 * angle - 0.01 * ang_vel

    # 3. 软着陆 proxy：在满足靠近、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = (next_dist < 0.1)
    low_speed = (vel_x + vel_y < 0.2)
    stable_angle = (angle < 0.1)
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_proxy = 0.5 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-112.368158, len=74.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.016123 | 0.017044 | 1.000000 | 1.000000 |
| soft_landing_proxy | 0.000986 | 0.000986 | 0.001971 | 0.061126 |
| stability_penalty | -0.014204 | 0.014204 | 1.000000 | -0.880965 |
| total_reward | 0.002905 | 0.008829 | 1.000000 | 0.180160 |
| generated_reward | 0.002905 | 0.008829 | 1.000000 | 0.180160 |
| original_env_reward | -1.588476 | 2.430089 | 1.000000 | -98.521943 |

## Distribution
- score: mean=-112.368158, min=-118.802286, max=-101.820135
- episode_length: mean=74.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_proxy + stability_penalty | -112.37 | -112.37 | 0.00 | 74.10 | progress_reward=0.016 soft_landing_proxy=0.001 stability_penalty=-0.014 | new_best |
