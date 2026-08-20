# 上一轮奖励函数代码（该轮得分: 66.522094）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：进度奖励（保持上轮的 scale=10，不做改动）
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = (dist - next_dist) * 10.0

    # 2. 稳定性惩罚（保持不变）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy：从"每步状态奖励"改为"一次性触地事件奖励"
    #    诊断：上轮 nonzero_rate=44%，agent 着陆后静坐收割 0.5/步，ratio=4.81 主导总奖励。
    #    修复：通过比较 obs vs next_obs 的腿接触状态，检测「着陆瞬间」——
    #    双足从未接触变为接触的那一步才给奖励，消除静坐 exploit。
    prev_both_contact = (obs[6] > 0.5 and obs[7] > 0.5)
    curr_both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    just_landed = curr_both_contact and not prev_both_contact

    # 着陆质量条件（阈值适当放宽，因为是事件触发，不用担心每步 exploit）
    near_target = (next_dist < 0.3)
    low_speed = (abs(next_obs[2]) + abs(next_obs[3]) < 0.5)
    stable_angle = (abs(next_obs[4]) < 0.2)

    soft_landing_proxy = 1.0 if (just_landed and near_target and low_speed and stable_angle) else 0.0

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，因为 obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = dist - next_dist  # 每一步靠近目标获得正奖励

    # 2. 稳定性惩罚（系数降低 10 倍：0.01 → 0.001）
    #    原因：上轮 ratio_to_progress = -0.88，惩罚几乎抵消全部进度信号，
    #    导致 agent 无有效梯度可学。削弱后预计 ratio ≈ -0.09，progress 将主导。
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy（本轮未改，留待下轮观察）
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
score=66.522094, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.045367 | 0.048510 | 0.999966 | 1.000000 |
| soft_landing_proxy | 0.077214 | 0.077214 | 0.077214 | 1.701983 |
| stability_penalty | -0.000628 | 0.000628 | 1.000000 | -0.013850 |
| total_reward | 0.121953 | 0.125075 | 1.000000 | 2.688133 |
| generated_reward | 0.121953 | 0.125075 | 1.000000 | 2.688133 |
| original_env_reward | -0.250546 | 4.386100 | 1.000000 | -5.522627 |

## Distribution
- score: mean=66.522094, min=40.938003, max=95.685433
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_proxy + stability_penalty | -112.37 | -112.37 | 0.00 | 74.10 | progress_reward=0.016 soft_landing_proxy=0.001 stability_penalty=-0.014 | new_best |
| 2 | progress_reward + soft_landing_proxy + stability_penalty | 141.58 | 141.58 | 0.00 | 1000.00 | progress_reward=0.004 soft_landing_proxy=0.214 stability_penalty=-0.000 | new_best |
| 3 | progress_reward + soft_landing_proxy + stability_penalty | 142.68 | 141.58 | 1.10 | 1000.00 | progress_reward=0.046 soft_landing_proxy=0.221 stability_penalty=-0.001 | no_meaningful_improvement |
| 4 | progress_reward + soft_landing_proxy + stability_penalty | 66.52 | 141.58 | -75.06 | 1000.00 | progress_reward=0.045 soft_landing_proxy=0.077 stability_penalty=-0.001 | no_meaningful_improvement |
