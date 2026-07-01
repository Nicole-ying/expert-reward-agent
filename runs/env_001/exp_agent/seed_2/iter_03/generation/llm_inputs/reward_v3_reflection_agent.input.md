# 当前奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断 1：stability_penalty 的 ratio_to_progress = -0.80，严重主导信号
    # 修复：整体系数降低 10 倍，从 0.5/0.2/0.1 降到 0.05/0.02/0.01
    # 目标：让 penalty 均值降到 progress 均值的 10% 以下
    # ============================================================
    
    # 主学习信号：progress_delta_reward（保持不变）
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta

    # 稳定约束：stability_penalty（系数降低 10 倍）
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_obs[4])      # 原 0.5 → 0.05
    angular_vel_penalty = 0.02 * abs(next_obs[5]) # 原 0.2 → 0.02
    speed_penalty = 0.01 * speed                  # 原 0.1 → 0.01
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # ============================================================
    # 诊断 2：landing_bonus 的 nonzero_rate = 0.6%，几乎从不触发
    # 修复：二值 if 条件 → 连续乘积形式
    # 每个因子用 max(0, 1 - x/threshold)，值域 [0,1]
    # 这样 agent 每靠近一步都能感受到梯度
    # ============================================================
    
    # 连续因子：距离因子（距离 < 0.5 时为正，越近越大）
    dist_factor = max(0.0, 1.0 - next_dist / 0.5)
    # 速度因子（速度 < 0.5 时为正，越慢越大）
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    # 姿态角因子（角度 < 0.3 时为正，越小越大）
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    # 接触因子：两个接触传感器都 > 0.5 时为 1，否则为 0（这个保持二值，因为接触是离散事件）
    contact_factor = 1.0 if (next_obs[6] > 0.5) and (next_obs[7] > 0.5) else 0.0
    
    # 连续乘积：每个因子都在 [0,1]，乘积也在 [0,1]
    # 系数从 2.0 提高到 5.0，因为连续乘积天然小于 1，需要补偿
    landing_bonus = 5.0 * dist_factor * speed_factor * angle_factor * contact_factor

    # 动作代价：energy_penalty（保持不变，ratio 很小没问题）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 计算当前位置到目标（0,0）的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta  # 正奖励表示更接近目标

    # 稳定约束：stability_penalty
    # 惩罚高速、大姿态角和大角速度
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_obs[4])  # 姿态角惩罚
    angular_vel_penalty = 0.2 * abs(next_obs[5])  # 角速度惩罚
    speed_penalty = 0.1 * speed  # 速度惩罚
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 任务完成 proxy：soft_landing_proxy
    # 当接近目标、低速、小姿态角且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_obs[4]) < 0.3
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 动作代价：energy_penalty（小权重）
    # 惩罚使用引擎的动作（action 1,2,3）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    # 构建组件字典
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈
# Training Feedback

## Training outcome
score=-110.337746, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.005890 | 0.005890 | 0.117793 | -0.036612 |
| landing_bonus | 0.017613 | 0.017613 | 0.006125 | 0.109489 |
| progress_reward | 0.160869 | 0.170178 | 0.999995 | 1.000000 |
| stability_penalty | -0.014270 | 0.014270 | 1.000000 | -0.088703 |
| total_reward | 0.158323 | 0.173897 | 1.000000 | 0.984174 |
| generated_reward | 0.158323 | 0.173897 | 1.000000 | 0.984174 |
| original_env_reward | -1.613310 | 2.329773 | 1.000000 | -10.028747 |

## Distribution
- score: mean=-110.337746, min=-124.079149, max=-95.059093
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -108.81 | -108.81 | 0.00 | 72.00 | energy_penalty=-0.007 landing_bonus=0.013 progress_reward=0.161 stability_penalty=-0.129 | new_best |
| 2 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -110.34 | -108.81 | -1.52 | 72.00 | energy_penalty=-0.006 landing_bonus=0.018 progress_reward=0.161 stability_penalty=-0.014 | no_meaningful_improvement |
