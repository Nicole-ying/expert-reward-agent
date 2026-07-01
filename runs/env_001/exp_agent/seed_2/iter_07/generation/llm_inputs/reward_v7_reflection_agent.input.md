# 当前奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 计算到目标(0,0)的距离变化
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist  # 正数表示更接近目标
    progress_delta_reward = 10.0 * progress_delta  # 权重10，鼓励每一步都更接近

    # ========== 稳定/安全约束：stability_penalty（已大幅削弱 + 距离门控） ==========
    # 诊断：上一轮 stability_penalty ratio=-3.45，严重主导，agent 不敢动
    # 修复1：系数降低10倍（从0.5/0.3/0.2 → 0.05/0.03/0.02）
    # 修复2：加入距离门控，只在靠近目标（dist<2.0）时生效，远处不约束
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])  # 姿态角偏离0的绝对值
    angular_vel_penalty = abs(next_obs[5])  # 角速度绝对值
    
    # 距离门控因子：dist<2.0 时逐渐生效，dist>=2.0 时完全关闭
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    # 大幅降低系数（10倍），并应用距离门控
    stability_penalty = -distance_gate * (0.05 * speed + 0.03 * angle_penalty + 0.02 * angular_vel_penalty)

    # ========== 任务完成 proxy：soft_landing_proxy（本轮不修改） ==========
    # 当飞行器接近目标、速度低、姿态稳定且两个支撑接触时给予小奖励
    near_target = (next_dist < 0.5)
    low_speed = (speed < 0.3)
    stable_angle = (abs(next_obs[4]) < 0.2)
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0

    # ========== 动作代价：energy_penalty（小权重，本轮不修改） ==========
    engine_use = 0.0
    if action == 1 or action == 3:
        engine_use = 0.5
    elif action == 2:
        engine_use = 1.0
    energy_penalty = -0.1 * engine_use

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
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
score=-116.949772, len=71.800000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.005432 | 0.005432 | 0.099739 | -0.033587 |
| progress_delta_reward | 0.161717 | 0.170881 | 0.999994 | 1.000000 |
| soft_landing_bonus | 0.010848 | 0.010848 | 0.005424 | 0.067079 |
| stability_penalty | -0.033476 | 0.033476 | 1.000000 | -0.207002 |
| total_reward | 0.133657 | 0.152343 | 1.000000 | 0.826490 |
| generated_reward | 0.133657 | 0.152343 | 1.000000 | 0.826490 |
| original_env_reward | -1.658587 | 2.353407 | 1.000000 | -10.256135 |

## Distribution
- score: mean=-116.949772, min=-143.017581, max=-103.429116
- episode_length: mean=71.800000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -108.81 | -108.81 | 0.00 | 72.00 | energy_penalty=-0.007 landing_bonus=0.013 progress_reward=0.161 stability_penalty=-0.129 | new_best |
| 2 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -110.34 | -108.81 | -1.52 | 72.00 | energy_penalty=-0.006 landing_bonus=0.018 progress_reward=0.161 stability_penalty=-0.014 | no_meaningful_improvement |
| 3 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -115.02 | -108.81 | -6.21 | 71.90 | energy_penalty=-0.006 landing_bonus=0.043 progress_reward=0.161 stability_penalty=-0.014 | no_meaningful_improvement |
| 4 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -227.71 | -108.81 | -118.89 | 697.20 | energy_penalty=-0.044 landing_bonus=2.333 progress_reward=0.009 stability_penalty=-0.007 | unsolved_stagnation_fresh_restart |
| 5 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -111.68 | -108.81 | -2.86 | 71.90 | energy_penalty=-0.008 progress_delta_reward=0.161 soft_landing_bonus=0.011 stability_penalty=-0.556 | no_meaningful_improvement |
| 6 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -116.95 | -108.81 | -8.14 | 71.80 | energy_penalty=-0.005 progress_delta_reward=0.162 soft_landing_bonus=0.011 stability_penalty=-0.033 | no_meaningful_improvement |
