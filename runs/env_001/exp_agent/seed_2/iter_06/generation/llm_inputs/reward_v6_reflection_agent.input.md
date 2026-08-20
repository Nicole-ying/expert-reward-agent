# 当前奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 使用 obs[0], obs[1] 和 next_obs[0], next_obs[1] 计算到目标(0,0)的距离变化
    # 目标位置是 (0, 0)，因为 obs 已经是相对于目标平台的坐标
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist  # 正数表示更接近目标
    progress_delta_reward = 10.0 * progress_delta  # 权重10，鼓励每一步都更接近

    # ========== 稳定/安全约束：stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度，鼓励稳定接近和着陆
    # 使用 next_obs 的状态，因为这是动作执行后的结果
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])  # 姿态角偏离0的绝对值
    angular_vel_penalty = abs(next_obs[5])  # 角速度绝对值
    
    # 权重：速度惩罚0.5，姿态角惩罚0.3，角速度惩罚0.2
    stability_penalty = -0.5 * speed - 0.3 * angle_penalty - 0.2 * angular_vel_penalty

    # ========== 任务完成 proxy：soft_landing_proxy ==========
    # 当飞行器接近目标、速度低、姿态稳定且两个支撑接触时给予小奖励
    # 这是对成功着陆的软近似，不是真正的 success flag
    near_target = (next_dist < 0.5)  # 距离目标小于0.5
    low_speed = (speed < 0.3)  # 速度小于0.3
    stable_angle = (abs(next_obs[4]) < 0.2)  # 姿态角小于0.2弧度
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)  # 两个支撑都接触
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0  # 小权重奖励，避免过度激励

    # ========== 动作代价：energy_penalty（小权重） ==========
    # 惩罚使用引擎，鼓励节能
    # action 0: no_engine, action 1: left_orientation, action 2: main, action 3: right_orientation
    engine_use = 0.0
    if action == 1 or action == 3:
        engine_use = 0.5  # 姿态发动机
    elif action == 2:
        engine_use = 1.0  # 主发动机
    energy_penalty = -0.1 * engine_use  # 权重很小，避免agent不敢动

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
score=-111.677964, len=71.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.007854 | 0.007854 | 0.143210 | -0.048738 |
| progress_delta_reward | 0.161144 | 0.170426 | 0.999995 | 1.000000 |
| soft_landing_bonus | 0.010838 | 0.010838 | 0.005419 | 0.067255 |
| stability_penalty | -0.555900 | 0.555900 | 1.000000 | -3.449699 |
| total_reward | -0.391771 | 0.411592 | 1.000000 | -2.431181 |
| generated_reward | -0.391771 | 0.411592 | 1.000000 | -2.431181 |
| original_env_reward | -1.624858 | 2.347336 | 1.000000 | -10.083237 |

## Distribution
- score: mean=-111.677964, min=-124.079149, max=-95.059093
- episode_length: mean=71.900000
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
