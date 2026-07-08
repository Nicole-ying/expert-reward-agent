# 上一轮奖励函数代码（该轮得分: -98.886267）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取下一步状态
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 权重和阈值（distance_cost 与 stability_cost 保持不变）
    w_dist = 1.0
    w_vel = 0.1
    w_angle = 0.1
    w_angvel = 0.05

    # 组件1: 距离成本（保持不变）
    distance_cost = -w_dist * (nx ** 2 + ny ** 2)

    # 组件2: 稳定性成本（保持不变）
    stability_cost = -(
        w_vel * (vx ** 2 + vy ** 2)
        + w_angle * abs(angle)
        + w_angvel * abs(omega)
    )

    # 组件3: 连续渐进着陆奖励（替代原稀疏 soft_landing_bonus）
    dist = (nx ** 2 + ny ** 2) ** 0.5
    vel_sq = vx ** 2 + vy ** 2

    proximity = 4.0 * (2.718281828 ** (-dist ** 2 / 0.5))
    velocity_ok = 4.0 * (2.718281828 ** (-vel_sq / 0.08))
    angle_ok = 4.0 * (2.718281828 ** (-angle ** 2 / 0.02))
    contact = 4.0 * (left_contact + right_contact) / 2.0

    approach_landing_reward = proximity + velocity_ok + angle_ok + contact

    # 合成总奖励
    total_reward = distance_cost + stability_cost + approach_landing_reward

    # 组件字典
    components = {
        'distance_cost': distance_cost,
        'stability_cost': stability_cost,
        'approach_landing_reward': approach_landing_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-98.886267, len=752.800000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-207.682766, 61.177651]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_landing_reward | 6315.711812 | 93.6% | 93.6% | 100.0% |
| distance_cost | -426.013595 | -6.3% | 6.3% | 100.0% |
| stability_cost | -6.479766 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
该环境是一个 2D 飞行器/载具轨迹优化任务。一个主体从视口顶部中央区域出发并受到随机初始力。目标是**尽快到达并稳定降落在视口中央的目标着陆垫上**，同时**尽量少使用引擎推力**。智能体需要学习接近目标、减速、保持稳定姿态并安全触地（通过左右支撑腿接触），而不是主体碰撞。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 默认 float32  
- obs[0] (x_position): 水平坐标（相对于目标垫的位置，负表示在左，正在右）  
- obs[1] (y_position): 垂直坐标（相对于目标垫高度的位置，负表示低于垫，正表示高于垫）  
- obs[2] (x_velocity): 水平线速度  
- obs[3] (y_velocity): 垂直线速度  
- obs[4] (body_angle): 主体朝向角（弧度或其它单位）  
- obs[5] (angular_velocity): 角速度  
- obs[6] (left_support_contact): 左支撑腿接触标志（1.0 表示接触，0.0 表示未接触）  
- obs[7] (right_support_contact): 右支撑腿接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete(4)  
- action 0: no_engine —— 不执行任何引擎推力  
- action 1: left_orientation_engine —— 点燃左姿态引擎（提供旋转力矩）  
- action 2: main_engine —— 点燃主引擎（提供主要推力，通常朝下或沿主体方向）  
- action 3: right_orientation_engine —— 点燃右姿态引擎（提供反向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` —— 主体停止运动且稳定。结合任务目标，这通常意味着已落在垫上并稳定，可视为成功着陆。  
- failure-like termination:  
  `crash_or_body_contact` —— 主体与其他表面发生不安全碰撞（区别于支撑腿安全接触）。  
  `horizontal_position_outside_viewport` —— 水平位置超出视口边界。  
- ambiguous termination: 无。注意：`body_not_awake_or_settled` 在极少数异常情况下（如卡在空中）才可能不是成功，但通常环境设计使其与成功着陆关联。  
- truncation: 无（该环境使用终止而非截断，info 为空且未提及 max steps）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 为空字典 {}，不允许推断或在此之上构造信号）  
- forbidden_or_uncertain_info_fields: 任何 info 字段均禁止使用，因为 info 中不包含 success、failure 或 termination_reason 等字段。

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) 及对应的 next_obs 值  
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) 及 next_obs  
- orientation: obs[4] (body_angle), obs[5] (angular_velocity)  
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) —— 表示着陆腿是否安全触地  
- action/engine: 当前动作 action（0~3），可用于惩罚引擎使用（如 action 不为 0 时惩罚推力）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_cost + soft_landing_bonus + stability_cost | -119.56 | -119.56 | 0.00 | 68.30 | distance_cost=-1.111 soft_landing_bonus=0.074 stability_cost=-0.147 | new_best |
| 2 | approach_landing_reward + distance_cost + stability_cost | -98.89 | -98.89 | 0.00 | 752.80 | approach_landing_reward=8.163 distance_cost=-0.521 stability_cost=-0.042 | new_best |
