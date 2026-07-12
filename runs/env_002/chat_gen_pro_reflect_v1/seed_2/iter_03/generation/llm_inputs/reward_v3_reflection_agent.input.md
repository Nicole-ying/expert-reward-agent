# Search objective
- target_score: 300.000000
- current_score: 304.498984
- gap_to_target: -4.498984
- target_achievement_ratio: 101.500%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 304.498984）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向前速度的凸化奖励
    horizontal_vel = obs[2]
    forward_progress_reward = 1.3 * (horizontal_vel ** 2)

    # 平衡约束（角度与角速度的软惩罚）
    hull_angle = obs[0]
    hull_angular_vel = obs[1]

    angle_threshold = 0.3
    angle_penalty = -0.5 * max(0.0, abs(hull_angle) - angle_threshold) ** 2

    angular_vel_threshold = 2.0
    angular_vel_penalty = -0.1 * max(0.0, abs(hull_angular_vel) - angular_vel_threshold) ** 2

    balance_penalty = angle_penalty + angular_vel_penalty

    # 健康门控：身体角度偏离越大，主奖励衰减越强
    gate_factor = max(0.0, min(1.0, 1.0 - abs(hull_angle) / 0.5))
    gated_forward_reward = forward_progress_reward * gate_factor

    total_reward = gated_forward_reward + balance_penalty

    components = {
        'forward_progress_reward': forward_progress_reward,
        'gated_forward_reward': gated_forward_reward,
        'balance_angle_penalty': angle_penalty,
        'balance_angular_vel_penalty': angular_vel_penalty,
        'gate_factor': gate_factor
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=304.498984, len=885.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[301.215542, 307.036430]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 837.271620 | 52.4% | 52.4% | 100.0% |
| forward_progress_reward | 390.622652 | 24.5% | 24.5% | 100.0% |
| gated_forward_reward | 369.146573 | 23.1% | 23.1% | 100.0% |
| balance_angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| balance_angular_vel_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：任务不是到达某个精确目标点，也不是跳跃或奔跑，而是持续稳定的双足步态前进。能量最小化是附属优化目标，不是核心任务目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推测)
- obs[0]: hull_angle，身体相对于竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，身体角速度，reward_usable: true
- obs[2]: horizontal_velocity，前后线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1地面接触标志(1.0=接触,0.0=无接触)，reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2地面接触标志(1.0=接触,0.0=无接触)，reward_usable: true
- obs[14..23]: lidar_0..9，10个LIDAR测距仪沿前方地形的距离测量值，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- continuous: true
- bounds: [-1.0, 1.0] per joint
- action[0]: hip_torque_leg1，腿1髋关节施加的扭矩
- action[1]: knee_torque_leg1，腿1膝关节施加的扭矩
- action[2]: hip_torque_leg2，腿2髋关节施加的扭矩
- action[3]: knee_torque_leg2，腿2膝关节施加的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain — 成功走完地形，属于成功终止
- failure-like termination: body_fallen_over — 身体摔倒，属于失败终止
- ambiguous termination: 无
- truncation: 无（step 返回 False 作为 truncation 标志）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空字典 {}，没有显式 success 标志）
- explicit_failure_flag_available: false（info 为空字典 {}，没有显式 failure 标志）
- allowed_info_fields: 无（info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为 info 为空

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，但可通过 LIDAR 和速度间接推断
- velocity: horizontal_velocity (obs[2]), vertical_velocity (obs[3]), hull_angular_velocity (obs[1]), joint speeds (obs[5], obs[7], obs[10], obs[12])
- orientation: hull_angle (obs[0]), joint angles (obs[4], obs[6], obs[9], obs[11])
- contact: leg1_contact (obs[8]), leg2_contact (obs[13])
- action/engine: action 本身 (4维扭矩)
- other: LIDAR 测距 (obs[14..23])

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | balance_angle_penalty + balance_angular_vel_penalty + forward_progress_reward + gate_factor + gated_forward_reward | 286.89 | 286.89 | 0.00 | 1024.30 | balance_angle_penalty=-0.001 balance_angular_vel_penalty=0.000 forward_progress_reward=0.078 gate_factor=0.805 gated_forward_reward=0.068 | new_best |
| 2 | balance_angle_penalty + balance_angular_vel_penalty + forward_progress_reward + gate_factor + gated_forward_reward | 304.50 | 304.50 | 0.00 | 885.10 | balance_angle_penalty=-0.002 balance_angular_vel_penalty=0.000 forward_progress_reward=0.153 gate_factor=0.833 gated_forward_reward=0.138 | target_solved_new_best |
