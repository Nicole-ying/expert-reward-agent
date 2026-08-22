# Search objective
- target_score: 2000.000000
- current_score: -228.023737
- gap_to_target: 2228.023737
- target_achievement_ratio: -11.401%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -228.023737）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_x_vel = next_obs[13]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_z = next_obs[0]

    # 直立因子（body_up_z ∈ [-1, 1]，1 为全直立）
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, body_up_z)

    # 前向奖励（仅当身体朝上时）
    forward = 1.0 * body_x_vel * upright_factor

    # 身体高度铰链惩罚：在高度接近 0.2 前发出警告
    height_threshold = 0.3
    w_height_pen = 1.0
    height_pen = -w_height_pen * max(0.0, height_threshold - body_z)

    total_reward = forward + height_pen

    components = {
        "forward": forward,
        "height_pen": height_pen
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-228.023737, len=215.400000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-2106.247513, 26.922948]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 65.935228 | 89.9% | 95.5% | 49.8% |
| height_pen | -3.281502 | -4.5% | 4.5% | 42.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 3D 四足机器人连续运动控制任务。机器人拥有 8 个扭矩控制的关节（4 条腿各含髋关节和踝关节），需要学习向前行走或跑动，同时保持身体直立并维持在健康高度范围内（0.2 ~ 1.0）。核心目标是稳定、持续地向前移动，而不是简单地保持平衡或待在原地。次要目标是动作平滑、能耗合理。不应混淆的目标：本任务没有明确的到达点或目标位置，因此不是导航到达任务；也无需抓取物体，不属于操作任务。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: [27]
- dtype: float32（推断）
- 每一维含义与奖励可用性：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | body_z | 机器人主身体垂直位置（高度） | true |
| 1 | quat_w | 身体姿态四元数实部 (w) | true（用于计算直立向量） |
| 2 | quat_x | 四元数分量 x | true（结合计算直立向量） |
| 3 | quat_y | 四元数分量 y | true |
| 4 | quat_z | 四元数分量 z | true |
| 5 | joint_1_angle | 第1髋关节角度 | true（用于动作/姿态罚） |
| 6 | joint_2_angle | 第1踝关节角度 | true |
| 7 | joint_3_angle | 第2髋关节角度 | true |
| 8 | joint_4_angle | 第2踝关节角度 | true |
| 9 | joint_5_angle | 第3髋关节角度 | true |
| 10 | joint_6_angle | 第3踝关节角度 | true |
| 11 | joint_7_angle | 第4髋关节角度 | true |
| 12 | joint_8_angle | 第4踝关节角度 | true |
| 13 | body_x_velocity | 身体世界坐标系下的前向速度（x 方向） | true（核心推进信号） |
| 14 | body_y_velocity | 身体侧向速度（y 方向） | true（可用于抑制侧移） |
| 15 | body_z_velocity | 身体垂直速度 | true（用于平稳性） |
| 16 | body_roll_velocity | 横滚角速度 | true |
| 17 | body_pitch_velocity | 俯仰角速度 | true |
| 18 | body_yaw_velocity | 偏航角速度 | true |
| 19 | joint_1_velocity | 第1髋关节角速度 | true |
| 20 | joint_2_velocity | 第1踝关节角速度 | true |
| 21 | joint_3_velocity | 第2髋关节角速度 | true |
| 22 | joint_4_velocity | 第2踝关节角速度 | true |
| 23 | joint_5_velocity | 第3髋关节角速度 | true |
| 24 | joint_6_velocity | 第3踝关节角速度 | true |
| 25 | joint_7_velocity | 第4髋关节角速度 | true |
| 26 | joint_8_velocity | 第4踝关节角速度 | true |

注意：没有接触力、没有世界坐标 x/y，前进进度只能通过 body_x_velocity 反映。

## 4. 动作空间 action_space
- type: Box (连续)
- shape: [8]
- 范围：[-1.0, 1.0] 每关节
- 动作维度映射：

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | hip_1_torque | 第1髋关节扭矩 |
| 1 | ankle_1_torque | 第1踝关节扭矩 |
| 2 | hip_2_torque | 第2髋关节扭矩 |
| 3 | ankle_2_torque | 第2踝关节扭矩 |
| 4 | hip_3_torque | 第3髋关节扭矩 |
| 5 | ankle_3_torque | 第3踝关节扭矩 |
| 6 | hip_4_torque | 第4髋关节扭矩 |
| 7 | ankle_4_torque | 第4踝关节扭矩 |

所有动作都是连续扭矩指令，施加到对应的关节。

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: 无显式成功终止。任务没有“到达目标”的概念，存活且前进是持续的。唯一正面终止是达到时间上限（truncation），表示成功存活整个回合。
- failure-like termination:
  - body_height_outside_healthy_range: 身体高度低于 0.2 或高于 1.0，分别表示跌倒或异常腾空，属于失败。
  - state_value_outside_finite_range: 任何状态出现 NaN 或 Inf，表示物理崩溃，属于失败。
- ambiguous termination: 无。
- truncation: time_limit_reached（达到最大步数），可视为存活成功，但需配合前进表现评估实际性能。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 字典为空，无 success 字段）
- explicit_failure_flag_available: false （无 failure 字段，但终止理由可由 terminated 推断）
- allowed_info_fields: [] （无可用字段）
- forbidden_or_uncertain_info_fields: 任何 info 键都不可用，因为官方 info 被禁止。特别禁止字段：reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin。

## 7. 可用于奖励函数的信号
- position: body_z（高度），四元数姿态（可算直立投影）
- velocity: body_x_velocity（前进速度）、body_y_velocity（侧向速度）、body_z_velocity（垂直速度）、body_roll/pitch/yaw_velocity（角速度），各关节角速度
- orientation: body_up_z = 1 - 2*(quat_x² + quat_y²)（直立度量，范围 [−1,1]，１为全直立）
- contact: 无
- action/engine: 关节扭矩（action），8维连续值
- other: 可以通过 next_obs - obs 计算速度变化、高度变化等瞬时增量

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
| 1 | forward + height + upright | -371.50 | -371.50 | 0.00 | 320.00 | forward=0.188 height=-0.010 upright=-14.477 | new_best |
| 2 | forward + height + upright | -32.40 | -32.40 | 0.00 | 89.65 | forward=0.539 height=-0.012 upright=-1.702 | new_best |
| 3 | forward + height | -521.05 | -32.40 | -488.65 | 462.60 | forward=0.104 height=-0.002 | no_meaningful_improvement |
| 4 | forward + height | -338.02 | -32.40 | -305.62 | 492.10 | forward=0.291 height=-0.002 | no_meaningful_improvement |
| 5 | forward + height + upright_pen | 263.65 | 263.65 | 0.00 | 591.40 | forward=0.581 height=-0.004 upright_pen=-0.703 | new_best |
| 6 | forward + height_gate + upright_pen | -53.61 | 263.65 | -317.26 | 308.95 | forward=0.426 height_gate=0.796 upright_pen=-0.768 | no_meaningful_improvement |
| 7 | forward + upright_pen | -145.24 | 263.65 | -408.90 | 167.00 | forward=0.440 upright_pen=-0.898 | no_meaningful_improvement |
| 8 | forward + height_pen | -228.02 | 263.65 | -491.68 | 215.40 | forward=0.220 height_pen=-0.017 | unsolved_high_achievement_continue_from_best |
