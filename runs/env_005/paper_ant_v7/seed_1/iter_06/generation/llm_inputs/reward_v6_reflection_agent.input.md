# 1. Search objective
- target_score: 2000.000000
- current_score: -1815.604024
- gap_to_target: 3815.604024
- target_achievement_ratio: -90.780%

# 2. 上一轮奖励函数代码（该轮得分: -1815.604024）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs after the action
    body_z = next_obs[0]
    qx = next_obs[2]          # quaternion x
    qy = next_obs[3]          # quaternion y
    body_x_vel = next_obs[13] # forward velocity
    body_y_vel = next_obs[14] # lateral velocity

    # 1. Health gate: height factor (safe range [0.3, 0.9])
    if body_z < 0.3:
        height_factor = (body_z - 0.2) / 0.1   # decreases to 0 at z=0.2
    elif body_z > 0.9:
        height_factor = (1.0 - body_z) / 0.1   # decreases to 0 at z=1.0
    else:
        height_factor = 1.0
    height_factor = max(0.0, min(1.0, height_factor))

    # 2. Health gate: upright factor (from body_up_z = 1 - 2*(qx^2 + qy^2))
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    # Map [ -0.2 , 0.5 ] to [0, 1]
    upright_factor = (body_up_z + 0.2) / 0.7
    upright_factor = max(0.0, min(1.0, upright_factor))

    health_gate = height_factor * upright_factor

    # 3. Main learning signal: squared forward velocity, gated by health
    forward_speed = max(0.0, body_x_vel)
    progress_reward = 2.0 * (forward_speed ** 2) * health_gate

    # 4. Light regularization against lateral drift
    lateral_penalty = -0.5 * (body_y_vel ** 2)

    total_reward = progress_reward + lateral_penalty
    components = {
        'forward_progress': progress_reward,
        'lateral_penalty': lateral_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 694.35 | 2.91 | ✅ |
| 2 | 增加侧向漂移惩罚迫使 agent 学习直线行走（减少无效能量消耗），提升 action_penalty 至 0.0... | 增加侧向漂移惩罚迫使 agent 学习直线行走（减少无效能量消耗），提升 action_penalty 至 0.0... | 625.10 | -416.10 | ❌ |
| 3 | 将 lateral_penalty 乘以 survival_gate 可在 agent 接近边界时减弱侧向惩罚，减... | 将 lateral_penalty 乘以 survival_gate 可在 agent 接近边界时减弱侧向惩罚，减... | 740.65 | 13.55 | ✅ |
| 4 | 放宽 upright gate 会解除对前倾的惩罚，让 agent 敢于尝试更快的步态，从而提升实际前进速度与外部... | 放宽 upright gate 会解除对前倾的惩罚，让 agent 敢于尝试更快的步态，从而提升实际前进速度与外部... | 583.05 | -464.62 | ❓ |
| 5 | 骨架变化: forward_progress + lateral_penalty | — | 1000.00 | -1815.60 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=-1815.604024, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-2481.114562, -1158.640038]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 205.693192 | 85.8% | 85.8% | 14.3% |
| lateral_penalty | -34.071995 | -14.2% | 14.2% | 24.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一项连续控制运动任务：控制一个具有 8 个扭矩关节的四足机器人在 3D 环境中向前行进（走路或奔跑），同时必须保持身体高度处于健康范围内（0.2 m ~ 1.0 m）。主要目标是实现稳定的前进速度，而非仅仅保持站立；次要目标可包括运动平滑性、能量效率及减少侧向漂移等。任务不应与简单悬停或固定位置平衡混淆。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32 （推断，未明确指定但通常如此）
- 各维度含义与奖励可用性：

| index | 名称               | 含义                                      | reward_usable |
|-------|-------------------|-------------------------------------------|---------------|
| 0     | body_z            | 身体质心高度 (m)                           | true          |
| 1     | quat_w            | 方向四元数实部 w                           | true (用于计算直立度) |
| 2     | quat_x            | 四元数 x 分量                              | true (用于计算直立度) |
| 3     | quat_y            | 四元数 y 分量                              | true (用于计算直立度) |
| 4     | quat_z            | 四元数 z 分量                              | true (用于计算直立度) |
| 5     | joint_1_angle     | 前左髋关节角度                              | true          |
| 6     | joint_2_angle     | 前左踝关节角度                              | true          |
| 7     | joint_3_angle     | 前右髋关节角度                              | true          |
| 8     | joint_4_angle     | 前右踝关节角度                              | true          |
| 9     | joint_5_angle     | 后左髋关节角度                              | true          |
| 10    | joint_6_angle     | 后左踝关节角度                              | true          |
| 11    | joint_7_angle     | 后右髋关节角度                              | true          |
| 12    | joint_8_angle     | 后右踝关节角度                              | true          |
| 13    | body_x_velocity   | 世界坐标系下身体前进速度 (m/s)               | true (主指标) |
| 14    | body_y_velocity   | 世界坐标系下侧向速度 (m/s)                   | true          |
| 15    | body_z_velocity   | 垂直速度 (m/s)                              | true          |
| 16    | body_roll_velocity| 滚转角速度 (rad/s)                          | true          |
| 17    | body_pitch_velocity| 俯仰角速度 (rad/s)                         | true          |
| 18    | body_yaw_velocity | 偏航角速度 (rad/s)                          | true          |
| 19    | joint_1_velocity  | 关节1角速度                                | true          |
| 20    | joint_2_velocity  | 关节2角速度                                | true          |
| 21    | joint_3_velocity  | 关节3角速度                                | true          |
| 22    | joint_4_velocity  | 关节4角速度                                | true          |
| 23    | joint_5_velocity  | 关节5角速度                                | true          |
| 24    | joint_6_velocity  | 关节6角速度                                | true          |
| 25    | joint_7_velocity  | 关节7角速度                                | true          |
| 26    | joint_8_velocity  | 关节8角速度                                | true          |

注：观测中不包含全局 x/y 位置、接触力或任何地面反力信息。

## 4. 动作空间 action_space
- type: Box (continuous)
- shape: (8,)
- 范围：每个关节 [-1.0, 1.0]（扭矩归一化或扭矩值本身）
- 各维度含义：

| action dim | 名称             | 含义                         |
|-----------|------------------|------------------------------|
| 0         | hip_1_torque     | 前左髋关节扭矩               |
| 1         | ankle_1_torque   | 前左踝关节扭矩               |
| 2         | hip_2_torque     | 前右髋关节扭矩               |
| 3         | ankle_2_torque   | 前右踝关节扭矩               |
| 4         | hip_3_torque     | 后左髋关节扭矩               |
| 5         | ankle_3_torque   | 后左踝关节扭矩               |
| 6         | hip_4_torque     | 后右髋关节扭矩               |
| 7         | ankle_4_torque   | 后右踝关节扭矩               |

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确成功终止；任务没有设定“到达目标点”等成功条件。
- failure-like termination: 
  1. body_height_outside_healthy_range：身体高度超出 (0.2, 1.0)（过低视为跌倒/倒塌，过高视为异常弹跳或失控）。
  2. state_value_outside_finite_range：任何状态值为 NaN 或无穷。
- ambiguous termination: 无（上述均为失败类终止）
- truncation: 时间限制到达（time_limit_reached），通常为 max_episode_steps 截断，不应视为成功或失败，只能作为“存活但未完成特定目标”。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false （终止标志 `terminated` 可间接视为失败触发，但未在 info 中提供明确 label）
- allowed_info_fields: [] （空，不能从 info 获取任何信息）
- forbidden_or_uncertain_info_fields:
  - reward_forward
  - reward_ctrl
  - reward_contact
  - reward_survive
  - x_position
  - y_position
  - distance_from_origin
  这些字段明确被禁止用于奖励函数设计。

## 7. 可用于奖励函数的信号
这些信号均可从 obs/next_obs/action 获得：

- position:
  - 身体高度 `body_z` (obs[0])，可评估与理想高度的偏差。
  - 所有关节角度 obs[5:13]，可鼓励或惩罚特定姿态。
- velocity:
  - 前进速度 `body_x_velocity` (obs[13])，直接反映前进表现。
  - 侧向速度 `body_y_velocity` (obs[14])，可用于惩罚侧向漂移。
  - 垂直速度 `body_z_velocity` (obs[15])，可惩罚过度跳跃或不稳定。
  - 身体角速度 (roll/pitch/yaw) obs[16:19]，可惩罚剧烈摇晃。
  - 关节角速度 obs[19:27]，可惩罚关节振动或高能耗。
- orientation:
  - 由四元数 obs[1:5] 可计算 body_up_z = 1 - 2*(quat_x² + quat_y²)（取值范围约 -1 到 1，1 表示完全直立），可用于奖励直立姿态。
- action/engine:
  - 动作扭矩 `action` 的 L2 范数或绝对值和，用于鼓励小力矩（节能）。
  - 动作变化率（需要前后步动作，但 `compute_reward` 只有单步信息，无法直接获前一动作），因此动作平滑（变化惩罚）只能通过两帧动作差来近似，若函数不提供 `prev_action`，该信号不可用。通常环境要求无状态，故可能禁用动作变化惩罚。

- other:
  - 存活信号可用 `body_z` 是否在健康范围内（但范围外的已终止不会调用 reward 函数，所以密集奖励可用到 “接近边界” 的惩罚）。

# 6. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 7. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + forward_gated | 2.91 | 2.91 | 0.00 | 694.35 | action_penalty=-0.002 forward_gated=0.425 | new_best |
| 2 | action_penalty + forward_gated + lateral_penalty | -416.10 | 2.91 | -419.01 | 625.10 | action_penalty=-0.043 forward_gated=0.145 lateral_penalty=-0.051 | no_meaningful_improvement |
| 3 | action_penalty + forward_gated + lateral_penalty | 13.55 | 13.55 | 0.00 | 740.65 | action_penalty=-0.042 forward_gated=0.307 lateral_penalty=-0.058 | new_best |
| 4 | action_penalty + forward_gated + lateral_penalty | -464.62 | 13.55 | -478.17 | 583.05 | action_penalty=-0.044 forward_gated=0.211 lateral_penalty=-0.055 | unsolved_stagnation_fresh_restart |
| 5 | forward_progress + lateral_penalty | -1815.60 | 13.55 | -1829.15 | 1000.00 | forward_progress=0.285 lateral_penalty=-0.101 | no_meaningful_improvement |
