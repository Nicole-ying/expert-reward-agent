# Search objective
- target_score: 2000.000000
- current_score: -224.845626
- gap_to_target: 2224.845626
- target_achievement_ratio: -11.242%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -224.845626）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract signals ----------
    body_z          = obs[0]
    qw, qx, qy, qz = obs[1], obs[2], obs[3], obs[4]
    body_x_vel      = obs[13]
    body_y_vel      = obs[14]
    roll_vel        = obs[16]
    pitch_vel       = obs[17]

    # compute torso uprightness (world z projection)
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)

    # -------- 1. forward progress (bounded, scaled up) --------
    forward_reward = 20.0 * (body_x_vel / (1.0 + abs(body_x_vel)))

    # -------- 2. health height (soft boundary cost) --------
    height_penalty = 0.0
    if body_z < 0.3:
        height_penalty = -20.0 * (0.3 - body_z)
    elif body_z > 0.9:
        height_penalty = -20.0 * (body_z - 0.9)

    # -------- 3. upright stability (posture + angular velocity) --------
    upright_reward = 1.0 * body_up_z
    ang_penalty    = -0.005 * (roll_vel**2 + pitch_vel**2)
    upright_stability = upright_reward + ang_penalty

    # -------- 4. lateral drift penalty --------
    lateral_penalty = -0.1 * (body_y_vel**2)

    # -------- total reward and components --------
    total = forward_reward + height_penalty + upright_stability + lateral_penalty

    components = {
        "forward_progress":       forward_reward,
        "height_penalty":         height_penalty,
        "upright_stability":      upright_stability,
        "lateral_drift_penalty":  lateral_penalty
    }

    return float(total), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-224.845626, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-391.757258, 585.070983]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 675.563807 | 22.2% | 68.1% | 95.5% |
| upright_stability | 970.297156 | 31.8% | 31.8% | 100.0% |
| lateral_drift_penalty | -1.207796 | -0.0% | 0.0% | 83.5% |
| height_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
学习一个连续控制的四足机器人（8个扭矩控制关节）在3D空间中稳定向前行走或跑动。任务的核心是产生持续的前向平移，同时保持身体高度在健康区间（0.2~1.0）且姿态基本直立。任何因为倒地、弹飞或数值异常导致的提前终止都视为失败。策略不能仅仅保持静止不倒，必须展现出有效的前进运动。

## 3. 观察空间 observation_space
- type: Box  
- shape: [27]  
- dtype: float32（推断）  
- 各维度含义与奖励可用性：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | body_z | 躯干质心的世界 z 坐标（高度） | true |
| 1 | quat_w | 方向四元数实部 w | true（用于计算姿态） |
| 2 | quat_x | 方向四元数虚部 x | true |
| 3 | quat_y | 方向四元数虚部 y | true |
| 4 | quat_z | 方向四元数虚部 z | true |
| 5 | joint_1_angle | 第1髋关节角 | true（位置信号） |
| 6 | joint_2_angle | 第1踝关节角 | true |
| 7 | joint_3_angle | 第2髋关节角 | true |
| 8 | joint_4_angle | 第2踝关节角 | true |
| 9 | joint_5_angle | 第3髋关节角 | true |
| 10 | joint_6_angle | 第3踝关节角 | true |
| 11 | joint_7_angle | 第4髋关节角 | true |
| 12 | joint_8_angle | 第4踝关节角 | true |
| 13 | body_x_velocity | 躯干在世界 x 方向（前方）的速度 | true **← 核心前进信号** |
| 14 | body_y_velocity | 躯干在世界 y 方向（侧向）的速度 | true |
| 15 | body_z_velocity | 躯干在 z 方向的速度 | true |
| 16 | body_roll_velocity | 绕 x 轴角速度（翻滚） | true |
| 17 | body_pitch_velocity | 绕 y 轴角速度（俯仰） | true |
| 18 | body_yaw_velocity | 绕 z 轴角速度（偏航） | true |
| 19 | joint_1_velocity | 第1髋关节角速度 | true |
| 20 | joint_2_velocity | 第1踝关节角速度 | true |
| 21 | joint_3_velocity | 第2髋关节角速度 | true |
| 22 | joint_4_velocity | 第2踝关节角速度 | true |
| 23 | joint_5_velocity | 第3髋关节角速度 | true |
| 24 | joint_6_velocity | 第3踝关节角速度 | true |
| 25 | joint_7_velocity | 第4髋关节角速度 | true |
| 26 | joint_8_velocity | 第4踝关节角速度 | true |

额外可计算信号：body_up_z = 1 - 2*(quat_x² + quat_y²)，表示躯干 z 轴在世界 z 轴的投影（1.0 为完全直立）。

## 4. 动作空间 action_space
- type: Box  
- shape: [8]  
- continuous: true  
- 界限: [-1.0, 1.0]（每个关节扭矩）  

| 动作维度 | 名称 | 含义 |
|----------|------|------|
| 0 | hip_1_torque | 第1髋关节扭矩 |
| 1 | ankle_1_torque | 第1踝关节扭矩 |
| 2 | hip_2_torque | 第2髋关节扭矩 |
| 3 | ankle_2_torque | 第2踝关节扭矩 |
| 4 | hip_3_torque | 第3髋关节扭矩 |
| 5 | ankle_3_torque | 第3踝关节扭矩 |
| 6 | hip_4_torque | 第4髋关节扭矩 |
| 7 | ankle_4_torque | 第4踝关节扭矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **无明确成功终止**。正常存活到时间上限（truncation）可视为一种成功存活信号，但未保证前进距离。
- failure-like termination: terminated=True 由以下条件触发：
  - body_height_outside_healthy_range: body_z ≤ 0.2 或 body_z ≥ 1.0（倒塌/弹飞）
  - state_value_outside_finite_range: 任何状态值变为 NaN 或无穷大
- ambiguous termination: 无。
- truncation: 时间上限到达，truncated=True。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 被屏蔽，无法获取任何标志）
- explicit_failure_flag_available: false （同上，但可从 terminated 原因推断失败）
- allowed_info_fields: [] （空列表，意味着 **info 完全不可用**）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止，包括 reward_forward、reward_ctrl、reward_contact、reward_survive、x_position、y_position、distance_from_origin 等。

## 7. 可用于奖励函数的信号
- position: body_z（高度），关节角度（8个），身体姿态四元数（可计算 body_up_z）
- velocity: body_x_velocity（前向速度），body_y_velocity（侧向漂移），body_z_velocity（上下），body_roll/pitch/yaw_velocity（姿态变化率），8个关节速度
- orientation: 通过四元数计算身体直立度（body_up_z），或惩罚较大滚动/俯仰角速度
- action/engine: 8维扭矩动作（可用于动作幅度惩罚或平滑性约束）
- other: 从姿态推导的 body_up_z，身体高度距离健康边界的距离

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_progress + height_penalty + lateral_drift_penalty + upright_stability | 434.48 | 434.48 | 0.00 | 1000.00 | forward_progress=0.054 height_penalty=-0.311 lateral_drift_penalty=-0.044 upright_stability=0.037 | new_best |
| 2 | forward_progress + height_penalty + lateral_drift_penalty + upright_stability | -224.85 | 434.48 | -659.32 | 1000.00 | forward_progress=1.057 height_penalty=-0.374 lateral_drift_penalty=-0.044 upright_stability=-0.146 | no_meaningful_improvement |
