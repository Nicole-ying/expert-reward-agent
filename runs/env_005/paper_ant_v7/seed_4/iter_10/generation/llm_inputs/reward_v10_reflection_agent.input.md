# 1. Search objective
- target_score: 2000.000000
- current_score: -388.820358
- gap_to_target: 2388.820358
- target_achievement_ratio: -19.441%

# 2. 上一轮奖励函数代码（该轮得分: -388.820358）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    body_z = obs[0]          # torso height
    quat_x = obs[2]          # quaternion x component
    quat_y = obs[3]          # quaternion y component
    body_x_vel = obs[13]     # forward velocity
    body_y_vel = obs[14]     # lateral velocity
    roll_vel  = obs[16]      # body angular velocities
    pitch_vel = obs[17]
    yaw_vel   = obs[18]

    # Compute uprightness (z-component of body up vector)
    # Assuming unit quaternion and near‑upright the w component ≈ 1.
    uprightness = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    uprightness = max(-1.0, min(1.0, uprightness))

    # Soft health gate – height component
    # Gate opens linearly from 0 at 0.25 m to 1 at 0.35 m.
    height_gate = (body_z - 0.25) / (0.35 - 0.25)
    height_gate = max(0.0, min(1.0, height_gate))

    # Soft health gate – uprightness component
    # Gate opens linearly from 0 at uprightness 0.5 to 1 at 0.9.
    upright_gate = (uprightness - 0.5) / (0.9 - 0.5)
    upright_gate = max(0.0, min(1.0, upright_gate))

    # Combined gate: minimum ensures both health conditions are partially satisfied
    forward_gate = min(height_gate, upright_gate)

    # Main learning signal: gated forward velocity
    w_fwd = 1.0
    forward_reward = w_fwd * body_x_vel * forward_gate

    # Stabilisation: suppress lateral drift
    w_lat = 0.1
    lateral_penalty = -w_lat * (body_y_vel**2)

    # Stabilisation: damp excessive body rotation
    w_ang = 0.05
    angular_penalty = -w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    total_reward = forward_reward + lateral_penalty + angular_penalty

    components = {
        'forward_reward': forward_reward,
        'lateral_penalty': lateral_penalty,
        'angular_penalty': angular_penalty,
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 647.80 | -544.72 | ✅ |
| 2 | 添加动作惩罚将抑制高扭矩操作，引导学习更稳定、节能的步态，预期可降低终止率并提高前进效率 | 添加动作惩罚将抑制高扭矩操作，引导学习更稳定、节能的步态，预期可降低终止率并提高前进效率 | 349.35 | -49.90 | ✅ |
| 3 | 将高度约束改为 gate 将迫使 agent 在高度不良时接受近乎零的前进奖励，打破「加速→失衡→死亡」的循环。添... | 将高度约束改为 gate 将迫使 agent 在高度不良时接受近乎零的前进奖励，打破「加速→失衡→死亡」的循环。添... | 16.35 | -9.56 | ✅ |
| 4 | 减小旋转和侧向惩罚系数使每步净奖励为正，agent 存活时间变长后能获取更多前进奖励，从而自发降低角速度。 | 减小旋转和侧向惩罚系数使每步净奖励为正，agent 存活时间变长后能获取更多前进奖励，从而自发降低角速度。 | 797.25 | -779.13 | ❌ |
| 5 | 将前进奖励系数放大 10 倍后，高速行走的回报将显著超过苟活策略，agent 会被迫学习更快、更稳定的步态，从而大... | 将前进奖励系数放大 10 倍后，高速行走的回报将显著超过苟活策略，agent 会被迫学习更快、更稳定的步态，从而大... | 493.45 | -337.63 | ❓ |
| 6 | 骨架变化: angular_velocity_penalty + gated_forward + height_ | — | 577.30 | -184.03 | ❌ |
| 7 | 骨架变化: angular_velocity_penalty + gated_forward + height_ | — | 663.90 | -282.04 | ❌ |
| 8 | 解耦前进奖励与姿态门控，将奖励地形变得平滑连续，agent 将敢于加速并自发保持必要的稳定姿态，从而在存活时间提升... | 解耦前进奖励与姿态门控，将奖励地形变得平滑连续，agent 将敢于加速并自发保持必要的稳定姿态，从而在存活时间提升... | 616.50 | -6.63 | ❓ |
| 9 | 骨架变化: angular_penalty + forward_reward + lateral_penalty | — | 702.30 | -388.82 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=-388.820358, len=702.300000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-1005.673116, -12.933700]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 640.835676 | 81.7% | 83.3% | 93.6% |
| angular_penalty | -118.070765 | -15.1% | 15.1% | 94.9% |
| lateral_penalty | -13.081515 | -1.7% | 1.7% | 94.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个连续控制运动任务，控制一台具有四条腿、八个扭矩驱动关节的 3D 四足机器人。核心任务是让它持续、稳定地向前行走或奔跑，同时保持身体竖直。  
次要目标包括：抑制侧向漂移、降低身体角速度、减小关节动作能耗。机器人身体高度必须维持在健康范围内（0.2～1.0），否则立刻终止；同时任何状态量若变为 `NaN` 或 `inf` 也会终止。  
不应被混淆的目标：这里没有预定义的目标位置，不能把任务简单地当成“站立平衡”任务，因为必须产生稳定的前向位移才能持续获得奖励。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: 连续浮点（推测 float32）
- 各维度含义（索引 0～26）：
  - obs[0]: `body_z`，身体质心在世界坐标系中的高度，可用于检查健康范围，reward_usable: true
  - obs[1]: `quat_w`，身体姿态四元数的实部，reward_usable: true（用于计算直立程度）
  - obs[2]: `quat_x`，四元数第一虚部，reward_usable: true
  - obs[3]: `quat_y`，四元数第二虚部，reward_usable: true
  - obs[4]: `quat_z`，四元数第三虚部，reward_usable: true
  - obs[5]: `joint_1_angle`（第一个髋关节角度），reward_usable: true（可选用于姿态约束）
  - obs[6]: `joint_2_angle`（第一个踝关节角度），reward_usable: true
  - obs[7]: `joint_3_angle`（第二个髋关节角度），reward_usable: true
  - obs[8]: `joint_4_angle`（第二个踝关节角度），reward_usable: true
  - obs[9]: `joint_5_angle`（第三个髋关节角度），reward_usable: true
  - obs[10]: `joint_6_angle`（第三个踝关节角度），reward_usable: true
  - obs[11]: `joint_7_angle`（第四个髋关节角度），reward_usable: true
  - obs[12]: `joint_8_angle`（第四个踝关节角度），reward_usable: true
  - obs[13]: `body_x_velocity`，身体在世界 x 方向的线速度（前进速度），核心信号，reward_usable: true
  - obs[14]: `body_y_velocity`，身体在世界 y 方向（横向）的线速度，reward_usable: true（可惩罚侧向漂移）
  - obs[15]: `body_z_velocity`，身体在竖直方向的线速度，reward_usable: true（可选）
  - obs[16]: `body_roll_velocity`，滚转角速度，reward_usable: true
  - obs[17]: `body_pitch_velocity`，俯仰角速度，reward_usable: true
  - obs[18]: `body_yaw_velocity`，偏航角速度，reward_usable: true
  - obs[19]: `joint_1_velocity`，第一个髋关节角速度，reward_usable: true（可选用于平滑或能耗）
  - obs[20]: `joint_2_velocity`，第一个踝关节角速度，reward_usable: true
  - obs[21]: `joint_3_velocity`，第二个髋关节角速度，reward_usable: true
  - obs[22]: `joint_4_velocity`，第二个踝关节角速度，reward_usable: true
  - obs[23]: `joint_5_velocity`，第三个髋关节角速度，reward_usable: true
  - obs[24]: `joint_6_velocity`，第三个踝关节角速度，reward_usable: true
  - obs[25]: `joint_7_velocity`，第四个髋关节角速度，reward_usable: true
  - obs[26]: `joint_8_velocity`，第四个踝关节角速度，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- 连续动作，每个维度范围 [-1.0, 1.0]（归一化扭矩指令）
- 各维度含义：
  - action[0]: `hip_1_torque`（第 1 条腿髋关节扭矩）
  - action[1]: `ankle_1_torque`（第 1 条腿踝关节扭矩）
  - action[2]: `hip_2_torque`
  - action[3]: `ankle_2_torque`
  - action[4]: `hip_3_torque`
  - action[5]: `ankle_3_torque`
  - action[6]: `hip_4_torque`
  - action[7]: `ankle_4_torque`

## 5. step 与终止条件分析
### 5.1 终止模式
- **失败型终止（failure-like termination）**：
  - 身体高度超出健康范围：`body_z ≤ 0.2` 或 `body_z ≥ 1.0`（collapse 或 jump/launch）。
  - 任意状态量变为 `NaN` 或 `inf`（在模拟中通常由极端碰撞或数值发散导致）。
- **截断型终止（truncation）**：
  - 达到时间步上限 `time_limit_reached`（超出最大步数），属于正常截断，不应直接视为成功或失败。

- **成功标志**：不存在显式的 `info["success"]` 字段。
- **模糊终止**：无。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均被禁止，包括 `reward_forward`、`reward_ctrl`、`reward_contact`、`reward_survive`、`x_position`、`y_position`、`distance_from_origin` 等官方内部项。

注意：任务没有提供任何显式的成功/失败标志，奖励函数设计必须完全基于观察和动作信号，不能依赖 `info` 中的任何信息。

## 7. 可用于奖励函数的信号
- **位置与姿态**：
  - `body_z`（obs[0]）：用于高度存活惩罚。
  - 身体姿态四元数（obs[1:5]）：通过 `body_up_z = 1 - 2*(quat_x^2 + quat_y^2)` 计算身体 z 轴在世界坐标系中的投影，1.0 表示完全直立，0 表示水平。
  - 关节角度（obs[5:13]）：可用于姿态正则（但默认理想姿态未知，慎用）。

- **速度**：
  - 前向速度 `body_x_velocity`（obs[13]）：核心前进奖励信号。
  - 侧向速度 `body_y_velocity`（obs[14]）：横向漂移惩罚信号。
  - 垂直速度 `body_z_velocity`（obs[15]）：可选用于稳定性（例如惩罚快速起落）。
  - 角速度 `body_roll/pitch/yaw_velocity`（obs[16:19]）：旋转惩罚信号。
  - 关节角速度（obs[19:27]）：可用于动作平滑或能耗建模。

- **接触/力**：无。环境未提供接触力或足端接触布尔量。

- **动作/引擎**：
  - 动作向量 `action`：可直接惩罚扭矩大小（`torque_penalty = sum(action^2)`）或动作变化（若允许使用 `prev_action`，但本接口为单步无状态，无法直接获取上一步动作，因此只能使用当前动作幅度惩罚）。

- **其他**：`next_obs` 可对上述信号计算差分或在下个状态施加条件（例如若使用 next_obs 的身体高度，但可能导致奖励归属偏移，需谨慎）。

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
| 1 | forward_upright_product + height_survival_penalty | -544.72 | -544.72 | 0.00 | 647.80 | forward_upright_product=0.253 height_survival_penalty=-0.023 | new_best |
| 2 | forward_upright_product + height_survival_penalty + torque_penalty | -49.90 | -49.90 | 0.00 | 349.35 | forward_upright_product=0.447 height_survival_penalty=-0.017 torque_penalty=-0.016 | new_best |
| 3 | forward_upright_gated + height_gate_value + lateral_penalty + rotation_penalty + torque_penalty | -9.56 | -9.56 | 0.00 | 16.35 | forward_upright_gated=0.125 height_gate_value=0.450 lateral_penalty=-0.105 rotation_penalty=-0.260 torque_penalty=-0.018 | new_best |
| 4 | forward_upright_gated + height_gate_value + lateral_penalty + rotation_penalty + torque_penalty | -779.13 | -9.56 | -769.57 | 797.25 | forward_upright_gated=0.154 height_gate_value=0.498 lateral_penalty=-0.023 rotation_penalty=-0.068 torque_penalty=-0.018 | no_meaningful_improvement |
| 5 | forward_upright_gated + height_gate_value + lateral_penalty + rotation_penalty + torque_penalty | -337.63 | -9.56 | -328.07 | 493.45 | forward_upright_gated=1.178 height_gate_value=0.395 lateral_penalty=-0.027 rotation_penalty=-0.096 torque_penalty=-0.017 | unsolved_stagnation_fresh_restart |
| 6 | angular_velocity_penalty + gated_forward + height_hinge_penalty + lateral_penalty | -184.03 | -9.56 | -174.48 | 577.30 | angular_velocity_penalty=-0.128 gated_forward=0.751 height_hinge_penalty=-0.151 lateral_penalty=-0.177 | no_meaningful_improvement |
| 7 | angular_velocity_penalty + gated_forward + height_hinge_penalty + lateral_penalty | -282.04 | -9.56 | -272.48 | 663.90 | angular_velocity_penalty=-0.114 gated_forward=0.600 height_hinge_penalty=-0.185 lateral_penalty=-0.143 | no_meaningful_improvement |
| 8 | forward_reward + height_hinge_penalty + upright_reward | -6.63 | -6.63 | 0.00 | 616.50 | forward_reward=1.271 height_hinge_penalty=-0.009 upright_reward=0.179 | unsolved_stagnation_fresh_restart |
| 9 | angular_penalty + forward_reward + lateral_penalty | -388.82 | -6.63 | -382.19 | 702.30 | angular_penalty=-0.172 forward_reward=0.378 lateral_penalty=-0.021 | no_meaningful_improvement |
