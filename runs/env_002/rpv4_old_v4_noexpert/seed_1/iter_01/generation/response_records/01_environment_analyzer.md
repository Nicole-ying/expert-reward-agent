# Response Record

# 匿名环境理解卡片

## 1. 任务目标

这是一个匿名 2D 双足步行任务。智能体需要控制两条腿的髋关节和膝关节，在起伏地形上尽量向前行走，走得越远、越快越好，同时尽可能降低能量消耗。核心目标是稳定前进并避免摔倒；能量消耗、关节动作平滑、躯干姿态稳定属于次目标，不应与“前进主目标”混淆。

---

## 2. 任务类型选择

selected_route_id: locomotion_continuous_control

confidence: high

reason: 核心目标是持续前进穿越地形，没有固定到达点，没有抓取/操控物体，也不是纯存活/平衡任务。任务要求通过连续关节力矩产生双足步态，能量消耗只是附属优化目标，不属于多目标权重相当的主目标结构。

---

## 3. 观察空间 observation_space

- type: Box
- shape: [24]
- dtype: float（spec 未明确精度，通常为 float32）
- obs[0]: hull_angle — 主躯干相对直立方向的角度；reward_usable: true
- obs[1]: hull_angular_velocity — 主躯干角速度；reward_usable: true
- obs[2]: horizontal_velocity — 前进/后退线速度；reward_usable: true
- obs[3]: vertical_velocity — 上下线速度；reward_usable: true
- obs[4]: hip1_angle — leg1 髋关节角度；reward_usable: true
- obs[5]: hip1_speed — leg1 髋关节角速度；reward_usable: true
- obs[6]: knee1_angle — leg1 膝关节角度；reward_usable: true
- obs[7]: knee1_speed — leg1 膝关节角速度；reward_usable: true
- obs[8]: leg1_contact — leg1 接地标志，1.0=contact，0.0=no contact；reward_usable: true
- obs[9]: hip2_angle — leg2 髋关节角度；reward_usable: true
- obs[10]: hip2_speed — leg2 髋关节角速度；reward_usable: true
- obs[11]: knee2_angle — leg2 膝关节角度；reward_usable: true
- obs[12]: knee2_speed — leg2 膝关节角速度；reward_usable: true
- obs[13]: leg2_contact — leg2 接地标志，1.0=contact，0.0=no contact；reward_usable: true
- obs[14]: lidar_0 — 前方地形 LIDAR 测距 0；reward_usable: true（用于地形感知，需谨慎）
- obs[15]: lidar_1 — 前方地形 LIDAR 测距 1；reward_usable: true
- obs[16]: lidar_2 — 前方地形 LIDAR 测距 2；reward_usable: true
- obs[17]: lidar_3 — 前方地形 LIDAR 测距 3；reward_usable: true
- obs[18]: lidar_4 — 前方地形 LIDAR 测距 4；reward_usable: true
- obs[19]: lidar_5 — 前方地形 LIDAR 测距 5；reward_usable: true
- obs[20]: lidar_6 — 前方地形 LIDAR 测距 6；reward_usable: true
- obs[21]: lidar_7 — 前方地形 LIDAR 测距 7；reward_usable: true
- obs[22]: lidar_8 — 前方地形 LIDAR 测距 8；reward_usable: true
- obs[23]: lidar_9 — 前方地形 LIDAR 测距 9；reward_usable: true

---

## 4. 动作空间 action_space

- type: Box
- shape: [4]
- continuous: true
- bounds: [-1.0, 1.0] per joint
- action_dim 0: hip_torque_leg1 — leg1 髋关节力矩
- action_dim 1: knee_torque_leg1 — leg1 膝关节力矩
- action_dim 2: hip_torque_leg2 — leg2 髋关节力矩
- action_dim 3: knee_torque_leg2 — leg2 膝关节力矩

---

## 5. step 与终止条件分析

### 5.1 终止模式

- success-like termination:
  - reached_end_of_terrain：到达地形末端，可能表示完成前进目标。
- failure-like termination:
  - body_fallen_over：身体摔倒，表示步态失败。
- ambiguous termination:
  - 当前返回的 `terminated` 是 `body_fallen_over or reached_end_of_terrain` 的混合结果，无法仅根据 `terminated` 区分是成功到达终点还是摔倒。
- truncation:
  - 当前 step source 返回 `False`，未标注时间截断。

### 5.2 success/failure 信号可用性

- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields:
  - 当前 step source 返回 `{}`，没有显式允许使用的 info 字段。
- forbidden_or_uncertain_info_fields:
  - 所有用户自定义 info 字段，例如 `info["success"]`、`info["failure"]`、`info["termination_reason"]`，当前均不可假设存在。
  - `original_reward` / `official_reward` 属于禁用信号。
  - 未在观察空间中声明的绝对位置、全局路程、真实能耗等字段均不可用。

---

## 6. reward 函数接口契约

函数签名：

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：

- obs
- action
- next_obs
- info 中明确允许的字段；当前 info 为空，因此不应依赖 info 任何字段
- training_progress 只有在 prompt 明确允许时才可使用；本任务未明确允许，默认不使用

禁止使用：

- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片
- 假设存在 `terminated`/`truncated` 原因标志

---

## 7. 可用于奖励函数的信号

- position:
  - 无全局 x 坐标、无绝对位置、无已行走距离。
  - 可用：lidar_0..9 作为前方地形相对距离/地形预览。
- velocity:
  - obs[2] horizontal_velocity：前进速度，主目标信号。
  - obs[3] vertical_velocity：可用于抑制不必要的竖向弹跳。
- orientation:
  - obs[0] hull_angle：躯干倾斜，可用于平衡惩罚。
  - obs[1] hull_angular_velocity：躯干旋转速度，可用于稳定性惩罚。
- contact:
  - obs[8] leg1_contact：leg1 接地标志。
  - obs[13] leg2_contact：leg2 接地标志。
  - 可用于步态周期分析、对称性判断、摔倒推断。
- action/engine:
  - action[0..3]：髋/膝力矩，可用于能量消耗惩罚或动作平滑惩罚。
- other:
  - obs[4..7]、obs[9..12]：髋膝角度和角速度，可用于步态对称性、关节幅度惩罚。
  - obs[14..23]：LIDAR 地形测量，可用于地形适应性 shaping。

---

## 8. 不确定或不可用的信号

- 绝对位置 / 全局前进距离
- 真实能量消耗 / 功率：只能通过力矩和控制量近似估计
- 显式 success/failure flag：info 为空，无法直接读取
- 终止原因：`terminated` 混合了摔倒和到达地形终点
- 脚底接触力 / 足底压力：未提供
- 地形完整高程图：仅有有限 LIDAR 测量
- 关节是否达到机械限位：未提供关节限位标志
- original_reward / official_reward：被 mask，禁止使用

---

## 9. 专家任务画像 expert_task_profile

```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: planar_bipedal_body
  actuator_type: torque-controlled_hip_and_knee_joints
  contact_structure: two_legs_with_binary_ground_contact_flags
primary_objectives:
  - walk_forward_across_uneven_terrain_as_far_as_possible
  - maintain_stable_bipedal_gait_and_avoid_falling
secondary_objectives:
  - maximize_forward_speed
  - minimize_energy_consumption
  - maintain_upright_trunk_orientation
  - reduce_unnecessary_vertical_oscillation
main_failure_risks:
  - falling_over_due_to_unstable_gait
  - low_forward_velocity_or_standing_still
  - excessive_torque_and_high_energy_consumption
  - asymmetric_gait_between_two_legs
  - excessive_torso_oscillation_or_vertical_bounce
```

---

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles

- role_id: forward_velocity_reward
  purpose: 鼓励正的 `horizontal_velocity`，推动智能体持续向前行走。
  why_required: 这是主目标，缺少前进奖励会导致智能体只学会站立、保持原地平衡或最小动作。
  usable_signals:
    - obs[2] horizontal_velocity
    - next_obs[2] horizontal_velocity
  risks:
    - 若同时奖惩 vertical_velocity，可能误伤步态。
    - 需要注意速度上限，避免奖励奇异行为。

- role_id: upright_stability_penalty
  purpose: 惩罚躯干偏离直立以及过大的躯干角速度。
  why_required: 双足行走的核心健康约束是保持躯干稳定，避免摔倒；该职责直接服务失败主风险。
  usable_signals:
    - obs[0] hull_angle
    - obs[1] hull_angular_velocity
    - next_obs[0] hull_angle
    - next_obs[1] hull_angular_velocity
  risks:
    - 过度惩罚会限制正常前倾，导致无法产生前进步态。
    - 应只惩罚过大偏角，不应惩罚小范围内的自然摆动。

### 10.2 条件职责 conditional_roles

- role_id: fall_prevention_penalty
  condition_to_use: 当可以从 `next_obs` 中较可靠地推断出摔倒状态时启用，例如躯干角过大、双足接触异常。
  purpose: 对接近摔倒或已经摔倒的状态施加额外惩罚，强化环境终止前风险。
  usable_signals:
    - next_obs[0] hull_angle
    - next_obs[1] hull_angular_velocity
    - next_obs[8] leg1_contact
    - next_obs[13] leg2_contact
  risks:
    - 当前没有显式失败 flag，阈值难以确定。
    - 可能把正常大幅度步态误判为摔倒。

- role_id: energy_penalty
  condition_to_use: 在智能体已经能稳定前进后再加入；不要在早期训练中作为主角色。
  purpose: 降低髋膝力矩幅值，近似实现最小能量消耗。
  usable_signals:
    - action[0] hip_torque_leg1
    - action[1] knee_torque_leg1
    - action[2] hip_torque_leg2
    - action[3] knee_torque_leg2
  risks:
    - 早期加入会导致智能体为了减小力矩而放弃前进。
    - 容易与前进奖励冲突。

- role_id: symmetric_gait_penalty
  condition_to_use: 当观察到明显不对称步态、某一腿长期不参与支撑/摆动时启用。
  purpose: 鼓励两条腿协调使用，防止“单腿主导”或周期性不对称。
  usable_signals:
    - obs[8] leg1_contact
    - obs[13] leg2_contact
    - obs[4..7] leg1 关节角度/速度
    - obs[9..12] leg2 关节角度/速度
  risks:
    - 若过早加入可能抑制探索出自然步态。
    - 容易和地形起伏引起的不对称真实需求冲突。

- role_id: terrain_preview_shaping
  condition_to_use: 当地形明显起伏且智能体需要提前调整步态时加入。
  purpose: 利用 LIDAR 提前感知地形，引导步态适应坡度/起伏。
  usable_signals:
    - obs[14..23] lidar_0..9
  risks:
    - LIDAR 距离与地形高度的关系复杂，直接奖励容易产生噪声。
    - 该职责不是主目标，早期不应作为主要奖励。

### 10.3 慎用/禁用职责 avoid_roles

- role_id: absolute_position_reward
  reason: 观察空间没有全局 x 坐标或已走距离，无法直接使用。
  forbidden_or_missing_signals:
    - global_x_position
    - distance_traveled

- role_id: explicit_success_failure_flag_reward
  reason: info 为空，禁止假设存在 success/failure 标志。
  forbidden_or_missing_signals:
    - info["success"]
    - info["failure"]
    - info["termination_reason"]

- role_id: true_energy_consumption_reward
  reason: 没有真实功率或能耗信号，只能用动作力矩近似，不能视为真实能耗。
  forbidden_or_missing_signals:
    - true_power
    - energy_consumption

- role_id: original_reward_imitation
  reason: official reward 已被 mask，禁止复现或模仿。
  forbidden_or_missing_signals:
    - original_reward
    - official_reward

---

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_velocity_reward | obs[2], next_obs[2] | global_x_position, distance_traveled | dense_state_signal, bounded_signal | 主前进奖励，应只使用水平速度，不应使用垂直速度作为前进信号 |
| upright_stability_penalty | obs[0], obs[1], next_obs[0], next_obs[1] | explicit_fall_flag | quadratic_penalty, bounded_signal | 惩罚较大的躯干偏角和角速度，需保留小幅自然摆动 |
| fall_prevention_penalty | next_obs[0], next_obs[1], next_obs[8], next_obs[13] | fall_reason_flag | threshold_penalty, quadratic_penalty | 条件使用；没有显式摔倒标志，阈值需谨慎 |
| energy_penalty | action[0..3] | true_energy_consumption | quadratic_penalty, bounded_signal | 条件使用；只在稳定前进后加入 |
| symmetric_gait_penalty | obs[8], obs[13], obs[4..7], obs[9..12] | foot_force, phase_label | phase_divergence, quadratic_penalty | 条件使用；地形本身可能导致不对称，需注意 |
| terrain_preview_shaping | obs[14..23] | full_elevation_map | bounded_signal, smoothness_operator | 条件使用；LIDAR 更适合作为辅助信号，避免主奖励依赖 |

---

## 12. 初始训练后应观察的 failure modes

| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 智能体学会原地站立，不前进或前进极慢 | horizontal_velocity 长期接近 0，前进奖励停止增长 | 强化 forward_velocity_reward，增加最小前进速度门槛 |
| 频繁摔倒，episode 过早结束 | hull_angle 频繁超过安全范围，角速度大，终止早 | 增强 upright_stability_penalty，必要时加入 fall_prevention_penalty |
| 力矩过大，能量消耗高 | action[0..3] 常接近 ±1.0，峰值力矩高 | 在稳定前进后引入 energy_penalty |
| 步态不对称，某一腿不参与支撑 | 两个 leg_contact 比例严重失衡，关节角度分布不对称 | 条件加入 symmetric_gait_penalty |
| 躯干上下弹跳严重 | vertical_velocity 高频大幅波动 | 对 vertical_velocity 幅值施加轻微惩罚 |
| 地形起伏导致步态不适应 | LIDAR 读数变化大，但前进速度低或摔倒多 | 条件使用 terrain_preview_shaping 或观察 LIDAR 与摔倒相关性 |
| 早期奖励使得腿部僵硬不动 | 关节角度变化极小，动作幅值接近 0 | 减少能量惩罚权重，先保证前进奖励 |
