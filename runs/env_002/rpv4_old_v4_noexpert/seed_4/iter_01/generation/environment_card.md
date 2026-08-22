# 匿名环境理解卡片

## 1. 任务目标

本环境要求一个平面双足身体在不平坦地形上向前行走，尽可能走得远、走得快，同时降低能量消耗。智能体需要协调两条腿的髋关节与膝关节力矩，形成稳定的双足步态；一旦身体摔倒，回合立即终止。

- 主目标：持续向前通过不平坦地形，最大化前进距离或前进速度。
- 次目标：降低能量消耗，并保持身体直立/稳定、避免摔倒。
- 不应混淆：这不是一个静态到达目标点任务，也不是只要不摔倒即可的纯平衡任务；能量消耗是次要目标，不应取代前进与稳定。

## 2. 任务类型选择

selected_route_id: locomotion_continuous_control

confidence: high

reason: 环境核心是让双足身体通过关节力矩持续前进，通过地形；动作空间为连续关节力矩，观察空间包含身体线速度、躯干姿态、关节状态、触地与地形感知。摔倒终止、到达地形尽头终止。虽然存在能量最小化目标，但它是附属优化，不是与前进目标权重相当且冲突的多目标核心，因此不选 multi_objective_task。

## 3. 观察空间 observation_space

- type: Box
- shape: [24]
- dtype: 未在匿名说明中明确给出；连续 Box 通常为 float32，但本卡不将 float32 当作已确认事实
- obs[0]: hull_angle，主躯干相对竖直方向的倾角，reward_usable: true
- obs[1]: hull_angular_velocity，主躯干角速度，reward_usable: true
- obs[2]: horizontal_velocity，前向/后向线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿 1 髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿 1 髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿 1 膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿 1 膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿 1 地面接触标志，1.0=接触，0.0=未接触，reward_usable: true
- obs[9]: hip2_angle，腿 2 髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿 2 髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿 2 膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿 2 膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿 2 地面接触标志，1.0=接触，0.0=未接触，reward_usable: true
- obs[14]: lidar_0，第 1 条前方 LIDAR 测距距离，reward_usable: true
- obs[15]: lidar_1，第 2 条前方 LIDAR 测距距离，reward_usable: true
- obs[16]: lidar_2，第 3 条前方 LIDAR 测距距离，reward_usable: true
- obs[17]: lidar_3，第 4 条前方 LIDAR 测距距离，reward_usable: true
- obs[18]: lidar_4，第 5 条前方 LIDAR 测距距离，reward_usable: true
- obs[19]: lidar_5，第 6 条前方 LIDAR 测距距离，reward_usable: true
- obs[20]: lidar_6，第 7 条前方 LIDAR 测距距离，reward_usable: true
- obs[21]: lidar_7，第 8 条前方 LIDAR 测距距离，reward_usable: true
- obs[22]: lidar_8，第 9 条前方 LIDAR 测距距离，reward_usable: true
- obs[23]: lidar_9，第 10 条前方 LIDAR 测距距离，reward_usable: true

## 4. 动作空间 action_space

- type: Box
- shape 或 n: [4]，连续动作
- action 0: hip_torque_leg1，施加到腿 1 髋关节的力矩，范围 [-1.0, 1.0]
- action 1: knee_torque_leg1，施加到腿 1 膝关节的力矩，范围 [-1.0, 1.0]
- action 2: hip_torque_leg2，施加到腿 2 髋关节的力矩，范围 [-1.0, 1.0]
- action 3: knee_torque_leg2，施加到腿 2 膝关节的力矩，范围 [-1.0, 1.0]

控制类型为连续关节力矩控制，不是位置/速度目标控制。

## 5. step 与终止条件分析

### 5.1 终止模式

- success-like termination:
  - reached_end_of_terrain：到达地形尽头，可能视为完成任务。
- failure-like termination:
  - body_fallen_over：身体摔倒，回合失败中止。
- ambiguous termination:
  - reached_end_of_terrain 可作为成功，但 step 返回的 info 为空，不能直接确认成功原因；只能从环境事件定义判断。
- truncation:
  - masked step source 中 truncated 恒为 False，未观察到显式时间截断或步数截断。

### 5.2 success/failure 信号可用性

- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无；本环境 info 返回空字典 `{}`，未声明任何信息字段
- forbidden_or_uncertain_info_fields: 任何未声明的 termination_reason、success、failure、timeout 等字段都不可用于 reward

## 6. reward 函数接口契约

函数签名：

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：

- obs
- action
- next_obs
- info 中明确允许的字段；根据当前 task spec 与 step source，尚无任何明确允许的 info 字段
- training_progress 仅在 prompt 明确允许时使用；本环境未明确允许，建议不使用

禁止使用：

- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片或任何外部状态

## 7. 可用于奖励函数的信号

- position:
  - 没有全局 x/y 坐标或全局身体位置。
  - 关节角度：hip1_angle, knee1_angle, hip2_angle, knee2_angle。
  - LIDAR 距离可提供前方地形相对距离，但不是身体全局位置。
- velocity:
  - horizontal_velocity：前向速度，契合主目标。
  - vertical_velocity：可辅助判断弹跳或异常上下运动。
  - hull_angular_velocity：躯干角速度，辅助摔倒前兆。
  - hip1_speed, knee1_speed, hip2_speed, knee2_speed：关节角速度。
- orientation:
  - hull_angle：主躯干倾角，与摔倒/直立高度相关。
- contact:
  - leg1_contact, leg2_contact：每条腿是否触地，可用于步态触地节奏。
- action/engine:
  - action 0..3：各关节力矩，可用于能量消耗近似。
- other:
  - lidar_0..lidar_9：前方地形测距，可用于地形感知辅助，但需要谨慎使用其物理含义。

## 8. 不确定或不可用的信号

- 真实能量消耗：没有电流、电压、功率或机械功字段，只能用动作力矩的模/平方等近似。
- 全局前进距离/里程计：观察中无 x,y 位置，不能直接计算精确前进位移。
- 显式摔倒标志：观察中没有直接的 body_fallen_over 标志，只能通过 hull_angle、hull_angular_velocity 等间接推断。
- LIDAR 对应的真实地形高度或障碍物成本：只有距离测量，缺少地面真实高度图或可行走代价定义。
- 接触力/地面反力：观察中只有 binary contact flag，没有法向力或摩擦幅值。
- 地形进度/完成比例：没有全局地形长度或剩余距离字段。

## 9. 专家任务画像 expert_task_profile

```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: planar_bipedal_body
  actuator_type: torque_controlled_hip_knee
  contact_structure: two_legs_discrete_contact_flags
primary_objectives:
  - walk_forward_as_far_and_fast_as_possible
  - cross_uneven_terrain_without_falling
secondary_objectives:
  - minimize_energy_consumption
  - maintain_upright_and_stable_gait
main_failure_risks:
  - falling_over_and_terminating
  - learning_jittery_or_in_place_gait
  - excessive_joint_torque_energy_cost
  - ignoring_terrain_perception_and_tripping
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles

- role_id: forward_progress
  purpose: 鼓励身体向前移动，形成持续前进的双足步态。
  why_required: 这是本任务主目标；没有前向速度信号，智能体可能学会原地平衡或抖动而不前进。
  usable_signals:
    - obs[2] horizontal_velocity
    - next_obs[2] horizontal_velocity
  risks:
    - 仅奖励前向速度可能诱导身体过度前倾或利用跳跃。
    - 需要与稳定性职责配合，避免只优化速度导致摔倒。

- role_id: upright_stability
  purpose: 惩罚主躯干过度倾斜或过快旋转，保持双足步态稳定。
  why_required: body_fallen_over 会终止回合；没有稳定性约束时，前向速度和力矩目标会迅速破坏平衡。
  usable_signals:
    - obs[0] hull_angle
    - obs[1] hull_angular_velocity
    - next_obs[0] hull_angle
    - next_obs[1] hull_angular_velocity
  risks:
    - 过强的直立约束可能限制跑步或前倾步态，使步态僵硬。
    - 早期过度惩罚会导致动作幅度过小，难以探索有效步态。

### 10.2 条件职责 conditional_roles

- role_id: energy_efficiency_penalty
  condition_to_use: 当智能体已出现有效前进但仍伴随高力矩/高关节速度时，或训练中后期需要降低能耗时加入。
  usable_signals:
    - action[0..3] 关节力矩
    - 可选 joint speeds：obs[5], obs[7], obs[10], obs[12]
  risks:
    - 训练早期加入能量惩罚会抑制探索，智能体可能只保持小动作、不前进。
    - 需要保证能量项不压过 forward_progress；否则会导致过慢甚至后退。

### 10.3 慎用/禁用职责 avoid_roles

- role_id: global_position_error
  reason: 观察空间没有全局 x/y 坐标或里程计，无法可靠计算精确位置误差。
  forbidden_or_missing_signals:
    - global_x_position
    - global_y_position
    - odometry_progress

- role_id: official_reward_reconstruction
  reason: 环境理解阶段不重建官方 reward；compute_reward 禁止使用 original_reward 或 official_reward。
  forbidden_or_missing_signals:
    - original_reward
    - official_reward

- role_id: terrain_height_error_shaping_from_lidar
  reason: LIDAR 只给出前方距离而非真实地形高度或可通行代价，直接构造地形高度误差容易引入错误先验。
  forbidden_or_missing_signals:
    - terrain_heightmap
    - obstacle_cost_map
    - exact_terrain_elevation

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | obs[2], next_obs[2] | global_x_position, odometry_progress | dense_state_signal, linear_surplus | 建议与 upright_stability 联合使用 |
| upright_stability | obs[0], obs[1], next_obs[0], next_obs[1] | explicit_fallen_flag | bounded_signal, quadratic_penalty | 关注 hull_angle 与 hull_angular_velocity 的异常幅值 |
| energy_efficiency_penalty | action[0..3], optionally obs[5], obs[7], obs[10], obs[12] | true_mechanical_power, motor_current | quadratic_penalty, bounded_penalty | 仅作为 conditional role，不覆盖主目标 |
| global_position_error | none | global_x/y, odometry_progress | prohibited | 信号缺失，不建议使用 |
| official_reward_reconstruction | none | original_reward, official_reward | prohibited | 禁止使用官方奖励 |
| terrain_height_error_shaping_from_lidar | lidar_0..9 | terrain_heightmap, obstacle_cost_map | prohibited | LIDAR 语义不足以支撑高度误差 |

## 12. 初始训练后应观察的 failure modes

| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 原地抖动或原地摆腿，不前进 | horizontal_velocity 接近 0，但 joint speeds 或 action 幅值较高 | 减小能量 penalty，加入更强的 forward_progress 信号 |
| 频繁摔倒，回合很快结束 | hull_angle 或 hull_angular_velocity 经常超过较大范围，回合长度短 | 加强 upright_stability 职责，或降低前向速度奖励的激进程度 |
| 动作过小导致步态过于保守 | action 幅度长期接近 0，horizontal_velocity 很低 | 减小能量惩罚强度，或增加探索噪声 |
| 前倾过快、利用不稳定姿态冲速度 | horizontal_velocity 高但 hull_angle 长期偏离直立中心 | 调整 upright_stability 的惩罚阈值，使其允许适当倾角但避免极端 |
| 忽视地形感知，遇到坡/障碍后摇晃 | lidar 信号变化与 hull_angle 突变同时出现 | 如使用地形相关辅助 reward，应基于距离变化而非绝对高度；否则不要对 lidar 直接奖惩 |
| 能量项过强导致不学习前进 | 回合表现为小力矩、几乎不移动 | 将 energy_efficiency_penalty 改为 conditional，或根据最低前进速度动态启用 |