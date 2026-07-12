# 匿名环境理解卡片

## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：任务不是到达某个特定目标点，也不是单纯保持平衡，而是持续前进与能耗之间的权衡。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 核心目标是持续前进通过地形，附属有能耗和平滑要求，没有明确的单一目标点到达，也不是单纯的生存平衡任务。

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
- success-like termination: reached_end_of_terrain（成功到达地形终点）
- failure-like termination: body_fallen_over（摔倒，任务失败）
- ambiguous termination: 无
- truncation: 无（truncated=False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（terminated=True 但无法区分是成功还是失败）
- explicit_failure_flag_available: false（同上）
- allowed_info_fields: 无（info 字典为空 {}）
- forbidden_or_uncertain_info_fields: 无（但无法从 terminated 中区分成功/失败）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测）
- action（当前动作）
- next_obs（下一时刻观测）
- info 中明确允许的字段（当前无）
- training_progress 只有 prompt 明确允许时才用

禁止使用：
- original_reward（官方奖励被屏蔽）
- official_reward（任何形式）
- 未声明的 info 字段
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，但可通过 LIDAR 间接推断前进距离
- velocity: horizontal_velocity (obs[2]), vertical_velocity (obs[3])
- orientation: hull_angle (obs[0]), hull_angular_velocity (obs[1])
- contact: leg1_contact (obs[8]), leg2_contact (obs[13])
- action/engine: action[0..3]（扭矩值，可用于能耗惩罚）
- other: 关节角度/速度 (obs[4..7], obs[9..12])，LIDAR (obs[14..23])

## 8. 不确定或不可用的信号
- 绝对位置/位移：不可用，无 x/y 坐标观测
- 成功/失败标志：不可用，terminated 无法区分成功/失败
- 地形高度/坡度：不可用，但 LIDAR 可提供前方地形距离信息
- 能量消耗：不可用，但可通过 action 扭矩平方和近似
- 步态周期/相位：不可用，需从关节角度和接触信号推断

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal (two-legged)
  actuator_type: torque-controlled hip and knee joints (4 actuators)
  contact_structure: two feet with binary ground contact flags
primary_objectives:
  - maximize forward horizontal velocity
  - maximize distance traveled (reach end of terrain)
  - maintain balance (avoid falling)
secondary_objectives:
  - minimize energy consumption (action torque magnitude)
  - maintain stable gait (smooth joint movements)
main_failure_risks:
  - falling over (body_fallen_over)
  - getting stuck on uneven terrain
  - inefficient gait leading to early termination
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励身体向前移动，最大化水平速度
  why_required: 核心任务目标是尽可能远、尽可能快地向前行走
  usable_signals: [horizontal_velocity (obs[2])]
  risks: 可能鼓励摔倒式前进，需配合平衡约束

- role_id: balance_maintenance
  purpose: 保持身体直立，防止摔倒
  why_required: 摔倒直接终止任务，是主要失败模式
  usable_signals: [hull_angle (obs[0]), hull_angular_velocity (obs[1])]
  risks: 过度惩罚可能导致动作僵硬，影响前进速度

- role_id: energy_efficiency
  purpose: 最小化能量消耗，惩罚过大扭矩
  why_required: 任务明确要求最小化能量消耗
  usable_signals: [action[0..3]]
  risks: 过度惩罚可能导致动作幅度过小，无法前进

### 10.2 条件职责 conditional_roles
- role_id: gait_regularity
  condition_to_use: 当观察到步态不稳定或效率低下时加入
  usable_signals: [hip_angles, knee_angles, contact_flags]
  risks: 可能限制探索，过早约束步态模式

- role_id: terrain_adaptation
  condition_to_use: 当 LIDAR 显示前方地形变化剧烈时
  usable_signals: [lidar_0..9 (obs[14..23])]
  risks: 需要复杂的地形理解，可能增加训练难度

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_reward
  reason: 无法从 terminated 中区分成功（到达终点）和失败（摔倒），无法直接使用
  forbidden_or_missing_signals: [success_flag, failure_flag]

- role_id: distance_traveled
  reason: 无绝对位置观测，无法直接计算前进距离
  forbidden_or_missing_signals: [x_position, displacement]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | 无 | dense_state_signal, linear_scaling | 直接使用水平速度作为正向奖励 |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | 无 | bounded_signal, quadratic_penalty | 角度偏离竖直方向越大惩罚越大 |
| energy_efficiency | action[0..3] | 无 | quadratic_penalty, l2_norm | 动作扭矩平方和作为能耗惩罚 |
| gait_regularity | hip_angles, knee_angles, contact_flags | 步态相位/周期 | periodic_signal, contact_pattern | 需要设计步态周期检测 |
| terrain_adaptation | lidar_0..9 (obs[14..23]) | 地形高度/坡度 | distance_to_obstacle, terrain_awareness | 根据前方地形调整步态 |
| explicit_success_reward | 无 | success_flag, failure_flag | 无 | 不可用，信号缺失 |
| distance_traveled | 无 | x_position, displacement | 无 | 不可用，信号缺失 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 过早摔倒 | 训练早期 terminated 频繁，horizontal_velocity 低 | 增加 balance_maintenance 权重，降低 forward_progress 权重 |
| 原地踏步 | horizontal_velocity 接近 0，但未摔倒 | 增加 forward_progress 权重，检查 balance_maintenance 是否过强 |
| 能耗过高 | action 值接近边界，关节速度大 | 增加 energy_efficiency 权重，或使用动作平滑惩罚 |
| 步态不对称 | 一条腿接触时间远多于另一条 | 引入 gait_regularity 职责，鼓励对称步态 |
| 地形适应不良 | 在 LIDAR 显示地形变化时摔倒 | 引入 terrain_adaptation 职责，根据 LIDAR 调整步态 |
| 动作抖动 | 相邻时间步 action 变化剧烈 | 增加动作平滑惩罚（action difference penalty） |