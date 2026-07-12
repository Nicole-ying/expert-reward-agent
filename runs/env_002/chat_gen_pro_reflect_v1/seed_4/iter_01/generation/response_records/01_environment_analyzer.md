# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：虽然能量消耗是优化项，但核心驱动力是前进速度和距离，能量是附属约束；任务不是单纯保持平衡（生存），也不是精确导航到某个点，而是持续前进的 locomotion 任务。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务核心是双足身体在连续地形上持续前进，动作空间是连续关节力矩控制，观察包含地形感知（LIDAR），终止条件包括摔倒和到达地形末端。这完全符合 locomotion_continuous_control 的定义：核心是持续前进通过地形，附属有能耗/平滑要求。

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
- obs[8]: leg1_contact，腿1地面接触标志（1.0=接触，0.0=无接触），reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2地面接触标志（1.0=接触，0.0=无接触），reward_usable: true
- obs[14..23]: lidar_0..9，10个LIDAR测距仪沿前方地形的距离测量值，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- dtype: float32
- bounds: [-1.0, 1.0] per dimension
- action[0]: hip_torque_leg1，腿1髋关节施加的力矩
- action[1]: knee_torque_leg1，腿1膝关节施加的力矩
- action[2]: hip_torque_leg2，腿2髋关节施加的力矩
- action[3]: knee_torque_leg2，腿2膝关节施加的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain — 成功到达地形末端，属于成功终止。
- failure-like termination: body_fallen_over — 身体摔倒，属于失败终止。
- ambiguous termination: 无。
- truncation: 无显式截断（step 返回 False 作为 truncation 标志）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false — info 字典为空 {}，没有显式 success 标志。
- explicit_failure_flag_available: false — info 字典为空 {}，没有显式 failure 标志。
- allowed_info_fields: 无（info 为空）。
- forbidden_or_uncertain_info_fields: 无（info 为空）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前步的观察（24维）
- action: 当前步的动作（4维）
- next_obs: 下一步的观察（24维）
- info: 当前步的 info 字典（当前为空，但未来可能扩展）
- training_progress: 仅当 prompt 明确允许时才使用

禁止使用：
- original_reward: 官方奖励被屏蔽，禁止使用
- 未声明的 info 字段
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，但可通过 horizontal_velocity 积分或 LIDAR 间接推断
- velocity: horizontal_velocity (obs[2])，vertical_velocity (obs[3])
- orientation: hull_angle (obs[0])，hull_angular_velocity (obs[1])
- contact: leg1_contact (obs[8])，leg2_contact (obs[13])
- action/engine: action 本身（4维力矩），可计算动作幅度、变化率
- other: LIDAR 距离 (obs[14..23])，关节角度和速度 (obs[4..7], obs[9..12])

## 8. 不确定或不可用的信号
- 绝对位置/位移：没有直接的位置观测，无法直接计算前进距离
- 地形高度/坡度：LIDAR 提供前方距离，但无直接地形高度
- 能量消耗：没有直接的能耗测量，只能通过 action 幅度间接估计
- 步态周期/相位：没有显式的步态相位信号
- 成功/失败标志：info 为空，无法直接使用 success/failure 标志

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal (two-legged)
  actuator_type: torque-controlled hip and knee joints (4 actuators)
  contact_structure: two point/line feet with binary ground contact flags
primary_objectives:
  - maximize forward horizontal velocity
  - maintain stable upright posture (avoid falling)
  - traverse uneven terrain (adapt to LIDAR readings)
secondary_objectives:
  - minimize energy consumption (action magnitude)
  - maintain smooth, efficient gait
main_failure_risks:
  - body falling over (hull_angle exceeds stability threshold)
  - getting stuck on terrain obstacles
  - inefficient gait leading to high energy cost
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励身体向前移动，最大化前进速度
  why_required: 任务核心目标是尽可能远、尽可能快地向前行走
  usable_signals: [horizontal_velocity (obs[2])]
  risks: 可能鼓励身体前倾摔倒，需要与平衡职责配合

- role_id: balance_maintenance
  purpose: 保持身体直立不摔倒，避免终止
  why_required: 摔倒直接终止任务，是核心约束
  usable_signals: [hull_angle (obs[0]), hull_angular_velocity (obs[1])]
  risks: 过度惩罚角度偏差可能导致身体僵硬，影响前进

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 当基础前进和平衡已建立，需要优化能耗时加入
  usable_signals: [action (4维力矩)]
  risks: 过度惩罚动作幅度可能导致步态无力，无法前进

- role_id: terrain_adaptation
  condition_to_use: 当 LIDAR 显示前方有显著地形变化时
  usable_signals: [lidar_0..9 (obs[14..23])]
  risks: 复杂地形信号可能引入噪声，需要谨慎设计

- role_id: gait_smoothness
  condition_to_use: 当步态出现抖动或不连续时
  usable_signals: [action (4维力矩), joint angles/speeds (obs[4..7], obs[9..12])]
  risks: 过度平滑可能降低响应速度

### 10.3 慎用/禁用职责 avoid_roles
- role_id: contact_phase_control
  reason: 没有显式的步态相位或接触时序信号，只有二进制接触标志，难以设计有效的相位奖励
  forbidden_or_missing_signals: [步态相位、接触时序]

- role_id: absolute_position_reaching
  reason: 没有绝对位置观测，无法计算到目标点的距离
  forbidden_or_missing_signals: [x/y/z position]

- role_id: terrain_height_avoidance
  reason: LIDAR 提供前方距离，但无直接地形高度，无法精确惩罚地形碰撞
  forbidden_or_missing_signals: [地形高度]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | 无 | dense_state_signal, bounded_signal | 直接使用水平速度作为正向奖励 |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | 无 | quadratic_penalty, bounded_signal | 对角度偏差和角速度施加惩罚 |
| energy_efficiency | action (4维) | 无 | quadratic_penalty, l2_norm | 对动作幅度施加二次惩罚 |
| terrain_adaptation | lidar_0..9 (obs[14..23]) | 地形高度 | dense_state_signal, threshold_signal | 根据 LIDAR 读数调整步态 |
| gait_smoothness | action, joint angles/speeds | 步态相位 | temporal_difference, l2_norm | 对动作变化率施加惩罚 |
| contact_phase_control | leg1_contact, leg2_contact | 步态相位、接触时序 | 无 | 信号不足，禁用 |
| absolute_position_reaching | 无 | x/y/z position | 无 | 信号缺失，禁用 |
| terrain_height_avoidance | lidar_0..9 | 地形高度 | 无 | 信号不足，禁用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 身体前倾摔倒 | hull_angle 持续增大，horizontal_velocity 突然归零 | 增加 balance_maintenance 惩罚权重，或引入 hull_angle 的硬边界 |
| 步态僵硬无法前进 | horizontal_velocity 接近零，action 幅度小 | 降低 energy_efficiency 权重，或增加 forward_progress 奖励 |
| 步态抖动不稳定 | hull_angular_velocity 波动大，关节速度高频振荡 | 引入 gait_smoothness 职责，对动作变化率施加惩罚 |
| 地形适应失败 | LIDAR 读数变化大时摔倒或停滞 | 引入 terrain_adaptation 职责，根据 LIDAR 调整步态 |
| 能量消耗过高 | action 幅度大但前进速度低 | 增加 energy_efficiency 权重，或优化步态周期 |
| 单腿过度依赖 | 一条腿接触标志长期为0，另一条腿承担全部支撑 | 引入对称性约束或接触平衡奖励 |
