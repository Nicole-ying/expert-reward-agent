# 匿名环境理解卡片

## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：没有明确的“到达目标点”或“抓取物体”要求，核心是持续前进与平衡维持。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务核心是双足身体在连续地形上持续前进，属于典型的连续控制运动任务。虽然包含能量消耗最小化，但前进距离和速度是主要目标，能量消耗是附属优化项，不构成多目标冲突。

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
- dtype: float32 (推测)
- action[0]: hip_torque_leg1，腿1髋关节施加的扭矩，范围[-1.0, 1.0]
- action[1]: knee_torque_leg1，腿1膝关节施加的扭矩，范围[-1.0, 1.0]
- action[2]: hip_torque_leg2，腿2髋关节施加的扭矩，范围[-1.0, 1.0]
- action[3]: knee_torque_leg2，腿2膝关节施加的扭矩，范围[-1.0, 1.0]

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain — 成功到达地形终点，属于成功终止
- failure-like termination: body_fallen_over — 身体摔倒，属于失败终止
- ambiguous termination: 无
- truncation: 无（step返回truncated=False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info为空字典{}，无显式成功标志）
- explicit_failure_flag_available: false（info为空字典{}，无显式失败标志）
- allowed_info_fields: 无（info为空字典）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前步的观测（24维向量）
- action: 当前步的动作（4维向量）
- next_obs: 下一步的观测（24维向量）
- info: 空字典，不可用
- training_progress: 仅当prompt明确允许时才使用

禁止使用：
- original_reward: 官方奖励已屏蔽，禁止使用
- official_reward: 禁止使用
- 未声明的info字段：info为空字典
- 未声明的obs切片：仅允许使用上述24维观测

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，可通过horizontal_velocity积分或LIDAR间接推断
- velocity: horizontal_velocity (obs[2]), vertical_velocity (obs[3]), hull_angular_velocity (obs[1])
- orientation: hull_angle (obs[0])
- contact: leg1_contact (obs[8]), leg2_contact (obs[13])
- action/engine: action[0..3] 扭矩值
- other: lidar_0..9 (obs[14..23]) 地形距离测量

## 8. 不确定或不可用的信号
- 绝对位置(x, y坐标)：不可用，无直接观测
- 地形高度/坡度：不可用，无直接观测
- 能量消耗：不可用，无直接观测
- 步态周期/相位：不可用，无直接观测
- 成功/失败标志：不可用，info为空字典
- 步数/时间：不可用，无直接观测

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal (two-legged)
  actuator_type: torque-controlled hip and knee joints (4 actuators)
  contact_structure: two point-feet with binary ground contact flags
primary_objectives:
  - maximize forward horizontal velocity
  - maximize distance traveled (survive until terrain end)
  - maintain body balance (avoid falling)
secondary_objectives:
  - minimize energy consumption (action torque magnitude)
  - maintain stable gait pattern
main_failure_risks:
  - body falling over (hull_angle exceeds stability threshold)
  - losing ground contact coordination between legs
  - inefficient gait leading to early termination
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励身体向前移动，最大化前进速度
  why_required: 任务核心目标是尽可能远、尽可能快地向前行走
  usable_signals: [horizontal_velocity (obs[2])]
  risks: 可能鼓励身体前倾或跳跃，需配合平衡约束

- role_id: balance_maintenance
  purpose: 保持身体直立不摔倒，避免终止
  why_required: 摔倒直接终止任务，是主要失败模式
  usable_signals: [hull_angle (obs[0]), hull_angular_velocity (obs[1])]
  risks: 过度约束可能导致动作僵硬，影响前进速度

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 当基础前进和平衡已建立，需要优化能耗时加入
  usable_signals: [action[0..3] 扭矩值]
  risks: 过早加入可能阻碍探索有效步态

- role_id: gait_smoothness
  condition_to_use: 当步态不稳定或关节运动剧烈时加入
  usable_signals: [hip1_speed, knee1_speed, hip2_speed, knee2_speed (obs[5,7,10,12])]
  risks: 可能限制必要的快速关节运动

- role_id: terrain_adaptation
  condition_to_use: 当LIDAR显示前方地形变化剧烈时加入
  usable_signals: [lidar_0..9 (obs[14..23])]
  risks: 信号维度高，需要降维或特征提取

### 10.3 慎用/禁用职责 avoid_roles
- role_id: contact_rhythm
  reason: 无步态相位或接触时序信号，无法直接实现
  forbidden_or_missing_signals: [步态周期、接触时序]

- role_id: height_maintenance
  reason: 无身体高度直接观测，vertical_velocity不提供绝对高度
  forbidden_or_missing_signals: [绝对高度]

- role_id: distance_traveled
  reason: 无绝对位置信号，无法直接计算前进距离
  forbidden_or_missing_signals: [x坐标]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | 无 | dense_state_signal, linear_scaling | 直接使用水平速度作为正向奖励 |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | 无 | bounded_signal, quadratic_penalty | 角度偏离0度时惩罚，角速度作为阻尼项 |
| energy_efficiency | action[0..3] | 无 | quadratic_penalty, l2_norm | 对动作扭矩的L2范数进行惩罚 |
| gait_smoothness | hip1_speed, knee1_speed, hip2_speed, knee2_speed (obs[5,7,10,12]) | 无 | quadratic_penalty, jerk_penalty | 对关节角速度变化率进行惩罚 |
| terrain_adaptation | lidar_0..9 (obs[14..23]) | 无 | feature_extraction, attention | 需要降维或学习地形特征 |
| contact_rhythm | leg1_contact, leg2_contact (obs[8,13]) | 步态周期、相位 | 无 | 信号不足，无法实现 |
| height_maintenance | vertical_velocity (obs[3]) | 绝对高度 | 无 | 信号不足，无法实现 |
| distance_traveled | 无 | x坐标 | 无 | 信号缺失，无法实现 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 身体前倾摔倒 | hull_angle持续增大，horizontal_velocity突然下降 | 加强balance_maintenance惩罚权重，增加角度阈值约束 |
| 原地跳跃不前进 | vertical_velocity波动大，horizontal_velocity接近0 | 增加forward_progress奖励权重，惩罚垂直运动 |
| 单腿跛行 | leg1_contact和leg2_contact交替不规律 | 引入gait_smoothness或contact_rhythm约束 |
| 动作幅度过大 | action值接近±1，关节速度剧烈波动 | 加入energy_efficiency惩罚，平滑动作 |
| 地形适应失败 | lidar显示地形变化时摔倒 | 引入terrain_adaptation，学习地形特征 |
| 过早终止 | 平均episode长度短，摔倒率高 | 检查balance_maintenance是否足够，调整角度惩罚阈值 |