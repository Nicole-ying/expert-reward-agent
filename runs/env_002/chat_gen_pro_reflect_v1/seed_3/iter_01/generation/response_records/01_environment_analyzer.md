# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务的核心目标是控制一个双足机器人，在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。机器人需要协调两条腿的髋关节和膝关节，产生稳定的双足步态。摔倒会导致回合终止。附属目标（省能量、速度快）是优化方向，但主目标是“持续前进通过地形”。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务描述明确要求“walk forward across uneven terrain as far and as fast as possible”，核心是持续前进通过地形，属于典型的连续控制 locomotion 任务。附属的能耗最小化是优化目标，不是独立核心目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推测)
- obs[0]: hull_angle，主体相对于竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，主体角速度，reward_usable: true
- obs[2]: horizontal_velocity，前后线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1地面接触标志(1.0=接触, 0.0=无接触)，reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2地面接触标志(1.0=接触, 0.0=无接触)，reward_usable: true
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
- success-like termination: reached_end_of_terrain（到达地形终点，可视为成功完成）
- failure-like termination: body_fallen_over（摔倒，明确失败）
- ambiguous termination: 无
- truncation: 无（truncated 始终为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空 {}，无 success 标志）
- explicit_failure_flag_available: false（info 字典为空 {}，无 failure 标志）
- allowed_info_fields: 无（info 始终为 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前步的观测（24维向量）
- action: 当前步的动作（4维向量）
- next_obs: 下一步的观测（24维向量）
- info: 当前步的 info 字典（但本环境中始终为空 {}）
- training_progress: 仅当 prompt 明确允许时才使用

禁止使用：
- original_reward: 官方奖励已屏蔽，禁止使用
- 未声明的 info 字段: info 始终为空，无可用字段
- 未声明的 obs 切片: 仅允许使用上述 24 维观测

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，可通过 horizontal_velocity 积分或 LIDAR 间接推断
- velocity: horizontal_velocity (obs[2])，vertical_velocity (obs[3])
- orientation: hull_angle (obs[0])，hull_angular_velocity (obs[1])
- contact: leg1_contact (obs[8])，leg2_contact (obs[13])
- action/engine: action 向量（4维扭矩），可用于能耗惩罚
- other: LIDAR 距离测量 (obs[14..23])，关节角度和速度 (obs[4..7], obs[9..12])

## 8. 不确定或不可用的信号
- 绝对位置/位移: 无直接 x/y 位置观测，无法直接计算前进距离
- 地形高度/坡度: 无直接观测，仅通过 LIDAR 间接感知
- 能耗/功率: 无直接观测，需通过 action 扭矩和关节速度估算
- 成功/失败标志: info 为空，无法区分成功终止和失败终止
- 时间步/步数: 无显式时间步计数
- 地形终点距离: 无直接观测，仅通过 LIDAR 和终止条件间接推断

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal_robot_with_hips_and_knees
  actuator_type: torque_actuated_joints
  contact_structure: two_point_feet_ground_contact
primary_objectives:
  - 在崎岖地形上持续向前行走
  - 保持身体平衡不摔倒
secondary_objectives:
  - 最大化前进速度
  - 最小化能量消耗（动作扭矩）
  - 适应不规则地形（LIDAR 感知）
main_failure_risks:
  - 身体倾斜过大导致摔倒
  - 步态不稳定导致无法持续前进
  - 关节角度超限或动作过大导致失控
  - 地形突变导致接触失效
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励机器人持续向前移动
  why_required: 任务核心目标是向前行走，必须提供前进激励
  usable_signals: [horizontal_velocity (obs[2])]
  risks: 可能鼓励机器人向前倾倒而非稳定行走

- role_id: balance_maintenance
  purpose: 保持身体直立，防止摔倒
  why_required: 摔倒直接终止回合，必须避免
  usable_signals: [hull_angle (obs[0]), hull_angular_velocity (obs[1])]
  risks: 过度惩罚倾斜可能导致机器人不敢迈步

- role_id: gait_cycle_encouragement
  purpose: 鼓励双腿交替接触地面，形成稳定步态
  why_required: 双足行走需要交替支撑，单腿接触或双腿同时离地会导致不稳定
  usable_signals: [leg1_contact (obs[8]), leg2_contact (obs[13])]
  risks: 可能鼓励原地踏步而非前进

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 当基础行走能力已建立，需要优化能耗时
  usable_signals: [action (4维扭矩向量)]
  risks: 过早加入可能导致机器人不敢用力，无法前进

- role_id: terrain_adaptation
  condition_to_use: 当 LIDAR 观测显示前方地形变化明显时
  usable_signals: [lidar_0..9 (obs[14..23])]
  risks: 信号维度高，需要合理降维或特征提取

- role_id: smooth_motion
  condition_to_use: 当动作抖动或关节速度变化剧烈时
  usable_signals: [action, joint_speeds (obs[5], obs[7], obs[11], obs[12])]
  risks: 可能限制必要的快速调整动作

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_reward
  reason: 无 success 标志，无法区分成功终止和失败终止
  forbidden_or_missing_signals: [info["success"], info["termination_reason"]]

- role_id: distance_to_goal
  reason: 无绝对位置观测，无法计算到终点的距离
  forbidden_or_missing_signals: [x_position, goal_position]

- role_id: foot_clearance
  reason: 无足部高度或位置观测
  forbidden_or_missing_signals: [foot_height, foot_position]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | 无 | dense_state_signal, clipped_scaling | 直接使用水平速度作为前进激励 |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | 无 | bounded_signal, quadratic_penalty | 对倾斜角度和角速度施加惩罚 |
| gait_cycle_encouragement | leg1_contact (obs[8]), leg2_contact (obs[13]) | 无 | binary_state_signal, periodic_pattern | 鼓励双腿交替接触地面 |
| energy_efficiency | action (4维) | 无 | quadratic_penalty, l2_norm | 对动作扭矩施加二次惩罚 |
| terrain_adaptation | lidar_0..9 (obs[14..23]) | 无 | feature_extraction, threshold_detection | 需要降维或提取地形特征 |
| smooth_motion | action, joint_speeds (obs[5],7,11,12) | 无 | temporal_difference, jerk_penalty | 对动作变化率或关节加速度惩罚 |
| explicit_success_reward | 无 | info["success"] | N/A | 信号缺失，禁用 |
| distance_to_goal | 无 | x_position, goal_position | N/A | 信号缺失，禁用 |
| foot_clearance | 无 | foot_height, foot_position | N/A | 信号缺失，禁用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 原地踏步/不前进 | horizontal_velocity 均值接近 0，但 leg_contact 交替正常 | 增加 forward_progress 权重，减少 gait_cycle 权重 |
| 向前倾倒行走 | hull_angle 持续偏向一侧，horizontal_velocity 高但角速度大 | 增加 balance_maintenance 权重，或加入 hull_angular_velocity 惩罚 |
| 动作过大/抖动 | action 值频繁接近 ±1，关节速度波动大 | 加入 energy_efficiency 或 smooth_motion 惩罚 |
| 单腿跳跃/蹦跳 | 双腿同时离地时间长，leg_contact 同时为 0 | 加强 gait_cycle_encouragement，惩罚双腿同时离地 |
| 地形适应失败 | LIDAR 显示前方有凸起但机器人未调整步态，摔倒 | 加入 terrain_adaptation 职责，利用 LIDAR 提前调整 |
| 过早摔倒 | 回合长度短，hull_angle 快速增大 | 检查 balance_maintenance 是否足够，或加入早期摔倒惩罚 |
| 步态不对称 | 一条腿的关节角度/速度明显不同于另一条 | 检查动作分布是否对称，或加入对称性约束 |
