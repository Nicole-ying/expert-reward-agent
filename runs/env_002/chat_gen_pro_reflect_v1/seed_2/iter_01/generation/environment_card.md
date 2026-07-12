# 匿名环境理解卡片

## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：任务不是到达某个精确目标点，也不是跳跃或奔跑，而是持续稳定的双足步态前进。能量最小化是附属优化目标，不是核心任务目标。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 核心目标是持续前进通过地形，属于典型的连续控制 locomotion 任务。附属有能耗和平滑要求，但主目标明确是前进距离和速度。

dynamics_subtype: planar_bipedal_gait
reason: 双足身体、髋关节和膝关节协调、平面运动、步态稳定是核心动力学特征。

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
- success-like termination: reached_end_of_terrain — 成功走完地形，属于成功终止
- failure-like termination: body_fallen_over — 身体摔倒，属于失败终止
- ambiguous termination: 无
- truncation: 无（step 返回 False 作为 truncation 标志）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空字典 {}，没有显式 success 标志）
- explicit_failure_flag_available: false（info 为空字典 {}，没有显式 failure 标志）
- allowed_info_fields: 无（info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为 info 为空

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前时间步的观测 (24维)
- action: 当前时间步的动作 (4维)
- next_obs: 下一时间步的观测 (24维)
- info: 空字典，不可用
- training_progress: 只有 prompt 明确允许时才用

禁止使用：
- original_reward: 官方奖励被屏蔽，禁止使用
- official_reward: 禁止使用
- 未声明的 info 字段: info 为空，无可用字段
- 未声明的 obs 切片: 只能使用上述定义的 24 维观测

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，但可通过 LIDAR 和速度间接推断
- velocity: horizontal_velocity (obs[2]), vertical_velocity (obs[3]), hull_angular_velocity (obs[1]), joint speeds (obs[5], obs[7], obs[10], obs[12])
- orientation: hull_angle (obs[0]), joint angles (obs[4], obs[6], obs[9], obs[11])
- contact: leg1_contact (obs[8]), leg2_contact (obs[13])
- action/engine: action 本身 (4维扭矩)
- other: LIDAR 测距 (obs[14..23])

## 8. 不确定或不可用的信号
- 绝对位置 (x, y): 不可用，无此观测
- 地形高度/坡度: 不可直接获得，但可通过 LIDAR 间接推断
- 能量消耗: 不可直接获得，但可通过 action 的平方和近似
- 步态周期/相位: 不可直接获得，需从关节角度和接触信号推断
- 成功/失败标志: info 为空，无显式标志
- 时间步数: 不可用，无此观测

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
  - maximize forward distance traveled
  - maximize forward speed
  - maintain balance (avoid falling)
secondary_objectives:
  - minimize energy consumption (action magnitude)
  - maintain smooth gait (avoid jerky movements)
main_failure_risks:
  - falling over (body_fallen_over)
  - getting stuck on uneven terrain
  - inefficient gait leading to early termination
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励身体向前移动，最大化前进距离和速度
  why_required: 这是任务的核心目标
  usable_signals: [horizontal_velocity (obs[2])]
  risks: 可能鼓励过度前倾导致摔倒，需要平衡

- role_id: balance_maintenance
  purpose: 保持身体直立，防止摔倒
  why_required: 摔倒直接终止任务，是生存必要条件
  usable_signals: [hull_angle (obs[0]), hull_angular_velocity (obs[1])]
  risks: 过度惩罚倾斜可能导致步态僵硬，前进速度下降

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 当基础前进和平衡已经稳定，需要优化能耗时
  usable_signals: [action (4维扭矩)]
  risks: 过早加入可能阻碍探索，导致 agent 不敢动作

- role_id: gait_smoothness
  condition_to_use: 当步态出现明显抖动或不自然时
  usable_signals: [joint angles (obs[4], obs[6], obs[9], obs[11]), joint speeds (obs[5], obs[7], obs[10], obs[12])]
  risks: 过度平滑可能降低响应速度

- role_id: terrain_adaptation
  condition_to_use: 当 LIDAR 显示前方地形有明显起伏时
  usable_signals: [lidar_0..9 (obs[14..23])]
  risks: LIDAR 信号噪声大，直接使用可能导致不稳定

### 10.3 慎用/禁用职责 avoid_roles
- role_id: contact_rhythm
  reason: 没有明确的步态周期信号，接触标志是二值且可能噪声大，难以构建稳定的节奏奖励
  forbidden_or_missing_signals: [无步态相位信号，接触标志只有二值]

- role_id: height_maintenance
  reason: 没有身体高度观测，无法直接控制身体高度
  forbidden_or_missing_signals: [无垂直位置观测]

- role_id: goal_reaching
  reason: 任务不是到达特定目标点，而是持续前进
  forbidden_or_missing_signals: [无目标位置信号]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | 无 | dense_state_signal, clipped_scaling | 直接使用水平速度作为正向奖励 |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | 无 | bounded_signal, quadratic_penalty | 角度偏离竖直方向越远惩罚越大 |
| energy_efficiency | action (4维) | 无 | quadratic_penalty, l2_norm | 动作平方和作为能耗惩罚 |
| gait_smoothness | joint angles, joint speeds | 无 | jerk_penalty, acceleration_penalty | 关节角加速度或加加速度惩罚 |
| terrain_adaptation | lidar_0..9 (obs[14..23]) | 无 | proximity_aware_penalty | 根据前方地形调整步态 |
| contact_rhythm | leg1_contact, leg2_contact | 步态相位信号 | 无 | 信号不足，不建议使用 |
| height_maintenance | 无 | 垂直位置 | 无 | 信号缺失，无法使用 |
| goal_reaching | 无 | 目标位置 | 无 | 信号缺失，无法使用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 过早摔倒 | 训练早期 episode 长度很短，hull_angle 快速偏离 | 增加 balance_maintenance 权重，或加入早期摔倒惩罚 |
| 原地踏步不前 | horizontal_velocity 接近 0，但 episode 未终止 | 增加 forward_progress 权重，或加入速度阈值奖励 |
| 步态僵硬/不自然 | 关节角度变化小，动作幅度小 | 加入 gait_smoothness 或减少 energy_efficiency 权重 |
| 过度前倾冲刺 | horizontal_velocity 高但 hull_angle 持续偏离 | 增加 balance_maintenance 权重，或加入角度-速度耦合惩罚 |
| 对地形变化不敏感 | 在 LIDAR 显示地形变化时仍保持相同步态 | 加入 terrain_adaptation 职责 |
| 动作过大/能耗高 | action 值接近边界，动作方差大 | 加入 energy_efficiency 职责 |
| 单腿主导 | 一条腿的接触时间远多于另一条 | 加入对称性奖励或 contact_rhythm（如果信号允许） |