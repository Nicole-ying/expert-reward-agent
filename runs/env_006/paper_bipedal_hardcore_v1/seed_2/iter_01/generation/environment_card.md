# 匿名环境理解卡片

## 1. 任务目标
一个双足机器人在崎岖不平、布满障碍（阶梯、树桩、坑洼）的地形上尽可能远且高效地向前行走。机器人配备前向激光雷达，可感知前方地形高度。核心目标是学会利用激光雷达信息调整步态，在不摔倒的前提下持续前进；附属目标是减少不必要的关节扭矩（能量效率）并争取到达地形终点。不应将“到达终点”误解为唯一成功信号——能够稳定行走不摔倒才是关键，终点到达是终止条件之一但无独立奖励标注。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 核心要求是穿越粗糙地面持续前进，无特定目标位置标记，附属目标（能耗）与主目标不冲突但可降级。终止条件包含摔倒（失败）和到达终点（可能成功），但到达终点并非明确的导航式目标到达，而是前行到地形尽头的自然终止，符合 locomotion 型任务形态。

额外推演动力学子类型：该环境是典型平面（或准平面）双足行走任务，每条腿有髋、膝关节，torque-driven，可归为 planar_bipedal_gait。因此 dynamics_subtype: planar_bipedal_gait。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: 连续浮点 + 部分二值
- 详细字段（按索引）：
  - obs[0]: hull_angle (name: hull_angle) — 身体俯仰/倾斜角，reward_usable: true，用于检测摔倒和姿态稳定
  - obs[1]: hull_angular_velocity — 身体角速度，reward_usable: true，辅助姿态惩罚
  - obs[2]: horizontal_speed — 质心水平速度，reward_usable: true，核心前进信号
  - obs[3]: vertical_speed — 质心垂直速度，reward_usable: true，可能帮助判断弹跳或摔倒
  - obs[4]: hip_1_angle — 第1髋关节角度，reward_usable: true（关节状态跟踪）
  - obs[5]: hip_1_speed — 第1髋关节角速度
  - obs[6]: knee_1_angle — 第1膝关节角度
  - obs[7]: knee_1_speed — 第1膝关节角速度
  - obs[8]: hip_2_angle — 第2髋关节角度
  - obs[9]: hip_2_speed — 第2髋关节角速度
  - obs[10]: knee_2_angle — 第2膝关节角度
  - obs[11]: knee_2_speed — 第2膝关节角速度
  - obs[12]: leg_1_ground_contact — 第1腿接地指示（0/1），reward_usable: true，可作为步态接触约束
  - obs[13]: leg_2_ground_contact — 第2腿接地指示，同上
  - obs[14]~[23]: lidar_1~lidar_10 — 10个激光测距仪读数，表示前方地形高度。reward_usable: 谨慎使用，不可直接作为奖励项，但可间接推导预见性调整；初始训练阶段不建议直接奖励，但可帮助分析失败模式。

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- bounds: [-1.0, 1.0] 连续值
- 动作含义：
  - action[0]: hip_1_torque — 施加到第1髋关节的扭矩
  - action[1]: knee_1_torque — 施加到第1膝关节的扭矩
  - action[2]: hip_2_torque — 施加到第2髋关节的扭矩
  - action[3]: knee_2_torque — 施加到第2膝关节的扭矩

所有动作均为连续扭矩控制，无离散动作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形尽头），但无显式成功标志。可视为**成功的行走存活**导致的终止。
- failure-like termination: body_fallen_over（身体摔倒），常见于 hull_angle 过大或质心跳跃、触地异常。
- ambiguous termination: 无。
- truncation: 未定义明确截断（step source 中仅 terminated，无 truncated 分支）。因此所有 episode 结束均由终止条件触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 标志）
- explicit_failure_flag_available: false（同上）
- allowed_info_fields: []（interface 规定 info_is_empty，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为环境实际不提供任何 info。

尽管如此，可**从观测推导**终止类型：
- 摔倒 (derived_possible): 身体倾斜角 |hull_angle| 超出临界阈值（如 >0.5 rad），或 hull_angular_velocity 突变，同时 leg contact 可能消失。
- 到达终点 (derived_possible): 水平速度仍较高、姿态稳定时 episode 突然终止；也可结合上一步位置推断（但观测无位置），只能依赖速度与姿态平滑终止时的表现进行事后推测。但其可靠性不足以成为奖励条件，可偶尔用于事后分析。

## 6. reward 函数接口契约
函数签名（规范）：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（当前观察）
- action（当前动作）
- next_obs（下一步观察）
- info 中明确允许的字段（当前环境为空，不可使用任何 info）
- training_progress：仅在 prompt/task 明确允许时可用，本环境未提及，因此**不能**使用。

禁止使用：
- original_reward（官方奖励屏蔽）
- official_reward
- 任何未声明的 info 字段
- 任何未在观察空间中声明的前处理量（如绝对位置、能量等）

## 7. 可用于奖励函数的信号
- position: 观察中无绝对位置，仅可通过速度累积间接推断位移；无直接可用位置坐标。
- velocity: horizontal_speed（obs[2]）、vertical_speed（obs[3]）、各关节速度（obs[5,7,9,11]）
- orientation: hull_angle（obs[0]）、hull_angular_velocity（obs[1]）
- contact: leg_1_ground_contact（obs[12]）、leg_2_ground_contact（obs[13]）
- action/engine: 动作本身（4维扭矩）
- other:
  - laser scan（obs[14:23]）——可用于推断地形粗糙度，但需谨慎映射为奖励时容易引入噪声；暂时建议不作为常规奖励信号。
  - derived_possible: 通过 hull_angle 阈值或角速度突变推断摔倒；通过 episode 终止时水平速度 & 姿态推断“疑似成功到达”。

## 8. 不确定或不可用的信号
- 绝对位置（x坐标）：不可用，无法直接度量“走得远”。
- 地形尽头触发标志：不可用（info 为空）。
- 能量/功率指标：不可用，只能通过关节扭矩与速度近似。
- 明确 success/failure 标签：不可用。
- 任何 info 字段：均不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: two-legged planar (or quasi-planar) biped
  actuator_type: torque-controlled hinge joints (4 DoF)
  contact_structure: foot-ground contact sensing (2 binary signals)
primary_objectives:
  - walk forward as far as possible without falling
secondary_objectives:
  - minimize unnecessary joint torque (energy efficiency)
  - reach the terrain end (soft preference, not hard success)
main_failure_risks:
  - falling over due to unstable gait or obstacle collision
  - getting stuck or moving too slowly
  - excessive torque usage leading to jerky movements
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励机器人向前移动，维持正的水平速度
  why_required: 核心任务，不鼓励前进则无法完成 locomotion 目标
  usable_signals: [horizontal_speed (obs[2])]
  risks: 如果仅奖励速度，可能导致摔倒前仍高速冲刺。需结合姿态惩罚平衡。

- role_id: balance_penalty
  purpose: 惩罚过大的身体倾斜和角速度，防止摔倒
  why_required: 生存是完成任务的前提；摔倒会导致 episode 终止且无进度
  usable_signals: [hull_angle (obs[0]), hull_angular_velocity (obs[1])]
  risks: 过度惩罚会使机器人不敢迈步，导致移动僵硬缓慢；需要适度权重。

- role_id: fall_termination_penalty
  purpose: 对因摔倒而终止的 episode 施加较大负奖励，以强化安全行走
  why_required: 终止时无显式失败标志，但可通过观测推断摔倒并给予强负反馈，避免摔倒即结束但前期奖励较高的作弊行为。
  usable_signals: [hull_angle (obs[0] / next_obs[0]), hull_angular_velocity (next_obs[1]), horizontal_speed (next_obs[2]), 以及 terminated flag] — 通过 next_obs 中 hull_angle 超出阈值判断摔倒。
  risks: 阈值选择不当会导致误判；如果机器人恰好停住，也可能错误处罚。需结合 terminated 和姿态阈值。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 在机器人已经能够稳定行走若干步后（可通过训练进度或平均水平速度超过某个阈值时）逐步引入，避免早期抑制探索。
  purpose: 减少不必要的关节扭矩，使步态更优雅、节能
  usable_signals: [action (4维扭矩)，可选关节速度 (obs[5,7,9,11])]
  risks: 早期施加会抑制任何动作尝试，导致机器人不敢动；需要权重 scheduling 或 mask。

- role_id: smart_terrain_anticipation_reward (可选)
  condition_to_use: 仅当专家认为需要使用激光雷达来引导提前抬腿等行为，但当前环境不易从激光雷达直接计算奖励，一般不建议常规使用，保留作为未来研究方向。
  usable_signals: [lidar_1~10]
  risks: 极易引入噪声，使奖励分布混乱，除非设计极其精巧的势能函数，否则不推荐。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: goal_reaching_bonus (到达终点额外奖励)
  reason: 无法可靠识别何时到达终点（info 为空，观测无位置），只能通过间接推断，误判风险极高。因此**禁止**作为奖励职责。正确的做法是：到达终点自然终止，由 survival 时长和前进距离奖励隐式体现。
  forbidden_or_missing_signals: [explicit_end_flag, absolute_x_position]

- role_id: alive_bonus (固定每步生存奖励)
  reason: 前进速度奖励已经可以鼓励生存，额外固定生存奖励可能带来“原地踏步但存活”的作弊行为，除非同步加入最小速度约束。当前环境尚无该约束，故暂不建议单独使用。
  forbidden_or_missing_signals: none，但会与主进度奖励冲突。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | horizontal_speed (obs[2]) | absolute displacement | dense_state_signal, bounded_signal | 可直接使用正值奖励，可限制最大速度防止过度激励 |
| balance_penalty | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | body orientation matrix | quadratic_penalty, hinge_penalty | 常用绝对值或平方误差 |
| fall_termination_penalty | hull_angle (next_obs[0]), hull_angular_velocity (next_obs[1]), terminated flag | explicit fall flag | indicator × large_negative_constant | 通过角度阈值判断“已摔倒”，在最后一步给予巨额负奖励 |
| energy_efficiency | action vector (