# 匿名环境理解卡片

## 1. 任务目标
这是一个三维四足机器人在连续空间中的运动控制任务。机器人拥有8个扭矩控制关节（4条腿各2关节），必须控制关节扭矩使机器人**稳定地向前行走或奔跑**。**主要目标**是产生持续的向前运动（forward locomotion），**必要存活约束**是保持身体高度在健康范围内（0.2 m ~ 1.0 m），一旦高度超出该范围即提前终止。同时机器人需要尽量**保持躯干直立**，避免侧翻或倾覆。任务强调“维持稳定前进而非仅仅保持站立”，因此奖励设计应鼓励有效的向前位移和步态稳定性，而非简单奖励存活时间。**注意**：不存在显式的位置目标或到达点位，核心是持续、稳健的向前推进。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务明确要求“向前行走/奔跑”作为核心目标，没有固定的目标位置，机器人需要在连续的平地上持续产生前进运动，符合 locomotion_continuous_control 族特征。附属的存活、直立、能耗等均为安全性或平滑性约束，不是权重相当的多目标。四足运动属于典型的多腿持续前进控制。

动力学子类型（dynamics_subtype）进一步细分为 **multi_legged_body_locomotion**，因为该环境具有多足（四足）、高维身体状态、需协调步态完成向前行进的特点。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: float32 (推断)
- 各维度含义：

| 索引 | 名称 | 含义 | 可用于奖励 |
|------|------|------|-------------|
| 0 | body_z | 主体中心的垂直高度 (m) | true |
| 1 | quat_w | 身体姿态四元数实部 | true (组合可得竖直分量) |
| 2 | quat_x | 身体姿态四元数 x 分量 | true |
| 3 | quat_y | 身体姿态四元数 y 分量 | true |
| 4 | quat_z | 身体姿态四元数 z 分量 | true |
| 5 | joint_1_angle | 第一髋关节角度 | true (运动学使用) |
| 6 | joint_2_angle | 第一踝关节角度 | true |
| 7 | joint_3_angle | 第二髋关节角度 | true |
| 8 | joint_4_angle | 第二踝关节角度 | true |
| 9 | joint_5_angle | 第三髋关节角度 | true |
| 10 | joint_6_angle | 第三踝关节角度 | true |
| 11 | joint_7_angle | 第四髋关节角度 | true |
| 12 | joint_8_angle | 第四踝关节角度 | true |
| 13 | body_x_velocity | 主体在世界 x 方向的速度（前进方向） | true（核心信号） |
| 14 | body_y_velocity | 主体侧向速度 | true（可抑制侧移） |
| 15 | body_z_velocity | 主体垂直速度 | true（着陆冲击等） |
| 16 | body_roll_velocity | 翻滚角速度 | true（稳定性） |
| 17 | body_pitch_velocity | 俯仰角速度 | true |
| 18 | body_yaw_velocity | 偏航角速度 | true（航向保持） |
| 19 | joint_1_velocity | 第一髋关节角速度 | true |
| 20 | joint_2_velocity | 第一踝关节角速度 | true |
| 21 | joint_3_velocity | 第二髋关节角速度 | true |
| 22 | joint_4_velocity | 第二踝关节角速度 | true |
| 23 | joint_5_velocity | 第三髋关节角速度 | true |
| 24 | joint_6_velocity | 第三踝关节角速度 | true |
| 25 | joint_7_velocity | 第四髋关节角速度 | true |
| 26 | joint_8_velocity | 第四踝关节角速度 | true |

备注：没有接触力、足端信息、相对位置等，仅提供纯本体状态量。

## 4. 动作空间 action_space
- type: Box (连续)
- shape: [8]
- dtype: float32
- 范围: 每个维度 [-1.0, 1.0]

| 维度 | 名称 | 含义 |
|------|------|------|
| 0 | hip_1_torque | 第一条腿髋关节力矩 |
| 1 | ankle_1_torque | 第一条腿踝关节力矩 |
| 2 | hip_2_torque | 第二条腿髋关节力矩 |
| 3 | ankle_2_torque | 第二条腿踝关节力矩 |
| 4 | hip_3_torque | 第三条腿髋关节力矩 |
| 5 | ankle_3_torque | 第三条腿踝关节力矩 |
| 6 | hip_4_torque | 第四条腿髋关节力矩 |
| 7 | ankle_4_torque | 第四条腿踝关节力矩 |

动作为直接扭矩控制，没有目标角度、目标角速度等高层接口，属于低层扭矩级控制。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **没有显式成功终止条件**。任务不存在“到达目标”概念，只有持续存活和前进。因此唯一可能被认为是“成功”的情况是 episode 自然截断（time limit），但需配合前向移动和高度稳定来推断其质量。
- failure-like termination: 
  - 身体高度超出健康范围（min_z=0.2, max_z=1.0），原因可能是摔倒、跃起过高等。
  - 状态值变为 NaN 或 Inf。
- ambiguous termination: 无。
- truncation: 达到最大步数限制（时间截断），不属于失败，但也不一定代表任务完成良好（可能仅仅维持站立不动）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  没有任何 info 字段指示成功，且任务本身没有二值成功定义。
- explicit_failure_flag_available: true  
  失败可以通过 terminated 且不是截断来间接推断，但不能在 compute_reward 中使用 terminated 标志（因为 compute_reward 不直接接收 terminated 标志，除非由环境在每步结束时提供）。在奖励函数中，只能利用 next_obs 中 body_z 是否在健康范围内来惩罚即将发生的失败；不能直接使用 episode 的终止状态。
- allowed_info_fields: [] （空，无任何可用于奖励的 info 字段）
- forbidden_or_uncertain_info_fields: （明确禁止）
  - reward_forward
  - reward_ctrl
  - reward_contact
  - reward_survive
  - x_position
  - y_position
  - distance_from_origin

**说明**：奖励函数必须完全基于 obs、action、next_obs 和禁止使用 original_reward、info 中的任何字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: numpy array shape (27,)
- action: numpy array shape (8,)
- next_obs: numpy array shape (27,)
- info: dict，但当前环境下为**空字典**，禁止使用任何字段
- training_progress: float 0.0~1.0，仅在 prompt 明确要求时使用（当前未要求，谨慎）

禁止使用：
- original_reward（已遮盖）
- official_reward
- info 中的任何字段（包括任何 reward_*）
- 未声明的 obs 切片扩展语义（如基于物理引擎的内部状态）
- 任何全局/非马尔可夫状态（如累积距离）

## 7. 可用于奖励函数的信号
- **position**:
  - 身体高度：obs[0] (body_z)；next_obs[0]
  - 身体姿态四元数：obs[1:5]；从中可计算躯干 z 轴在世界系投影 upright_proj = 1 - 2*(quat_x² + quat_y²)
  - 各个关节角度：obs[5:13]，用于构造姿态偏好或关节限位惩罚（若有边界隐式，但未给出具体限位；可假设一定范围）
- **velocity**:
  - 前进速度（主方向）：obs[13] (body_x_velocity)，可直接作为正向激励
  - 侧向速度：obs[14]，鼓励接近 0
  - 垂直速度：obs[15]，结合触地判断（无触地信息）或避免过大冲击
  - 本体角速度：obs[16:19] (roll, pitch, yaw rate)，用于惩罚翻倒、急转
  - 关节角速度：obs[19:27]，可用于动作平滑性或能耗惩罚
- **orientation**:
  - 通过四元数计算 upright = 1 - 2*(q_x²+q_y²)，范围为 [-1, 1]，1.0表示完全直立
- **contact**: 无接触力、无足底压力等信号
- **action/engine**:
  - action 本身可用于惩罚过大扭矩或 torch 变化率（相邻步 action 差值在智能体中可用，但 reward 函数是单步无记忆，且没有提供 prev_action，故不能直接使用相邻步差。若需要平滑，只能用相邻步 action 差，但 compute_reward 只接收当前步和下一步的 obs, action。没有 prev_action，无法计算 action 平滑。但可以通过 next_obs 的关节加速度间接推断？不直接。我们只能利用当前 action 大小。）
- **other**:
  - termination 标记无法在 compute_reward 中使用，因为函数不接收 terminated 布尔值（但可通过 next_obs 有效状态推断，如果 next_obs 无效？但 step 返回终止后应停止调用 compute_reward，所以不会用到无效 next_obs。不过可以通过观察当前 obs 和 next_obs 的高度是否进入危险区来塑造）。

## 8. 不确定或不可用的信号
- 世界绝对位置 x, y 被禁止，不能重建
- 无接触信息，无法给足端触地奖励或空翻检测
- 无能耗量（电机转矩*速度）可直接计算，但可利用 action（力矩）和关节速度乘积近似，但无关节电流等精确能耗
- 无显式存活奖励计数器或步数

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous (torque)
morphology:
  body_type: 3D quadruped, four legs, each with 2 DoF (hip + ankle)
  actuator_type: torque-controlled (8 independent joints)
  contact_structure: four point feet contacting ground, no contact force observation
primary_objectives:
  - maximize forward velocity (body_x_velocity)
  - maintain body height within safe range (0.2, 1.0)
secondary_objectives:
  - keep body upright (quaternion-derived up-projection close to 1.0)
  - minimize lateral drift (body_y_velocity near 0)
  - smooth locomotion (low action magnitudes or low body angular velocities)
main_failure_risks:
  - falling forward/backward or sideways causing early termination
  - standing statically without meaningful forward progress (low velocity)
  - excessive jumping or launching that triggers upper height bound
  - high-frequency torque oscillation causing instability or energy waste
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward
  purpose: 驱动机器人产生正向前进速度，使 agent 不止于站立。
  why_required: 这是任务的核心目标（locomotion），若无此奖励，策略可能只学会稳定站立。
  usable_signals: [obs[13] (body_x_velocity)]，必要时可做限幅或死区。
  risks: 若无限激励，可能导致机器人奔跑不稳、频繁摔倒，需结合存活和稳定性约束。

- role_id: height_survival_reward (or penalty)
  purpose: 将身体高度维持在健康范围 (0.2, 1.0) 内，避免提前终止；靠近边界时给予惩罚，正常范围给予中性或小额奖励。
  why_required: 高度超出直接终止，策略必须学会避免，属于安全硬约束。
  usable_signals: [obs[0], next_obs[0]]，可定义区域，如接近0.25或0.9时给予负奖励。
  risks: 如果惩罚太强，可能抑制探索；太弱则不能有效防止摔倒。

- role_id: upright_orientation_reward
  purpose: 保持躯干基本直立（世界z轴），防止侧翻。
  why_required: 摔倒与高度崩塌高度相关，且对向前运动有破坏。
  usable_signals: [obs[1:5] 四元数]，计算 body_up_z = 1 - 2*(quat_x²+quat_y²)，当该值 > 0.7 左右视为稳定。
  risks: 过于苛刻会阻碍轻微倾斜步态的自然学习，需要容忍一定范围的倾角。

### 10.2 条件职责 conditional_roles
- role_id: lateral_motion_penalty
  purpose: 抑制侧向漂移，使前进方向更纯净。
  condition_to_use: 当训练初期出现明显侧向漂移、或因侧向速度过大导致不稳定时加入；稳定后可降低系数。
  usable_signals: [obs[14] (body_y_velocity)]，可加绝对值或平方。
  risks: 若双侧对称运动天然有小幅侧向震荡，惩罚过大会抑制正常步态，应保守使用。

- role_id: action_magnitude_penalty
  purpose: 降低能量消耗、抑制高频振荡，提高样本效率。
  condition_to_use: 当观察到动作大幅震荡或 motor overheat 时加入；或作为微调后的细化项。
  usable_signals: [action](8维向量)，可求和平方。
  risks: 过早加入会抑制探索，导致步伐极弱或站姿；需在基础前进能力稳定后使用。

- role_id: joint_velocity_penalty
  purpose: 与动作惩罚类似，通过关节速度间接鼓励平滑动作。
  condition_to_use: 若无法获取上一动作差值，可替代平滑项使用。
  usable_signals: [obs[19:27] 关节速度]。
  risks: 可能误伤需要的快速摆动，最好与前进速度奖励协同调权。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_survival_time_reward
  reason: 无 info 字段，无法获取生存步数。若想用，可能扭曲为“尽量不动”，不可行。
- role_id: distance_from_origin_reward
  reason: info 中的 x_position 被禁止，无法安全构造；利用速度积分也违背无状态约束。
- role_id: contact_foot_reward
  reason: 没有任何接触信号，无法判断哪只脚着地，故禁用。
- role_id: multi_objective_weights_learning
  reason: 不属于当前环境的需求，且无辅助信号。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes