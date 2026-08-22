# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个双足机器人依靠独立髋、膝关节力矩在不规则地形（包含梯子、树桩、坑洞等）上前进，尽可能走得更远且高效。机器人配备 10 个前方地面 LiDAR 距离传感器，可预判障碍物。主要目标是稳定行走并抵达地形终点，避免摔倒；次要目标是尽量减少不必要的关节力矩消耗。奖励设计需要鼓励向前移动、维持身体平衡、控制能耗，但不能直接使用任何官方奖励，且无额外显式的成功或失败信息字段。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心任务是让双足机器人持续前进穿越崎岖地形，主目标是“前进”而非到达指定目标点，没有抓取或操控物体，也没有强安全约束或生存导向的存活平衡。附属的节能、平稳属于典型连续控制任务的次要目标，不构成多目标权重冲突。因此属于 locomotion_continuous_control 族。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32
- obs[0]: hull_angle, 身体倾角，reward_usable: true（可用于摔倒检测）
- obs[1]: hull_angular_velocity, 身体角速度，reward_usable: true（辅助平衡惩罚）
- obs[2]: horizontal_speed, 质心水平速度，reward_usable: true（前进奖励核心信号）
- obs[3]: vertical_speed, 质心垂直速度，reward_usable: true（用于检测坠落或异常）
- obs[4]: joint_0_angle (hip_1), reward_usable: true（关节角度，可辅助步态）
- obs[5]: joint_0_speed (hip_1 角速度), reward_usable: true
- obs[6]: joint_1_angle (knee_1), reward_usable: true
- obs[7]: joint_1_speed (knee_1 角速度), reward_usable: true
- obs[8]: joint_2_angle (hip_2), reward_usable: true
- obs[9]: joint_2_speed (hip_2 角速度), reward_usable: true
- obs[10]: joint_3_angle (knee_2), reward_usable: true
- obs[11]: joint_3_speed (knee_2 角速度), reward_usable: true
- obs[12]: leg_1_ground_contact (二值 0/1)，reward_usable: true（步态分析，摔倒检测）
- obs[13]: leg_2_ground_contact (二值 0/1)，reward_usable: true
- obs[14]~obs[23]: lidar_1~lidar_10, LiDAR 距离读数，reward_usable: true（可用于地形预判，但非直接奖励信号，主要用于策略内部）

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: [-1.0, 1.0]
- action[0]: hip_1_torque, 第一髋关节力矩，连续值
- action[1]: knee_1_torque, 第一膝关节力矩，连续值
- action[2]: hip_2_torque, 第二髋关节力矩，连续值
- action[3]: knee_2_torque, 第二膝关节力矩，连续值

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形终点，通常视为成功，但无显式标志）
- failure-like termination: body_fallen_over（身体倾倒，通常视为失败，也无显式标志）
- ambiguous termination: 无
- truncation: 未说明 episod 有最大步数截断，从描述看假设仅由上述条件终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典）
- forbidden_or_uncertain_info_fields: 所有自定义字段均不可用
- 推断方法：终止时，可通过下一状态或终止时刻的观测信号推断类型：
  - body_fallen_over 可间接从 hull_angle 的绝对值突然超过某个阈值（例如 > 1.0 rad）、vertical_speed 的剧烈负值或 leg_ground_contact 变为零持续多步推导得到，标记为 derived_possible。
  - reached_end_of_terrain 可间接从 agent 持续前进了较远距离后 episode 终止，但 hull_angle、vertical_speed、leg_contact 等均保持正常（无跌倒迹象）推导得到，同样标记为 derived_possible。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：所有 24 维观测
- action：4 维动作（力矩）
- next_obs：下一时刻的 24 维观测
- info：空字典（无可用字段）
- training_progress：当前 prompt 没有明确允许，不使用

禁止使用：
- original_reward（已被掩盖）
- 任何未在 info 中声明的字段
- 直接读取终止原因（因为无 info 字段提供）

## 7. 可用于奖励函数的信号
- position/distance: 无绝对 x 坐标，只能通过水平速度积分相对前进距离，可使用 horizontal_speed 作为主信号。
- velocity: horizontal_speed (obs[2]), vertical_speed (obs[3]), hull_angular_velocity (obs[1])
- orientation: hull_angle (obs[0])，用于检测倾倒及姿态惩罚
- contact: leg_1_ground_contact (obs[12]), leg_2_ground_contact (obs[13])，可用于步态、摔倒推断
- action/engine: 动作扭矩 action[0:4]，用于能耗惩罚
- other: LiDAR 读数 (obs[14:24])，环境提供，但不建议直接用于奖励，因为它是感知信息而非任务指标；不过可用于奖励机制如“异常地形导致调整过大”的辅助惩罚，必要性低。
- derived_possible: 通过信号组合推断的 success（到达终点）和 failure（摔倒）事件，用于构建 sparse 成功/失败奖励，但无法直接读取标志，需在 compute_reward 中使用基于阈值的条件判断（例如在 terminated 状态下检查 next_obs 的 hull_angle 是否过大，或 vertical_speed 是否异常等）。此类推断有误判风险，但可在 reward shaping 中利用。

## 8. 不确定或不可用的信号
- explicit success/failure flags: 无
- 绝对位置、终点距离：无
- 地形类型、障碍物标签：无
- 能耗真实测量：仅能通过动作平方和近似
- 稳定站立判定：leaf 复杂接触不存在

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: bipedal_rough_terrain_locomotion
control_type: continuous
morphology:
  body_type: bipedal with rigid hull
  actuator_type: torque-controlled hip and knee joints (2 legs, 2 DOF each)
  contact_structure: two feet, binary ground contact sensors
primary_objectives:
  - Maximize forward distance traveled (survive and progress)
  - Avoid falling over (maintain balance)
secondary_objectives:
  - Minimize joint torque usage (energy efficiency)
  - Smooth gait (avoid excessive joint accelerations)
main_failure_risks:
  - Falling due to rough terrain or improper gait adaptation
  - Overreaction to LiDAR inputs causing instability
  - Stuck in local optima of inefficient but stable walking
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励机器人持续向前移动，最大化行进距离
  why_required: 任务是前进型 locomotion，没有前进奖励策略不会学习行走
  usable_signals: [horizontal_speed (obs[2])]
  risks: [可能鼓励高速但不稳定的步态，需配合平衡惩罚]

- role_id: fall_prevention
  purpose: 检测并惩罚身体倾倒，终止时给予较大负奖励
  why_required: 摔倒终止是主要失败模式，必须被惩罚
  usable_signals: [hull_angle (obs[0]), hull_angular_velocity (obs[1]), vertical_speed (obs[3]), leg_ground_contacts (obs[12,13])；终止时 next_obs 中可推断摔倒事件]
  risks: [基于阈值的伪死判定可能误判暂时的倾斜为摔倒；需要合理的阈值和延迟确认]

### 10.2 条件职责 conditional_roles
- role_id: energy_penalty
  purpose: 惩罚过大的关节力矩，促进高效步态
  condition_to_use: 当任务要求“最小化不必要的关节扭矩”时启用；作为次要目标，权重应低于前进奖励
  usable_signals: [action[0:4] 或连续力矩的平方]
  risks: [过高的惩罚权重可能导致机器人不敢用力，无法行走]

- role_id: smooth_gait_penalty
  purpose: 惩罚关节速度/加速度突变，鼓励流畅步态
  condition_to_use: 若环境需要适应粗糙地形，光滑性可能有利，但非强制
  usable_signals: [joint speeds (obs[5,7,9,11])，或相邻动作差]
  risks: [可能抑制必要的快速调整，削弱对地形的响应能力]

- role_id: survival_bonus
  purpose: 每存活一步给予小额正奖励，缓解稀疏性
  condition_to_use: 当仅使用 terminated 信号而不够密集时
  usable_signals: [不存在显式存活信号，但可通过“未摔倒”判断，即每步只要未终止则视为存活]
  risks: [在终止不可提前判断时难以实现，但此环境 terminates 仅由仿真状态引起，compute_reward 在 call 时已知 terminated 参数，因此可通过外部调用传入 terminated 标志（实际 compute 函数包含 terminated 参数吗？接口原型只给了 obs, action, next_obs, original_reward, info, training_progress，未包含 terminated）。所以 survival_bonus 需要 terminated 参数，若接口不允许则不可直接使用。但通常 reward 函数会在 step 内被调用并可访问 terminated 变量，此处约定未明确禁止，但谨慎处理。可以考虑如果无法获取 terminated，则使用连续正向速度奖励替代存活奖励。]
  risks: [若无 terminated，则只能通过正向速度实现，可视为同一职责]

### 10.3 慎用/禁用职责 avoid_roles
- role_id: lidar_usage_reward
  reason: LiDAR 是感知输入，不应直接作为奖励信号，否则策略可能学会“欺骗”奖励而非真正前进；仅在特殊场景中作为辅助特征使用，但无可靠度量标准。
  forbidden_or_missing_signals: [缺少 LiDAR 读数与地形相关的奖励度量（如通过障碍得分）]

- role_id: target_matching
  reason: 任务无显式目标位置，仅有终点；终点没有观测距离或方向，不适合做引导奖励。若使用 inferred success 给予稀疏奖励，但需要可靠检测。
  forbidden_or_missing_signals: [无终点距离信息]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | horizontal_speed (obs[2]) | absolute x position | dense_state_signal (linear scaling) | 直接使用水平速度作为即时奖励 |
| fall_prevention | hull_angle (obs[0]), hull_angular_velocity, vertical_speed, leg_contacts; derived termination flag (body_fallen_over inferred) | explicit fall flag | threshold_penalty (角度 > 阈值 大负奖励), termination_moment_penalty | 在 terminated 时基于 next_obs 中的倾角等判定是否摔倒，给予负奖励 |
| energy_penalty | action[0:4] | true torque cost | quadratic_penalty (sum of squares) | 惩罚力矩平方和 |
| smooth_gait_penalty | joint speeds (obs[5,7,9,11]), actions difference | none | bounded_signal (penalize high speed), action_diff_penalty | 可选，当步态过于抖动时启用 |
| survival_bonus (若 terminated 可用) | terminated flag (from env.step) | none | constant_bonus per step | 每步存活给小额奖励，需在 reward 函数中传入 terminated 参数，如不可行则使用 forward_progress 替代 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 机器人原地摇摆，不前进 | horizontal_speed 接近零，动作震荡 | 增大 forward_progress 权重，或加入步态对称性奖励 |
| 快速摔倒（早期终止） | hull_angle 迅速超过阈值，vertical_speed 突变 | 加强 fall_prevention 惩罚权重，引入平衡姿态奖励（如 hull_angle 的绝对惩罚） |
| 缓慢但高能耗行走 | energy_penalty 权重不足，动作方差大 | 逐渐增大 energy_penalty 系数，或者添加动作平滑度惩罚 |
| 到达终点但奖励总和不达预期 | reached_end 推断成功但 forward_progress 与 survival 总和较低 | 为成功终止添加较大的稀疏奖励（当可靠检测成功时） |
| LiDAR 使用不当导致策略不稳定 | 动作随 LiDAR 读数剧烈变化 | 不宜在奖励中直接使用 LiDAR，检查策略利用 LiDAR 的方式，也许引入 smoothness penalty |
