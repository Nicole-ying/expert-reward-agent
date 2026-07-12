# 匿名环境理解卡片

## 1. 任务目标
这是一个连续控制运动任务。一个3D四足机器人拥有四条腿和八个力矩控制关节，必须向前行走或奔跑，同时保持身体直立。机器人的身体高度必须保持在健康区间内（最低高度至最高高度之间），一旦高度低于最低值（趴下）或高于最高值（弹飞）则回合立刻终止。智能体的核心目标是产生持续、稳定的向前运动，而不是仅仅维持站立不倒下。次要目标包括减小非必要的关节力矩以及维持机身姿态平稳。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务的核心目标是持续向前行走/奔跑，身体高度与姿态只是保证任务可持续的健康约束，不属于独立的同等权重目标。因此归为连续控制前进类，而非多目标或幸存平衡。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: float
- 各维度含义：
  - obs[0]: body_z，身体垂直高度，reward_usable: true
  - obs[1]: quat_w，身体朝向四元数 w 分量，reward_usable: true (用于计算直立程度)
  - obs[2]: quat_x，身体朝向四元数 x 分量，reward_usable: true
  - obs[3]: quat_y，身体朝向四元数 y 分量，reward_usable: true
  - obs[4]: quat_z，身体朝向四元数 z 分量，reward_usable: true
  - obs[5]: joint_1_angle，第一个髋关节角度，reward_usable: false (一般不用)
  - obs[6]: joint_2_angle，第一个踝关节角度，reward_usable: false
  - obs[7]: joint_3_angle，第二个髋关节角度，reward_usable: false
  - obs[8]: joint_4_angle，第二个踝关节角度，reward_usable: false
  - obs[9]: joint_5_angle，第三个髋关节角度，reward_usable: false
  - obs[10]: joint_6_angle，第三个踝关节角度，reward_usable: false
  - obs[11]: joint_7_angle，第四个髋关节角度，reward_usable: false
  - obs[12]: joint_8_angle，第四个踝关节角度，reward_usable: false
  - obs[13]: body_x_velocity，身体世界x方向速度（前进速度），reward_usable: true (主要目标信号)
  - obs[14]: body_y_velocity，身体世界y方向速度（横向速度），reward_usable: true (可用，但非主要)
  - obs[15]: body_z_velocity，身体垂直速度，reward_usable: true (可用于平稳性)
  - obs[16]: body_roll_velocity，滚转角速度，reward_usable: true (姿态稳定性)
  - obs[17]: body_pitch_velocity，俯仰角速度，reward_usable: true
  - obs[18]: body_yaw_velocity，偏航角速度，reward_usable: true (非必须，但可约束)
  - obs[19]: joint_1_velocity，第一个髋关节角速度，reward_usable: false (很少用)
  - obs[20]: joint_2_velocity，第一个踝关节角速度，reward_usable: false
  - obs[21]: joint_3_velocity，第二个髋关节角速度，reward_usable: false
  - obs[22]: joint_4_velocity，第二个踝关节角速度，reward_usable: false
  - obs[23]: joint_5_velocity，第三个髋关节角速度，reward_usable: false
  - obs[24]: joint_6_velocity，第三个踝关节角速度，reward_usable: false
  - obs[25]: joint_7_velocity，第四个髋关节角速度，reward_usable: false
  - obs[26]: joint_8_velocity，第四个踝关节角速度，reward_usable: false

注意：前进方向对应obs[13]，且info完全不可访问，所有必须的奖励信号只能来自obs和action。

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- dtype: float
- 连续动作空间，每个分量范围[-1.0, 1.0]，代表关节力矩。
  - action_dim 0: hip_1_torque，第一个髋关节力矩
  - action_dim 1: ankle_1_torque，第一个踝关节力矩
  - action_dim 2: hip_2_torque，第二个髋关节力矩
  - action_dim 3: ankle_2_torque，第二个踝关节力矩
  - action_dim 4: hip_3_torque，第三个髋关节力矩
  - action_dim 5: ankle_3_torque，第三个踝关节力矩
  - action_dim 6: hip_4_torque，第四个髋关节力矩
  - action_dim 7: ankle_4_torque，第四个踝关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止，只有达到时间上限truncated才算“存活满时间”。
- failure-like termination:
  - body_height_outside_healthy_range：高度低于0.2或高于1.0（跌倒/弹飞），视为失败。
  - state_value_outside_finite_range：任何状态值变为NaN或inf，视为失败。
- ambiguous termination: 无
- truncation: time_limit_reached，表示达到最大步数，并非失败，应该视为中性或轻微正向。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false (只能通过terminated推断，但官方不允许直接使用terminated标记作为奖励信号，除非我们从状态判断)
- allowed_info_fields: []  (info为空)
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等全部被禁止

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段（此处info为空，禁止使用任何info字段）
- training_progress 只有prompt明确允许时才用（本次未获允许，所以不能用）

禁止使用：
- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片
- 环境内部未暴露的任何信号（如接触力、关节力矩等）

## 7. 可用于奖励函数的信号
- position: body_z (obs[0]), quaternion (obs[1:5]) 可计算直立投影 (body_up_z = 1 - 2*(quat_x²+quat_y²))
- velocity: body_x_velocity (obs[13]), body_y_velocity (obs[14]), body_z_velocity (obs[15]), body_roll_velocity (obs[16]), body_pitch_velocity (obs[17]), body_yaw_velocity (obs[18])
- orientation: quat可直接用于惩罚翻滚/俯仰过大
- contact: 无
- action/engine: action 力矩本身可用于平滑性惩罚
- other: 身体高度是否在健康区间内 (可根据obs[0]判定濒死区域)

## 8. 不确定或不可用的信号
- 绝对位置x,y：不可用（被禁止）
- 接触力/脚掌压力：观察空间中没有
- 是否有脚掌离地、步态事件等：没有
- 成功/失败标志：无，只能从高度判跌倒
- 训练进度training_progress: 未允许使用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 3D_quadruped_rigid_body
  actuator_type: torque_controlled_joints (8 DoF: 4 hip + 4 ankle)
  contact_structure: foot_ground_contacts (implicit, not observable)
primary_objectives:
  - maximize_forward_velocity (body_x_velocity)
  - maintain_healthy_body_height (0.2 < body_z < 1.0)
  - keep_body_upright (body_up_z close to 1.0)
secondary_objectives:
  - penalize_large_joint_torques (action)
  - penalize_lateral_drift (body_y_velocity)
  - penalize_high_angular_velocities (roll/pitch/yaw)
main_failure_risks:
  - falling over (body_z < 0.2)
  - launching or jumping too high (body_z > 1.0)
  - NaN/Inf instability (poorly tuned actions)
  - getting stuck with zero forward velocity but still standing
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward
  purpose: 激励向前移动的速度
  why_required: 任务的核心目标是前进，缺少此责则无法形成运动策略
  usable_signals: [obs[13] (body_x_velocity)]
  risks: 仅激励速度可能忽略稳定性，需与生存条件结合

- role_id: body_height_survival
  purpose: 驱使机器人保持身体高度在健康区间内，远离上下边界
  why_required: 健康高度是继续任务的必要条件，一旦超出回合终止，因此需设计为惩罚接近边界
  usable_signals: [obs[0] (body_z)]
  risks: 仅仅使用区间内奖励可能导致不思进取，需要设计成边界处急剧下降的势能

- role_id: upright_orientation_penalty
  purpose: 维持机身大致直立，防止翻滚
  why_required: 过度倾斜会导致无法有效向前移动，且容易跌倒
  usable_signals: [obs[1:5] quaternion 可计算 body_up_z]
  risks: 过强惩罚可能抑制必要的侧向倾斜，影响转弯或平衡调整

### 10.2 条件职责 conditional_roles
- role_id: joint_torque_regularization
  condition_to_use: 当策略已经能稳定前进后，用于降低能耗、提高动作效率
  usable_signals: [action (力矩)]
  risks: 早期加入过强的力矩惩罚会阻碍探索有效步态

- role_id: lateral_velocity_penalty
  condition_to_use: 当策略出现明显的横向漂移且影响前进效率时启用
  usable_signals: [obs[14] (body_y_velocity)]
  risks: 轻微侧移可能在某些步态下是自然现象，过度压制可能导致步态僵硬

- role_id: vertical_velocity_penalty
  condition_to_use: 当机器人出现过度跳跃或上下颠簸时启用
  usable_signals: [obs[15] (body_z_velocity)]
  risks: 机器人在行走时垂直方向速度本就不为零，阈值需仔细设计

- role_id: angular_velocity_penalty
  condition_to_use: 当姿态变化过于剧烈时启用，辅助保持稳定
  usable_signals: [obs[16], obs[17], obs[18]]
  risks: 抑制转弯灵活性，不建议在训练早期使用

### 10.3 慎用/禁用职责 avoid_roles
- role_id: terminal_bonus_based_on_time_survived
  reason: 没有info字段可用来获得生存步数，且training_progress不可用；强行从环境隐含状态推断复杂且不可靠，且可能误导reward shaping
  forbidden_or_missing_signals: [无准确生存时间信号]

- role_id: foot_contact_coordination
  reason: 环境未提供任何接触力或触地检测信号
  forbidden_or_missing_signals: [contact sensors]

- role_id: distance_from_origin
  reason: 官方明确禁止使用x_position或distance_from_origin，且这些信号不直接从 obs 暴露
  forbidden_or_missing_signals: [x_position, y_position]

- role_id: curriculum_on_velocity_target
  reason: 不允许使用官方reward项，且无法获取训练进度变量
  forbidden_or_missing_signals: [training_progress, any curriculum signal]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_velocity_reward | obs[13] (body_x_velocity) | 无 | dense_state_signal → clip or linear scale | 可直接乘以正系数，或使用有上限的饱和函数防止异常大值 |
| body_height_survival | obs[0] (body_z) | 无 | bounded_signal → penalty for near_boundaries (outside healthy range) | 设计为健康区域为0代价，接近边界急剧上升的二次或指数惩罚 |
| upright_orientation_penalty | obs[1:5] → body_up_z = 1 - 2*(q_x^2+q_y^2) | 无 | bounded_signal → 1 - body_up_z 作为惩罚 | body_up_z在[0,1]，直立为1 |
| joint_torque_regularization | action[0:8] | 无 | quadratic_penalty on action | 需缩放权重以免与前进奖励竞争 |
| lateral_velocity_penalty | obs[14] | 无 | absolute_value or squared penalty | 谨慎权重，可能为0或很小 |
| vertical_velocity_penalty | obs[15] | 无 | absolute_value or squared penalty, with small deadzone | 行走自然会有小幅上下波动 |
| angular_velocity_penalty | obs[16], obs[17], obs[18] | 无 | sum of squares | 对roll/pitch特别敏感，yaw可忽略 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 策略收敛到静止站立不动（零前进速度） | body_x_velocity始终接近0，身体高度稳定 | 检查高度生存奖励是否过强压制了前进项，降低生存奖励权重或增加速度敏感度 |
| 频繁跌倒或过早终止（高度过低） | 平均 episode 长度短，body_z经常接近0.2 | 加强高度惩罚的陡峭程度，或在奖励中增加平滑的“健康高度”势能 |
| 机器人以怪异姿势前进（如倒立或侧翻行走） | body_up_z 显著小于1，但身体高度仍在范围内且前进速度快 | 提高直立惩罚系数，或重新训练 |
| 高 oscilation / 关节抖动 | action 变化剧烈，joint_velocities 很大 | 加入 torque_regularization 和平滑惩罚 |
| 前进方向漂移严重 | body_y_velocity 绝对值很大 | 加入横向速度惩罚 |
| NaN/Inf 终止频繁 | 日志中出现非有限值 | 检查动作输出是否有极大值，clip 或降低力矩范围，或加入平稳状态奖励 |