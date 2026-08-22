# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个 3D 四足机器人（8 个扭矩控制关节）实现稳定前进。核心目标是保持机器人身体直立且高度位于健康范围（0.2~1.0），并产生持续的向前速度。次要目标是提升前进速度，但不得以牺牲稳定性为代价。任务不是简单的“存活/保持平衡”，而是在存活前提下最大化前进进度。若机器人倾倒或高度越界，episode 提前终止。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务主目标是在保持平衡（健康高度、直立）的条件下持续前进，属于典型的“持续前进通过地形”类控制，到达目标位置或抓取操作不可见。速度是附属优化，前进进度才是主干，因此不属于 survival_balance_task（并非只求不倒）或多目标冲突型。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: float (continuous)
- obs[0]: body_z, 机身垂直高度, reward_usable: true（健康门控、摔倒推断）
- obs[1]: quat_w, 身体朝向四元数实部, reward_usable: true（与 quat_x,y,z 联合用于直立计算）
- obs[2]: quat_x, 身体朝向四元数虚部 x, reward_usable: true
- obs[3]: quat_y, 身体朝向四元数虚部 y, reward_usable: true
- obs[4]: quat_z, 身体朝向四元数虚部 z, reward_usable: true
- obs[5]: joint_1_angle, hip_1 角度, reward_usable: true（关节状态/姿态偏好）
- obs[6]: joint_2_angle, ankle_1 角度, reward_usable: true
- obs[7]: joint_3_angle, hip_2 角度, reward_usable: true
- obs[8]: joint_4_angle, ankle_2 角度, reward_usable: true
- obs[9]: joint_5_angle, hip_3 角度, reward_usable: true
- obs[10]: joint_6_angle, ankle_3 角度, reward_usable: true
- obs[11]: joint_7_angle, hip_4 角度, reward_usable: true
- obs[12]: joint_8_angle, ankle_4 角度, reward_usable: true
- obs[13]: body_x_velocity, 机身世界 x 轴前进速度, reward_usable: true（主进度信号）
- obs[14]: body_y_velocity, 机身世界 y 轴侧向速度, reward_usable: true（侧向漂移惩罚）
- obs[15]: body_z_velocity, 机身垂直速度, reward_usable: true（与健康/摔倒相关）
- obs[16]: body_roll_velocity, 翻滚角速度, reward_usable: true
- obs[17]: body_pitch_velocity, 俯仰角速度, reward_usable: true
- obs[18]: body_yaw_velocity, 偏航角速度, reward_usable: true
- obs[19]: joint_1_velocity, hip_1 角速度, reward_usable: true（动作平滑/能耗）
- obs[20]: joint_2_velocity, ankle_1 角速度, reward_usable: true
- obs[21]: joint_3_velocity, hip_2 角速度, reward_usable: true
- obs[22]: joint_4_velocity, ankle_2 角速度, reward_usable: true
- obs[23]: joint_5_velocity, hip_3 角速度, reward_usable: true
- obs[24]: joint_6_velocity, ankle_3 角速度, reward_usable: true
- obs[25]: joint_7_velocity, hip_4 角速度, reward_usable: true
- obs[26]: joint_8_velocity, ankle_4 角速度, reward_usable: true

## 4. 动作空间 action_space
- type: Box (continuous)
- shape: [8]
- action[0]: hip_1_torque, 第一个髋关节扭矩，范围 [-1, 1]
- action[1]: ankle_1_torque, 第一个踝关节扭矩，范围 [-1, 1]
- action[2]: hip_2_torque, 第二个髋关节扭矩，范围 [-1, 1]
- action[3]: ankle_2_torque, 第二个踝关节扭矩，范围 [-1, 1]
- action[4]: hip_3_torque, 第三个髋关节扭矩，范围 [-1, 1]
- action[5]: ankle_3_torque, 第三个踝关节扭矩，范围 [-1, 1]
- action[6]: hip_4_torque, 第四个髋关节扭矩，范围 [-1, 1]
- action[7]: ankle_4_torque, 第四个踝关节扭矩，范围 [-1, 1]

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止（如到达目标）。但环境中唯一正常的“存活到截断”可视为软成功，因为 agent 未因失败而终止。
- failure-like termination: body_height_outside_healthy_range（高度 <0.2 或 >1.0，很可能指示摔倒/崩溃或跳起）；state_value_outside_finite_range（数值发散/崩溃）。
- ambiguous termination: 无。
- truncation: time_limit_reached（步数耗尽），属于正常截断，不与任何具体失败绑定。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 为空字典，禁止任何 info 字段）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 均禁止使用。info 中无 termination_reason 标志。
- 推断路径（derived_possible）：
  - **失败（摔倒/崩溃）**：可从 body_z（obs[0]）突然低于 0.2 或 body_z_velocity 突变、身体 quat 指示大幅度倾斜（body_up_z 远小于 1）等组合中检测。但这些信号在终止时间步可能仍在过渡中，可靠性中等。
  - **截断为“存活到时间结束”**：若 episode 被 truncated 且摔倒信号未触发，则可推断为存活完成。
  - **前进任务的“成功”**：不能被单步信号直接判断，只能在 episode 结束后根据存活时间长度与正向速度累积来推断，但无法在 reward 的 step 计算内访问 episode 全局。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，但不要在步奖励中依赖未来或历史状态）
- action
- next_obs
- training_progress（仅用于退火或 schedule，如果训练框架允许，但谨慎使用以避免环境特定依赖）

禁止使用：
- original_reward（被掩码的官方奖励）
- 任何 info 字段（即使环境中存在也不允许）
- x_position / y_position 或任何全局位置信号
- official_reward 的任何成分

## 7. 可用于奖励函数的信号
- position: body_z (obs[0]) – 机体高度
- orientation: quat_w,x,y,z (obs[1:5]) – 可从四元数计算 body_up_z ≈ 1 – 2*(x²+y²)，衡量机体直立程度
- velocity (body): body_x_velocity (obs[13]) – 世界 x 轴前进速度（主进度信号）；body_y_velocity (obs[14]) – 侧向速度（可用于惩罚侧滑）；body_z_velocity (obs[15]) – 垂直速度（与摔倒/健康相关）；roll/pitch/yaw angular velocity (obs[16:19]) – 角速度稳定性信号
- velocity (joint): joint_1_velocity ~ joint_8_velocity (obs[19:27]) – 关节角速度
- joint angles: joint_1_angle ~ joint_8_angle (obs[5:13]) – 关节位置
- action/engine: 动作扭矩 action[0:8]，可用于动作幅度平滑惩罚
- other: 通过 body_z 和 quat 组合可间接推断摔倒/健康；通过 body_up_z 可推断倾斜惩罚；通过 body_x_velocity 可反映前进进度

## 8. 不确定或不可用的信号
- 显式成功/失败标志：不可用
- 全局 x 坐标：不可用，无法知道绝对前进距离，也无法直接奖励“完成一段行程”
- 接触力/接触标志：不可用（版本无接触信息）
- 能量/燃料消耗：没有直接功率或扭矩平方和信号，但可通过 action 幅度间接近似
- 来自 info 的任何项：完全禁止
- 速度命令/目标速度：未提供，只能观察实际速度

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 3D quadruped with rigid body and four legs
  actuator_type: torque control (8 joints: 4 hips + 4 ankles)
  contact_structure: implicit ground contact via MuJoCo; no explicit contact sensors
primary_objectives:
  - maintain body height in (0.2, 1.0)
  - maintain upright body orientation (body_up_z close to 1.0)
  - produce sustained forward velocity (body_x_velocity > 0)
secondary_objectives:
  - maximize forward speed
  - penalize excessive lateral / vertical velocity (drift, bobbing)
  - smooth joint actuation (keep torques moderate)
main_failure_risks:
  - falling below height threshold (collapse)
  - exceeding height threshold (launch/unstable jump)
  - large tilt leading to unrecoverable orientation
  - numerical instability leading to NaN/Inf state
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_progress
  purpose: 通过 body_x_velocity 直接反馈前进进度
  why_required: 任务核心是前进，且观测提供明确前进轴速度。该信号连续、密集，是主信号的天然候选。
  usable_signals: [next_obs[13] (body_x_velocity)]
  risks:
    - 若 robot 倾倒但身体仍向前滑动，body_x_velocity 可能维持正数，造成奖励空洞；需与健康门控联合使用。
    - 过早 reward 较大时，agent 可能只追求速度而牺牲稳定性，需正交惩罚介入。

- role_id: healthy_height_gate
  purpose: 根据 body_z 是否在安全范围 (0.2, 1.0) 内，生成一个在 [0,1] 的门控因子，与前进奖励相乘；一旦高度接近边界，进度奖励迅速衰减。
  why_required: 防止 forward_velocity 在非健康状态（尤其是接近跌倒）时仍产生强正反馈，避免 policy 学会“倒前冲一把”或“低爬滑行”。
  usable_signals: [next_obs[0] (body_z)]
  risks:
    - 边界计算需要平滑（如 hinge），否则可能导致数值梯度断裂。
    - 门控太严格会使早期探索无法获得任何奖励，需初始宽容区。

- role_id: upright_penalty
  purpose: 用 body_up_z = 1 – 2*(quat_x²+quat_y²) 衡量的直立程度，当低于阈值时施加 hinge penalty。
  why_required: 方向是保持平衡的关键部分；纯高度门控无法防止机身大幅度俯仰/翻滚，因为躯干高度可能仍在范围内但已大幅倾斜，导致不稳定步态。
  usable_signals: [next_obs[1:5] (quat)]
  risks:
    - 需要使用 hinge，避免在安全倾角范围内无用惩罚干扰步态自然微调。
    - 阈值需经验设定（如 body_up_z < 0.8 时惩罚）。

### 10.2 条件职责 conditional_roles
- role_id: lateral_drift_penalty
  condition_to_use: 如果侧向速度显著或任务希望走直线（task_description 未明确禁止侧向，但“walk or run forward”暗示单向）。
  usable_signals: [next_obs[14] (body_y_velocity)]
  risks:
    - 太强的惩罚可能妨碍转弯的自然步态（但本环境无转弯需求，风险低）。
    - 使用绝对值或平方惩罚时注意幅度缩放。

- role_id: action_smoothness
  condition_to_use: 如果我们期望自然、低抖动的动作，可加入动作变化量惩罚（当前动作与上一步动作之差的范数）。
  usable_signals: [action (8D)]
  risks:
    - 需要在函数内维护上一动作（增加状态需求），或使用 next_obs 的 joint velocity 间接反映。
    - 过量惩罚会降低响应速度和探索能力，权重需轻。

- role_id: joint_velocity_regularization
  condition_to_use: 如果训练早期关节振荡剧烈，可以对 joint_velocity 的 L2 范数进行轻量惩罚。
  usable_signals: [next_obs[19:27]]
  risks:
    - 过度惩罚会抑制步态的大幅摆动，不利于高速行走。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: energy_efficiency
  reason: 没有直接的功率或扭矩平方和与能量读数，只能从 action 幅度推断。但 task_description 未强调“省能”，且早期能量惩罚易与前进探索冲突，因此不作为显式 role，但可通过 action_smoothness 间接轻度贡献。
  forbidden_or_missing_signals: [能量/功率直接测量信号不此版本]
- role_id: target_reaching / distance_to_goal
  reason: 没有目标位置或距离读数；x 坐标不可用。
  forbidden_or_missing_signals: [x_position 禁止使用]
- role_id: foot_contact_reward
  reason: 无 contact 信息，无法判断脚是否触地。
  forbidden_or_missing_signals: [contact flag 不存在]
- role_id: survival_time 作为主信号
  reason: 虽然存活是必要条件，但任务明确要求持续的向前运动，仅存活不足以满足任务。
  usable_signals: 可间接推断存活，但不应作为主进度信号。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_velocity_progress | next_obs[13] (body_x_velocity) | – | dense_state_signal, bounded_signal, linear or log-modulated | 核心进度信号，需与 health gate 相乘 |
| healthy_height_gate | next_obs[0] (body_z) | – | hinge_gate, deadzone_gate | 输出 0~1 gate，接近边界衰减，与 velocity 相乘 |
| upright_penalty | next_obs[1:5] → derived body_up_z | – | hinge_penalty
