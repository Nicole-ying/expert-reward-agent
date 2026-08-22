# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主目标：让双足机器人在崎岖地形（含梯子、树桩、坑洞、不平表面）上尽可能快地稳健前进，最终到达地形尽头。次目标：在保证稳定不摔倒的前提下，最小化关节力矩消耗。不应混滑的目标：无需精确到达特定坐标，无需抓取物体，也无需保持绝对静止；地形只是阻碍前进的扰动，不应被当作需要精确避开的多目标之一。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心要求是双足机器人跨越崎岖地面持续前进，没有指定的目标点，只有“尽可能远和高效”的进度目标；地形多样性只是增加了控制的难度，附属的能量节省不构成任务族切换。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推断)
- obs[0] hull_angle： 身体倾角，reward_usable: true
- obs[1] hull_angular_velocity： 身体角速度，reward_usable: true
- obs[2] horizontal_speed： 质心水平速度，reward_usable: true
- obs[3] vertical_speed： 质心垂直速度，reward_usable: true
- obs[4] joint_0_angle： 髋关节1角度，reward_usable: true (可能用于姿态约束)
- obs[5] joint_0_speed： 髋关节1角速度，reward_usable: true (可能用于平滑/冲击惩罚)
- obs[6] joint_1_angle： 膝关节1角度，reward_usable: true
- obs[7] joint_1_speed： 膝关节1角速度，reward_usable: true
- obs[8] joint_2_angle： 髋关节2角度，reward_usable: true
- obs[9] joint_2_speed： 髋关节2角速度，reward_usable: true
- obs[10] joint_3_angle： 膝关节2角度，reward_usable: true
- obs[11] joint_3_speed： 膝关节2角速度，reward_usable: true
- obs[12] leg_1_ground_contact： 腿1地面接触标志（1.0 接触，0 未接触），reward_usable: true
- obs[13] leg_2_ground_contact： 腿2地面接触标志，reward_usable: true
- obs[14..23] lidar_1..lidar_10： 前方地形高度测量（激光雷达），reward_usable: true (用于预判障碍，但当前不作为主奖励)

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- bounds: [-1.0, 1.0]
- action_dim 0: hip_1_torque，施加给第一个髋关节的力矩
- action_dim 1: knee_1_torque，施加给第一个膝关节的力矩
- action_dim 2: hip_2_torque，施加给第二个髋关节的力矩
- action_dim 3: knee_2_torque，施加给第二个膝关节的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain (到达地形尽头，简称为“成功到达”)
- failure-like termination: body_fallen_over (身体摔倒)
- ambiguous termination: 无
- truncation: 当前环境可能没有最大步数截断，或因超时截断但未被明确列出，暂不考虑

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（info 恒为空）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs: 当前观测数组
- action: 当前动作数组
- next_obs: 下一时刻观测数组
- info: 仅当明确允许其字段时才可使用，本环境 info 恒为空，故实质上禁止使用
- training_progress: 当前 prompt 未声明允许使用，故禁止

禁止使用：
- original_reward
- official_reward
- info 中任何字段（因为 info 为空）
- 未声明的任何其他变量

## 7. 可用于奖励函数的信号
- position/velocity:
  - horizontal_speed (obs[2], next_obs[2]): 可直接作为前进奖励
  - vertical_speed (obs[3], next_obs[3]): 可用于惩罚异常弹跳
- orientation:
  - hull_angle (obs[0], next_obs[0]): 可用于惩罚倾斜，间接预防摔倒
  - hull_angular_velocity (obs[1], next_obs[1]): 可用于惩罚急剧旋转
- joint state:
  - joint_*_angle / joint_*_speed: 可用于姿态约束或关节冲击惩罚
- contact:
  - leg_1_ground_contact, leg_2_ground_contact: 可用于步态健康约束（避免单腿停留过久或双腿同时离地），或检测摔倒征兆
- action/engine:
  - hip_1_torque .. knee_2_torque: 力矩大小可用于能效惩罚
- other:
  - lidar_1..lidar_10: 可用于预判前方陡坡，但更适合作为策略输入，作为奖励信号用途有限
- 间接推断（derived_possible）：
  - 摔倒可被推断：当 hull_angle 超过某个阈值（如 1.0 rad），或 vertical_speed 负向过大且接触信号突变，或 hull_angular_velocity 异常时，大概率已摔倒。可设计一个 penalty 但不依赖终止状态本身。
  - 到达终点可被推断：如果在连续前进过程中 episode 突然 truncated 且未检测到明显摔倒信号，可能意味着到达终点。但无法在单步奖励中准确获得该事件，仅可用于 hindsight 分析，不适合做实时单步奖励。

## 8. 不确定或不可用的信号
- info 字段全部不可用：无法获得 rewards、success、failure、distance_to_goal 等
- 无法获得精确的全局位置或里程计（除非从速度积分，但噪声大）
- 无法获得地形类别或障碍物类型
- 无法获得能量、接触力等附加物理量

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal_agent
  actuator_type: torque_controlled_rotational_joints
  contact_structure: two_legs_with_ground_contact_signals
primary_objectives:
  - make fast and stable forward progress across irregular terrain
secondary_objectives:
  - minimize joint torque consumption
main_failure_risks:
  - falling over on rough ground
  - getting stuck in pits or against tree stumps
  - wasting energy by excessive joint torque
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress_reward
  purpose: 鼓励机器人向前移动，奖励水平速度
  why_required: 这是任务核心，缺少会导致机器人原地踏步或后退。
  usable_signals: [horizontal_speed (next_obs[2])]
  risks: 若权重过高可能导致机器人用不安全的高速冲刺，增加摔倒风险；需结合姿态约束。

- role_id: posture_stability_penalty
  purpose: 惩罚身体过度倾斜和快速旋转，以避免摔倒
  why_required: 摔倒直接导致终止且无法继续前进，是主要失败模式。
  usable_signals: [hull_angle (next_obs[0]), hull_angular_velocity (next_obs[1])]
  risks: 若惩罚过强可能抑制必要的身体摆动，导致步态僵硬，反而不利于越障；需适度。

### 10.2 条件职责 conditional_roles
- role_id: joint_effort_penalty
  condition_to_use: 训练后期或已有稳定步态后，为提升效率而启用，训练初期可关闭或衰减
  usable_signals: [action (hip_1_torque..knee_2_torque)]
  risks: 早期加入可能阻碍探索出大幅度的越障动作，导致 stuck 在局部最优。

- role_id: vertical_bounce_penalty
  condition_to_use: 当检测到 vertical_speed 过大或频繁上下波动时启用，以减少多余弹跳
  usable_signals: [vertical_speed (next_obs[3])]
  risks: 过度惩罚可能妨碍跳过小障碍，建议仅对负面弹跳或不合理的高频垂直运动施加微调

### 10.3 慎用/禁用职责 avoid_roles
- role_id: step_pattern_constraint
  reason: 虽然接触标志可用，但强制规定“左右交替”步态可能限制应对复杂地形的自然非对称步态，且没有证据表明不交替会导致失败；不适合早期强制
  forbidden_or_missing_signals: 没有步态周期的标记，仅凭接触信号难以设计稳健的步态约束

- role_id: termination_based_completion_reward
  reason: 无法从 info 获得确切的成功/失败标志，且终止事件不能直接用于单步 reward 反馈（会导致稀疏 reward）；到达终点的信号也无法可靠推导，故排除
  forbidden_or_missing_signals: 缺少 explicit_success_flag

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress_reward | next_obs[2] (horizontal_speed) | None | dense_state_signal, bounded_signal (可clip) | 简单线性或对数映射 |
| posture_stability_penalty | next_obs[0] (hull_angle), next_obs[1] (hull_angular_velocity) | None | quadratic_penalty, hinge_penalty | 超过阈值再施加惩罚，避免干扰小振幅自然摆动 |
| joint_effort_penalty | action (all 4 dims) | None | squared_sum, bounded_signal (loss=sum(action^2)) | 建议在初始训练阶段关闭，后期逐步引入 |
| vertical_bounce_penalty | next_obs[3] (vertical_speed) | None | quadratic_penalty, if |v_speed|>threshold | 若地形允许小幅度跳跃，需谨慎选择阈值 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 频繁摔倒 (hull_angle 过大) | episode 长度短，平均回报低；hull_angle 轨迹频繁触及高值；接触信号频繁消失 | 增大 posture_stability_penalty 权重，或降低 forward_progress_reward 权重 |
| 机器人原地粘滞或前进极慢 | horizontal_speed 均值接近零，episode 很长但没有前进；lidar 读数几乎不变 | 检查是否因 penalty 过高抑制了动作；可暂时关闭 vertical_bounce_penalty，并降低 joint_effort_penalty |
| 用过高力矩换速度，能耗大 | action 各维度的绝对值常接近 1.0；动作平滑性差 | 适时加入 joint_effort_penalty 并逐步加码，或施加 joint_speed 变化率惩罚 |
| 只迈一边腿的僵化步态 | leg_1_contact 和 leg_2_contact 长时间不均，前进速度依然慢 | 当前不考虑步态惩罚，但可观察；若成为问题，可引入轻微的接触对称性鼓励（但需谨慎） |
