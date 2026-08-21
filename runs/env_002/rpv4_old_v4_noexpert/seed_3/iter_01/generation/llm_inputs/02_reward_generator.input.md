# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个匿名 2D 双足运动任务。智能体控制一个两腿躯干，通过髋、膝共 4 个连续力矩动作，在不规则地形上稳定向前行走。任务主目标是尽可能远、尽可能快地前进；必要健康约束是保持身体不倒，因为摔倒会立即终止回合。次要目标是降低能量消耗或关节力矩成本。

该任务不应被理解为纯平衡任务：保持平衡是完成前进目标的条件，而不是最终目标。也不应把“站立不动但保持平衡”当作成功。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

confidence: high

reason: 核心目标是控制腿部关节产生连续步态，在不规则地形上持续前进；动作是连续力矩；没有固定到达点，没有抓取操作，也没有多个权重相当且冲突的核心目标。摔倒只是失败/终止约束，不构成独立的多目标核心。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: 未显式给出，按常规推断为 float32
- obs[0]: hull_angle，躯干相对竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，躯干角速度，reward_usable: true
- obs[2]: horizontal_velocity，前向/后向线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，leg1 髋关节角度，reward_usable: true
- obs[5]: hip1_speed，leg1 髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，leg1 膝关节角度，reward_usable: true
- obs[7]: knee1_speed，leg1 膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，leg1 接地标志，1.0=接地，0.0=未接地，reward_usable: true
- obs[9]: hip2_angle，leg2 髋关节角度，reward_usable: true
- obs[10]: hip2_speed，leg2 髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，leg2 膝关节角度，reward_usable: true
- obs[12]: knee2_speed，leg2 膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，leg2 接地标志，1.0=接地，0.0=未接地，reward_usable: true
- obs[14..23]: lidar_0..9，10 个前方地形 LIDAR 测距读数，reward_usable: true，但属于间接/地形预览信号，使用需谨慎

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: 每个关节 [-1.0, 1.0]
- action_dim 0: hip_torque_leg1，施加在 leg1 髋关节的力矩
- action_dim 1: knee_torque_leg1，施加在 leg1 膝关节的力矩
- action_dim 2: hip_torque_leg2，施加在 leg2 髋关节的力矩
- action_dim 3: knee_torque_leg2，施加在 leg2 膝关节的力矩

动作是连续力矩控制，不存在离散动作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain，到达地形终点，可能表示成功，但未提供显式成功标志。
- failure-like termination: body_fallen_over，躯干摔倒，任务失败，应被惩罚。
- ambiguous termination: step 中 terminated=True 可能来自 body_fallen_over 或 reached_end_of_terrain；info={} 为空，无法直接从 info 区分原因。
- truncation: 给定 step 返回 truncated=False；未显示外部时间限制。不过真实运行环境可能包了 TimeLimit wrapper，但当前信息中不可依赖。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无 / []
- forbidden_or_uncertain_info_fields: success、failure、termination_reason、reached_goal、any info key not explicitly declared

必须注意：终止原因只能从下一状态或状态轨迹推断，不能假设 info 中存在成功/失败字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段；本环境当前为空，因此 info 不应使用
- training_progress 仅在外部 prompt 明确允许时才使用；当前 prompt 未明确允许，因此默认禁止使用

禁止使用：
- original_reward
- official_reward
- 任何未声明的 info 字段
- 任何未声明的 obs 切片
- 不得尝试重构被 mask 的官方 reward

## 7. 可用于奖励函数的信号
- position:
  - 无绝对 root x/y 位置
  - 无累计行进距离
  - lidar 只提供前方地形距离，不提供绝对位置
- velocity:
  - horizontal_velocity，obs[2]
  - vertical_velocity，obs[3]
  - hull_angular_velocity，obs[1]
  - hip1_speed，obs[5]
  - knee1_speed，obs[7]
  - hip2_speed，obs[10]
  - knee2_speed，obs[12]
- orientation:
  - hull_angle，obs[0]
- contact:
  - leg1_contact，obs[8]
  - leg2_contact，obs[13]
- action/engine:
  - action[0..3] 力矩向量，可用于能量/平滑惩罚
- other:
  - hip1_angle, knee1_angle, hip2_angle, knee2_angle
  - lidar_0..9，obs[14..23]，可用于地形感知 shape，但不是直接前进目标信号

## 8. 不确定或不可用的信号
- 绝对位置 / 已行进距离
- explicit success flag
- explicit failure flag
- 地面反作用力或摩擦系数
- 真实能量消耗或执行器功率；可用 torque * joint_speed 近似，但未直接给出
- 参考步态轨迹或目标关节角度序列
- 地形的完整高度图；仅有前方 LIDAR 读数
- info 中的任何字段，当前 info 为空

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: planar two-legged walker with torso and two legs
  actuator_type: continuous torque motors at hip and knee joints
  contact_structure: two leg ground contact flags (leg1_contact, leg2_contact)
primary_objectives:
  - maximize forward progress over uneven terrain (as far/fast as possible)
  - maintain stable bipedal gait and avoid falling
secondary_objectives:
  - minimize energy consumption / torque cost
main_failure_risks:
  - torso pitch instability leading to fall
  - stationary or too-slow gait due to over-penalizing action/energy
  - jerky or aggressive gait that wastes energy or destabilizes on uneven terrain
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 奖励智能体沿前进方向移动的速度或正向位移代理。
  why_required: 任务主目标是尽可能远、尽可能快前进。
  usable_signals: [horizontal_velocity, next_obs[2]]
  risks:
    - 只奖励速度可能诱发震荡或摔倒，必须与稳定性职责配合。
    - 没有绝对位置，无法直接奖励累计距离。
    - 需要校准 horizontal_velocity 的正方向是前进方向。

- role_id: balance_and_fall_prevention
  purpose: 惩罚躯干过度倾斜、剧烈旋转或异常垂直运动，避免摔倒。
  why_required: 摔倒会终止回合，阻碍前进目标。
  usable_signals: [hull_angle, hull_angular_velocity, vertical_velocity]
  risks:
    - 对小幅 hull_angle 惩罚过重会限制正常步态。
    - 只用 terminal fall 进行奖励不可行，因为没有显式 fall flag。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 在已能驱动前进后加入；或检测到动作过大、能耗过高时启用。作为低权重次级正则，不应压过前进目标。
  usable_signals: [action[0..3], joint_speeds]
  risks:
    - 权重过高会导致智能体站立不动或迈步过小。
    - 与 forward_progress 冲突时，必须降低该职责权重。

- role_id: gait_symmetry_or_contact_pattern
  condition_to_use: 当观察到一条腿闲置、拖行或步态明显不对称时启用。
  usable_signals: [leg1_contact, leg2_contact, hip1_angle, hip2_angle, knee1_angle, knee2_angle, joint_speeds]
  risks:
    - 强制对称可能限制自然步态。




# expert_reward_context.md

# Expert Schema Disabled

This run is the w/o Expert Schema ablation. Design from environment facts only; do not use the fixed expert task templates or formula-operator library.
