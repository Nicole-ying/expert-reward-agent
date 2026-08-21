# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 双足 locomotion 任务。智能体需要控制两条腿的髋关节和膝关节，在非平坦地形上持续、稳定地前进，尽可能走得远且快，同时尽量减少能量消耗。核心目标是保持稳定的双足步态并持续向前推进；摔倒会立即终止回合。能量消耗是明确但次级的附属目标，不应与“保持平衡、向前移动”的主目标混淆。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务核心是连续控制双足身体在地形上前进，没有目标点到达类目标；终止条件包含摔倒，但主任务仍是稳定持续前进；能量、姿态等属于附属优化，不是多目标冲突。

动力学子类型 dynamics_subtype: planar_bipedal_gait

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推测，连续 Box 常见类型；任务说明未显式给出，但所有字段均为连续数值）
- obs[0]: hull_angle，主躯干相对竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，主躯干角速度，reward_usable: true
- obs[2]: horizontal_velocity，前进/后退线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿 1 髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿 1 髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿 1 膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿 1 膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿 1 地面接触标志（1.0=接触，0.0=无接触），reward_usable: true
- obs[9]: hip2_angle，腿 2 髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿 2 髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿 2 膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿 2 膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿 2 地面接触标志（1.0=接触，0.0=无接触），reward_usable: true
- obs[14]: lidar_0，前方地形第 1 个 LIDAR 距离测量，reward_usable: true（是否用于奖励需谨慎）
- obs[15]: lidar_1，前方地形第 2 个 LIDAR 距离测量，reward_usable: true（是否用于奖励需谨慎）
- obs[16]: lidar_2，前方地形第 3 个 LIDAR 距离测量，reward_usable: true（是否用于奖励需谨慎）
- obs[17]: lidar_3，前方地形第 4 个 LIDAR 距离测量，reward_usable: true（是否用于奖励需谨慎）
- obs[18]: lidar_4，前方地形第 5 个 LIDAR 距离测量，reward_usable: true（是否用于奖励需谨慎）
- obs[19]: lidar_5，前方地形第 6 个 LIDAR 距离测量，reward_usable: true（是否用于奖励需谨慎）
- obs[20]: lidar_6，前方地形第 7 个 LIDAR 距离测量，reward_usable: true（是否用于奖励需谨慎）
- obs[21]: lidar_7，前方地形第 8 个 LIDAR 距离测量，reward_usable: true（是否用于奖励需谨慎）
- obs[22]: lidar_8，前方地形第 9 个 LIDAR 距离测量，reward_usable: true（是否用于奖励需谨慎）
- obs[23]: lidar_9，前方地形第 10 个 LIDAR 距离测量，reward_usable: true（是否用于奖励需谨慎）

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 连续动作，每个关节力矩范围为 [-1.0, 1.0]
- action[0]: hip_torque_leg1，施加在腿 1 髋关节的力矩
- action[1]: knee_torque_leg1，施加在腿 1 膝关节的力矩
- action[2]: hip_torque_leg2，施加在腿 2 髋关节的力矩
- action[3]: knee_torque_leg2，施加在腿 2 膝关节的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain，到达地形终点，可能表示任务完成
- failure-like termination: body_fallen_over，身体摔倒，回合结束
- ambiguous termination: 终止布尔量仅表示 `terminated = body_fallen_over or reached_end_of_terrain`，不能直接区分原因
- truncation: 任务说明中未显式给出 truncation 条件；step source 返回 `False` 作为 truncation，因此本环境没有显式时间截断信号

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无。step source 返回 `{}`，info 为空，不能从 info 获取 success/failure/termination_reason
- forbidden_or_uncertain_info_fields: 所有未在 step source 中显式出现的 info 字段，例如 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段。但本环境 info 恒为空，因此实际不应依赖 info
- training_progress 只有当后续 prompt 明确允许时才可用于课程/权重调整

禁止使用：
- original_reward
- official_reward
- 任何未在 step source 中出现的 info 字段
- 未声明的观测切片
- 不得试图恢复或复现 masked official reward

## 7. 可用于奖励函数的信号
- position: 无绝对 x/y 位置。没有世界坐标位置，也没有到终点的距离
- velocity:
  - horizontal_velocity：obs[2]，适合作为前进进度代理信号
  - vertical_velocity：obs[3]，可用于判断异常起跳/下坠，但不是主目标
  - hull_angular_velocity：obs[1]，可用于稳定约束
  - hip1_speed / knee1_speed / hip2_speed / knee2_speed：关节速度，可用于能耗或运动平滑
- orientation:
  - hull_angle：obs[0]，躯干倾角，稳定奖励核心信号
  - hip1_angle / knee1_angle / hip2_angle / knee2_angle：关节角度，可用于姿态约束
- contact:
  - leg1_contact：obs[8]
  - leg2_contact：obs[13]
- action/engine:
  - action[0..3]：四个关节力矩，用于能耗惩罚
- other:
  - lidar_0..9：obs[14..23]，前方地形距离，可用于地形感知，但奖励用法不确定，初期不建议强依赖

## 8. 不确定或不可用的信号
- 绝对世界坐标 x/y：不可用
- 前进距离或里程：不可用，没有显式位置观测
- 成功标志：不可用，info 为空，terminated 不区分原因
- 失败/摔倒标志：不可用，除非从 obs 中推理 hull_angle 等接近倒地状态
- 能量消耗的真实度量：不确定，只能通过 action 和关节速度近似
- 地形高度图：不确定，只能通过 LIDAR 距离间接推断
- 接触力/足部压力：不可用，只有 0/1 接触标志
- original_reward / official_reward：禁止使用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: planar_bipedal_torso_with_two_legs
  actuator_type: torque_controlled_continuous_hinge_joints
  contact_structure: two_leg_ground_contacts_binary
primary_objectives:
  - move_forward_across_uneven_terrain_as_far_and_fast_as_possible
  - maintain_stable_bipedal_gait_and_avoid_falling
secondary_objectives:
  - minimize_energy_consumption
  - produce_smooth_joint_control
main_failure_risks:
  - body_fallen_over
  - no_forward_progress_or_standing_still
  - excessive_hip_knee_torque_jitter
  - over_penalizing_stability_or_energy_causing_immobility
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 奖励持续向前的水平速度/移动趋势，推动智能体走得远且快
  why_required: 任务核心是前进，没有该职责智能体可能学会原地站立甚至静止
  usable_signals:
    - obs[2] horizontal_velocity
    - next_obs[2] horizontal_velocity
  risks:
    - 如果只用瞬时速度，可能产生前后振荡或冲刺式起伏
    - 速度项不宜无上界线性奖励，否则可能放大速度噪声

- role_id: upright_stability
  purpose: 惩罚主躯干过度倾斜或角速度过大，降低摔倒概率
  why_required: 摔倒直接终止回合，稳定是完成前进目标的必要健康约束
  usable_signals:
    - obs[0] hull_angle
    - obs[1] hull_angular_velocity
    - next_obs[0] hull_angle
    - next_obs[1] hull_angular_velocity
  risks:
    - 过度惩罚躯干倾斜可能让智能体采取僵硬直立策略，失去自然步态
    - 建议使用小角度死区和阈值外惩罚，而不是全程强二次惩罚

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency_penalty
  condition_to_use: 当智能体已经出现稳定前进步态后再加入；或训练后期从轻权重逐步增强
  purpose: 惩罚过大关节力矩，满足最小化能量消耗的次级目标
  usable_signals:
    - action[0..3] 关节力矩
    - 可选结合 obs 关节速度近似功率
  risks:
    - 早期加入过强能量惩罚会导致智能体选择不动或极小动作
    - 可能压制必要的探索性大动作，减慢学习

- role_id: terrain_aware_lidar_penalty
  condition_to_use: 如果实验发现智能体在非平坦地形频繁失足或撞障碍，可谨慎使用 LIDAR 做地形感知辅助
  purpose: 利用 LIDAR 距离信息帮助智能体感知前方地形变化，提前调整步态
  usable_signals:
    - obs[14..23] lidar_0..9
  risks:
    - LIDAR 测量存在噪声，可能引入虚假地形惩罚
    - 不是核心步态学习的必要信号，容易过度工程化

### 10.3 慎用/禁用职责 avoid_roles
- role_id: goal_distance_reward
  reason: 没有绝对位置或到终点距离观测，无法计算真实距离
  forbidden_or_missing_signals:
    - absolute_x_position
    - distance_to_goal

- role_id: success_bonus_on_termination
  reason: info 为空，terminated 无法区分 reached_end_of_terrain 和 body_fallen_over，不能可靠给予成功奖励
  forbidden_or_missing_signals:
    - explicit_success_flag
    - termination_reason

- role_id: foot_grounding_phase_reward
  reason: 虽然有 0/1 接触标志，但没有足够的步态相位历史；过早施加严格的接触相位奖励会干扰自然步态学习
  forbidden_or_missing_signals:
    - continuous_foot_force
    - gait_phase_label

- role_id: original_reward_proxy
  reason: 官方原始奖励被 masked，禁止复现或作为替代目标
  forbidden_or_missing_signals:
    - original_reward
    - official_reward

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | obs[2], next_obs[2] horizontal_velocity | absolute_x_position, distance_to_goal | dense_state_signal, bounded_signal | 主推进项，建议做幅度限制或死区，避免速度噪声奖励 |
| upright_stability | obs[0], obs[1], next_obs[0], next_obs[1] | explicit_fall_flag | quadratic_penalty, bounded_signal | 对 hull_angle 小偏差可忽略，大偏差惩罚 |
| energy_efficiency_penalty | action[0..3], obs[5], obs[7], obs[10], obs[12] | true_joint_power | quadratic_penalty, l2_norm, bounded_signal | 后训练条件使用，避免早期压制行动 |
| terrain_aware_lidar_penalty | obs[14..23] | terrain_height_map, obstacle_label | bounded_signal, threshold_penalty | 初期不建议依赖 |
| goal_distance_reward | 无 | absolute_position | 不适用 | 缺少位置观测，禁用 |
| success_bonus_on_termination | 无 | explicit_success_flag, termination_reason | 不适用 | info 为空，禁用 |
| foot_grounding_phase_reward | obs[8], obs[13] | continuous_foot_force, gait_phase_label | 不适用 | 慎用，可能破坏自然步态 |
| original_reward_proxy | 无 | original_reward | 不适用 | 禁止使用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 智能体站在原地不动或只轻微晃动 | 水平速度长期接近 0，回合未因摔倒终止但前进距离短 | 加强 forward_progress 角色；检查 action 是否过小；降低能量惩罚 |
| 频繁前倾或后仰摔倒 | hull_angle 绝对值经常剧烈增大，回合快速结束 | 增强 upright_stability 中的角度和角速度惩罚；加入角度死区 |
| 步态有速度但极不稳定，关节力矩抖动大 | 关节速度高频振荡，action 频繁达到饱和 | 加入轻量关节力矩平滑/能耗惩罚；降低前进速度线性奖励 |
| 不对称步态，一条腿主导 | 两腿髋/膝角度或接触标志长期不对称，单侧接触频繁 | 检查是否需要接触相关辅助；但不要过早强加固定步态相位 |
| 因平坦速度奖励学会前扑式前进而非稳健行走 | horizontal_velocity 提升但 hull_angle 长期偏大，经常接近摔倒 | 将 forward_progress 与 upright_stability 联合考虑；给前进速度添加要求躯干接近直立的门限 |
| 下坡或地形起伏时频繁失稳 | LIDAR 读数快速变化，hull_angle/vertical_velocity 异常 | 谨慎加入 LIDAR 地形感知惩罚或降低大步幅奖励 |
| 能量效率目标与前进目标冲突导致行为僵硬 | action 范数很小，步态频率极低，前进速度不足 | 后期再增强 energy_efficiency_penalty；前期保持轻量或零能量惩罚 |