# 匿名环境理解卡片

## 1. 任务目标
这是一个连续控制运动任务。一个 3D 四足机器人拥有四条腿和八个扭矩控制关节，必须持续向前行走或奔跑并保持身体直立。机器人的身体高度必须保持在健康的范围（0.2 到 1.0）内；低于最小高度（趴下）或超过最大高度（弹射）均会终止回合。策略的主要目标是产生稳定、持续的前向运动，而不仅仅是保持直立不倒。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标是让四足机器人在平面地形上产生持续的前进运动。速度、姿态和高度等附属约束服务于这一前向移动主任务，不构成权重相当的多目标冲突。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: float32（默认连续环境）
- obs[0]: body_z，身体垂直高度，reward_usable: true（用于生存/高度约束）
- obs[1]: quat_w，身体朝向四元数 w 分量，reward_usable: true（用于计算直立投影）
- obs[2]: quat_x，四元数 x 分量，reward_usable: true（用于计算直立投影）
- obs[3]: quat_y，四元数 y 分量，reward_usable: true（用于计算直立投影）
- obs[4]: quat_z，四元数 z 分量，reward_usable: true（用于计算直立投影）
- obs[5]: joint_1_angle，第一个髋关节角度，reward_usable: true（用于姿态/能耗惩罚）
- obs[6]: joint_2_angle，第一个踝关节角度，reward_usable: true
- obs[7]: joint_3_angle，第二个髋关节角度，reward_usable: true
- obs[8]: joint_4_angle，第二个踝关节角度，reward_usable: true
- obs[9]: joint_5_angle，第三个髋关节角度，reward_usable: true
- obs[10]: joint_6_angle，第三个踝关节角度，reward_usable: true
- obs[11]: joint_7_angle，第四个髋关节角度，reward_usable: true
- obs[12]: joint_8_angle，第四个踝关节角度，reward_usable: true
- obs[13]: body_x_velocity，世界坐标系 x 方向线速度（前进速度），reward_usable: true（前向进度主信号）
- obs[14]: body_y_velocity，世界 y 方向线速度（侧向速度），reward_usable: true（侧向漂移惩罚）
- obs[15]: body_z_velocity，世界 z 方向线速度（垂直速度），reward_usable: true（降低震动）
- obs[16]: body_roll_velocity，滚转角速度，reward_usable: true（姿态稳定）
- obs[17]: body_pitch_velocity，俯仰角速度，reward_usable: true
- obs[18]: body_yaw_velocity，偏航角速度，reward_usable: true（保持方向）
- obs[19]: joint_1_velocity，第一个髋关节角速度，reward_usable: true（能耗惩罚）
- obs[20]: joint_2_velocity，第一个踝关节角速度，reward_usable: true
- obs[21]: joint_3_velocity，第二个髋关节角速度，reward_usable: true
- obs[22]: joint_4_velocity，第二个踝关节角速度，reward_usable: true
- obs[23]: joint_5_velocity，第三个髋关节角速度，reward_usable: true
- obs[24]: joint_6_velocity，第三个踝关节角速度，reward_usable: true
- obs[25]: joint_7_velocity，第四个髋关节角速度，reward_usable: true
- obs[26]: joint_8_velocity，第四个踝关节角速度，reward_usable: true

注：所有维度均可用于奖励函数。

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- continuous：每维取值范围 [-1.0, 1.0]，对应归一化扭矩指令
- action 0: hip_1_torque，第一个髋关节扭矩
- action 1: ankle_1_torque，第一个踝关节扭矩
- action 2: hip_2_torque，第二个髋关节扭矩
- action 3: ankle_2_torque，第二个踝关节扭矩
- action 4: hip_3_torque，第三个髋关节扭矩
- action 5: ankle_3_torque，第三个踝关节扭矩
- action 6: hip_4_torque，第四个髋关节扭矩
- action 7: ankle_4_torque，第四个踝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确的成功终止。回合通常因时间截断 (truncation) 结束，若在截断前未触发失败终止，可视为存活至结束，但并不保证已完成前向移动目标。
- failure-like termination:
  - body_height_outside_healthy_range：身体高度 z 不在 (0.2, 1.0) 范围内（趴下或飞起），对应失效。
  - state_value_outside_finite_range：观测中出现 NaN 或 inf，数值不稳定，应为失败。
- ambiguous termination: 无机械定义的模糊终止。
- truncation: 时间限制到达，属于正常结束，存活且未触发以上失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 字典为空且不允许使用）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，包括任何官方奖励项 reward_forward、reward_ctrl 等以及 x/y 位置等。终止原因无法从 info 中提取。

备注：尽管没有显式 success/failure 标记，但可以通过观测到的 next_obs[0] (body_z) 是否在 (0.2,1.0) 内以及数值是否有限来推断失败终止，但这需要 reward 函数在训练时能访问 terminated 标志（即使如此，此处分析仅基于允许使用的 obs/action/next_obs/info，不能假设 terminated 可作为输入）。在 stateless reward 设计中无法直接使用 termination 信号，但可以通过 body_z 和有限性检查接近该信息。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
允许使用：
- `obs`：全部 27 维观测均可使用。
- `action`：全部 8 维动作均可使用。
- `next_obs`：与 obs 同结构的观测，可用于计算变化或检查终止条件相关的信号（如高度、有限性）。
- `info`：不允许使用任何字段（info 为空且被禁止）。
- `training_progress`：仅在 prompt 明确允许时使用，本环境未提及，故谨慎使用或不用。

禁止使用：
- `original_reward`（已遮蔽）。
- 任何来自 info 的官方奖励项或位置信息。
- 任何未在上述观察空间中声明的隐含信息。

## 7. 可用于奖励函数的信号
- 位置/姿态信号：
  - body_z (obs[0])：可作为生存/高度奖励的基础。
  - 姿态直立度：从四元数 obs[1:5] 可计算 body_up_z = 1 - 2*(quat_x² + quat_y²)，值域 [-1, 1]，1 表示完全直立。可用于 orientation 奖励。
- 速度信号：
  - body_x_velocity (obs[13])：前进速度，直接反映任务进度。
  - body_y_velocity (obs[14])：侧向速度，可用于惩罚漂移。
  - body_z_velocity (obs[15])：垂直速度，惩罚跳跃或剧烈振动。
  - body_roll_velocity (obs[16])、body_pitch_velocity (obs[17])、body_yaw_velocity (obs[18])：惩罚姿态晃动。
- 关节角度/角速度信号：
  - 关节角度 obs[5:13]：可惩罚过大关节偏离。
  - 关节角速度 obs[19:27]：可惩罚抖动或高能耗。
- 动作/控制信号：
  - action[0:8]：扭矩指令，可惩罚大幅动作、动作变化率（若结合上一动作）或直接作为能耗惩罚。
- 其他：可通过 next_obs 推断是否进入危险区域（body_z 过低/过高）进行惩罚，但注意 reward 是 step-wise 且无累积状态。

## 8. 不确定或不可用的信号
- 绝对水平位置（x, y）：不可用，禁止使用。
- 接触力/触地信息：不可用，环境中无相关观测。
- 官方奖励分量（reward_forward, reward_ctrl 等）：不可用，禁止。
- 明确的 success/failure 标记：不可用，info 为空。
- 躯干俯仰/滚转角度：不能直接从 27 维观测中获得欧拉角（除非从四元数转换，但需要额外计算），但可间接通过四元数投影获得 body_up_z。
- 动作变化率（jerk）：需要存储上一动作，而 compute_reward 为无状态函数，无法直接获取，除非将上一动作作为额外状态传入，但这不是标准接口。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 3D quadruped (four legs)
  actuator_type: eight torque-controlled joints (hip + ankle per leg)
  contact_structure: legs interact with ground, but no contact force/state observable
primary_objectives:
  - maximize forward velocity (body_x_velocity) while remaining stable
  - maintain body height within healthy range [0.2, 1.0] to avoid termination
  - keep body upright (body_up_z close to 1)
secondary_objectives:
  - minimize lateral drift (body_y_velocity magnitude)
  - suppress excessive body angular rates (roll, pitch, yaw) for smooth gait
  - reduce energy consumption (action magnitudes, joint velocities)
  - avoid large joint angle deviations that could lead to instability
main_failure_risks:
  - body_z falls below 0.2 (collapse), or exceeds 1.0 (launch)
  - orientation flips (body_up_z near -1) leads to upside-down state then often triggers height violation
  - numerical divergence (NaNs/infs) due to unstable joint configurations
  - agent learns to stay still to avoid termination but fails to make forward progress
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward  
  purpose: 激励机器人向前移动，直接体现任务核心目标。  
  why_required: 任务要求稳定前向运动，如果没有这个奖励，智能体可能学会静止不动以保持存活。  
  usable_signals: [obs[13] body_x_velocity]  
  risks: 若奖励设计不当（如单纯使用原始速度），可能导致机器人采用危险的跑步策略（易翻倒）或学习“冲刺然后摔倒”的局部最优。需要与其他稳定职责结合。

- role_id: survival_and_height_reward  
  purpose: 维持身体高度在健康区间内，避免摔倒或飞起终止。  
  why_required: 终止条件直接与高度相关，必须让智能体学会保持安全高度，否则无法存活。  
  usable_signals: [obs[0] body_z]  
  risks: 若只奖励待在某个固定高度，可能抑制前向运动，需要设计为仅惩罚偏离安全区的部分。

- role_id: upright_orientation_reward  
  purpose: 保持躯干大致直立，防止翻转。  
  why_required: 翻倒会迅速导致高度违规或失去前进能力；直立是稳定运动的基础。  
  usable_signals: [obs[1:5] quaternion; 可计算 body_up_z = 1 - 2*(quat_x² + quat_y²)]  
  risks: 四足动物在快速奔跑时可能有自然的俯仰，对直立约束过严可能妨碍高速步态，可通过适度容忍稍低的 body_up_z 或分级奖励缓解。

### 10.2 条件职责 conditional_roles
- role_id: lateral_drift_penalty  
  purpose: 惩罚侧向移动，使机器人沿正前方行走。  
  condition_to_use: 当观测到 body_y_velocity 偏离 0 时作用。最好在中后期加入，早期可放宽以保证探索。  
  usable_signals: [obs[14] body_y_velocity]  
  risks: 若惩罚过重可能限制机器人必要的侧向平衡调整步态，导致僵硬行走，需要系数调优。

- role_id: angular_rate_penalty  
  purpose: 减小身体滚转、俯仰、偏航角速度，让运动更平滑，降低摔倒风险。  
  condition_to_use: 始终可用，但初期可降低权重，待基本行走出现后再加强。  
  usable_signals: [obs[16] body_roll_velocity, obs[17] body_pitch_velocity, obs[18] body_yaw_velocity]  
  risks: 过大会抑制动态平衡所需的微小角速度，使运动过于僵硬；偏航角速度惩罚在需要转弯时可能不合适，但本任务只需直线前进故无问题。

- role_id: action_regularization  
  purpose: 降低扭矩指令能量，鼓励节能、连贯的动作。  
  condition_to_use: 当基本前进行为建立后加入更有效，早期可能阻碍探索。  
  usable_signals: [action[0:8]]  
  risks: 太