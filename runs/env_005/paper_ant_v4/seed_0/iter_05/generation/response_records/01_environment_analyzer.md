# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务是三维空间中的连续控制任务：四足机器人拥有 8 个扭矩可控关节，需要在前向方向上进行稳定、持续的行走或奔跑，同时保持身体直立，身体高度必须始终处于健康区间（0.2, 1.0）内，否则立即终止。  
主目标是实现**稳定前向运动**，而不仅仅是维持平衡或存活；次要目标包括通过合理动作保持身体姿态平稳、避免侧向漂移与剧烈旋转；高度维持是生存必要条件，不应作为奖励的核心驱动目标。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务说明明确核心是让四足机器人持续前进，而非仅仅站立或存活，高度限制只是终止条件；动作空间连续、观察空间含有丰富的前向速度与姿态信号，符合连续运动控制族特征；不涉及到达目标点、稀疏事件探索或多目标冲突。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32（通常）
- obs[0]: body_z, 身体垂直高度（主身体 z 坐标），可用作生存监测，不宜直接用于前进奖励，reward_usable: true（仅用于高度边界预警或姿态约束）
- obs[1]: quat_w, 身体朝向四元数实部，用于计算直立程度，reward_usable: true
- obs[2]: quat_x, 四元数虚部 x，参与 body_up_z 计算，reward_usable: true
- obs[3]: quat_y, 四元数虚部 y，参与 body_up_z 计算，reward_usable: true
- obs[4]: quat_z, 四元数虚部 z，reward_usable: true
- obs[5]: joint_1_angle, 第一髋关节角度，reward_usable: true（可用于动作平滑或关节限位惩罚）
- obs[6]: joint_2_angle, 第一踝关节角度，reward_usable: true
- obs[7]: joint_3_angle, 第二髋关节角度，reward_usable: true
- obs[8]: joint_4_angle, 第二踝关节角度，reward_usable: true
- obs[9]: joint_5_angle, 第三髋关节角度，reward_usable: true
- obs[10]: joint_6_angle, 第三踝关节角度，reward_usable: true
- obs[11]: joint_7_angle, 第四髋关节角度，reward_usable: true
- obs[12]: joint_8_angle, 第四踝关节角度，reward_usable: true
- obs[13]: body_x_velocity, 世界坐标系下的前向线速度，**主奖励信号**，reward_usable: true
- obs[14]: body_y_velocity, 侧向线速度，可用于惩罚横向漂移，reward_usable: true
- obs[15]: body_z_velocity, 垂直线速度，可用于检测上下波动，reward_usable: true（需谨慎，避免干扰前进）
- obs[16]: body_roll_velocity, 横滚角速度，可用于姿态稳定惩罚，reward_usable: true
- obs[17]: body_pitch_velocity, 俯仰角速度，可用于姿态稳定惩罚，reward_usable: true
- obs[18]: body_yaw_velocity, 偏航角速度，可用于防止旋转，reward_usable: true
- obs[19]: joint_1_velocity, 第一髋关节角速度，reward_usable: true（可与动作配合做平滑）
- obs[20]: joint_2_velocity, 第一踝关节角速度，reward_usable: true
- obs[21]: joint_3_velocity, 第二髋关节角速度，reward_usable: true
- obs[22]: joint_4_velocity, 第二踝关节角速度，reward_usable: true
- obs[23]: joint_5_velocity, 第三髋关节角速度，reward_usable: true
- obs[24]: joint_6_velocity, 第三踝关节角速度，reward_usable: true
- obs[25]: joint_7_velocity, 第四髋关节角速度，reward_usable: true
- obs[26]: joint_8_velocity, 第四踝关节角速度，reward_usable: true

注：所有观察均可用于奖励设计，但必须遵守接口约束，禁止使用 info 中的任何字段。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- 连续控制，每个维度范围 [-1.0, 1.0]
- action_dim 0: hip_1_torque，第一髋关节力矩
- action_dim 1: ankle_1_torque，第一踝关节力矩
- action_dim 2: hip_2_torque，第二髋关节力矩
- action_dim 3: ankle_2_torque，第二踝关节力矩
- action_dim 4: hip_3_torque，第三髋关节力矩
- action_dim 5: ankle_3_torque，第三踝关节力矩
- action_dim 6: hip_4_torque，第四髋关节力矩
- action_dim 7: ankle_4_torque，第四踝关节力矩

所有动作均为施加在关节上的扭矩，可直接在奖励中使用动作向量大小或变化量进行平滑/效率约束。

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: 无显式成功终止。任务本质是无限持续前进，成功通过长期存活与高前向速度间接体现，不存在“到达目标”或“完成阶段”的终止。
- failure‑like termination:  
  - body_height_outside_healthy_range：身体高度 z ∉ (0.2, 1.0)，通常表示摔倒或异常弹跳，属于失败终止。  
  - state_value_outside_finite_range：任意状态值非有限（NaN/Inf），系统崩溃，明确失败。
- ambiguous termination: 无模糊终止。
- truncation: 时间限制（time_limit_reached）导致截断，此截断在当前训练设置下是正常结束方式，不代表成功或失败，不应在奖励中直接使用。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
  （接口规定 info 中禁止所有字段，且终止信号不会传入奖励函数）
- allowed_info_fields: 无（空列表，任何 info 字段均不得使用）
- forbidden_or_uncertain_info_fields: 所有 info 字段（特别是 reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 明确被禁）

因此，奖励函数必须完全基于 obs 与 action 设计，不能依赖终止标志或环境提供的内部奖励项。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（来自当前步或 next_obs，具体看调用约定，通常 next_obs 可用于计算下一状态的特征，但必须以给定接口为准）
- action
- next_obs
- info 中明确允许的字段：无（不允许任何 info 字段）
- training_progress：仅当任务 prompt 明确要求时使用，本任务未提及，不可使用

禁止使用：
- original_reward（官方奖励已被屏蔽，不能引用）
- official_reward
- 任何 info 字段
- 未声明的 obs 切片（只能使用上述已列出的 27 维观察）

## 7. 可用于奖励函数的信号
- position:  
  - body_z（身体高度），可用作高度安全监控  
  - 身体朝向四元数 obs[1:5]，可计算 body_up_z = 1 - 2*(quat_x² + quat_y²) 衡量身体直立程度（1.0 为完全直立）  
  - 8 个关节角度，可用于姿态约束或关节限制惩罚
- velocity:  
  - body_x_velocity（前向速度）：主奖励源  
  - body_y_velocity（侧向速度）：用于惩罚  
  - body_z_velocity（垂向速度）：用于抑制剧烈上下运动  
  - 身体角速度（roll, pitch, yaw）：用于姿态稳定性惩罚  
  - 8 个关节角速度：用于运动平滑或能耗相关惩罚
- orientation: 四元数可提取 body_up_z，详见上述
- contact: 无接触力信息（本版本不提供）
- action/
