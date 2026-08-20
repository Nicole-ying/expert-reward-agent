# 匿名环境理解卡片

## 1. 任务目标
这是一项 2D 双足移动任务。智能体需要控制两条腿的髋、膝共四个关节的连续力矩，驱动身体在不同地形上尽可能远、尽可能快地向前行走，同时尽量减少能量消耗。机器人必须保持直立，避免摔倒，因为摔倒将直接终止当前回合。次级目标包括：保持平稳步态、不出现过多不必要的跳跃或抖动、使用合理的关节力矩。该任务不能混淆为定点到达或生存平衡：核心驱动信号是前进速度，成功条件是到达地形尽头，而非存活时长。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务核心是在连续动作空间下持续向前通过地形，属于典型的连续控制移动任务。虽然包含能耗等次要目标，但“远、快前进”始终是第一目标，并非多目标权重相当，因此排除 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float32（连续值）  
- obs[0]: hull_angle – 本体与垂线的夹角（弧度），可用于惩罚倾斜，保持直立。reward_usable: true  
- obs[1]: hull_angular_velocity – 本体角速度，用于抑制旋转抖动。reward_usable: true  
- obs[2]: horizontal_velocity – 水平线速度（正向为前进），是主进度信号。reward_usable: true  
- obs[3]: vertical_velocity – 垂直线速度，可用于限制跳跃/过重落地。reward_usable: true  
- obs[4]: hip1_angle – 腿1髋关节角度。reward_usable: potentially  
- obs[5]: hip1_speed – 腿1髋关节角速度。reward_usable: potentially  
- obs[6]: knee1_angle – 腿1膝关节角度。reward_usable: potentially  
- obs[7]: knee1_speed – 腿1膝关节角速度。reward_usable: potentially  
- obs[8]: leg1_contact – 腿1是否接触地面（1.0/0.0），可用于步态协调。reward_usable: true  
- obs[9]: hip2_angle – 腿2髋关节角度。reward_usable: potentially  
- obs[10]: hip2_speed – 腿2髋关节角速度。reward_usable: potentially  
- obs[11]: knee2_angle – 腿2膝关节角度。reward_usable: potentially  
- obs[12]: knee2_speed – 腿2膝关节角速度。reward_usable: potentially  
- obs[13]: leg2_contact – 腿2是否接触地面。reward_usable: true  
- obs[14]~obs[23]: lidar_0~lidar_9 – 10个前方激光雷达测距值，反映地形轮廓，早期可不依赖。reward_usable: true，但不强制使用。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- continuous: true  
- bounds: 每个关节力矩限制在 [-1.0, 1.0]  
- action[0]: hip_torque_leg1 – 腿1髋关节力矩  
- action[1]: knee_torque_leg1 – 腿1膝关节力矩  
- action[2]: hip_torque_leg2 – 腿2髋关节力矩  
- action[3]: knee_torque_leg2 – 腿2膝关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain – 到达地形尽头，视为成功完成。  
- failure-like termination: body_fallen_over – 机器人摔倒，视为失败。  
- ambiguous termination: 无。  
- truncation: 环境中无内部步数截断（外部训练循环可能限定最大步数）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （step 返回 info 为空字典）  
- explicit_failure_flag_available: false （同上）  
- allowed_info_fields: [] （空字典，无任何允许字段）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均为非法，因为环境中不提供任何信息字段。  
注意：奖励函数无法访问 `terminated` 或 `done`，必须完全基于 `obs`、`next_obs`、`action` 推断状态好坏。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（当前观测数组）
- action（本次执行的动作数组）
- next_obs（下一观测数组，若可用）
- info：只能为空，不得假设任何键存在
- training_progress：仅当任务描述明确允许时可使用，此处无此类提示，建议不使用

禁止使用：
- original_reward（官方奖励被屏蔽，不可重现）
- 任何未在观测空间中明确声明的切片或字段
- 任何 info 字段（如 “success”、“failure”、“termination_reason” 等）
- 内部终止标志（done/terminated）

## 7. 可用于奖励函数的信号
- position: 无直接位移坐标，可通过对 horizontal_velocity 在时间上的积分近似，但奖励中通常直接使用速度信号。  
- velocity: next_obs[2]（或 obs[2]）horizontal_velocity 作为前进速度；obs[3]/next_obs[3] vertical_velocity 可用于惩罚跳跃。  
- orientation: next_obs[0] hull_angle 衡量姿态倾斜；next_obs[1] hull_angular_velocity 抑制旋转。  
- contact: next_obs[8] leg1_contact, next_obs[13] leg2_contact，用于步态协调。  
- action/engine: 本次动作 action[0:4] 可用于力矩惩罚（能耗）；也可使用动作变化量（需结合上一步动作，但接口未提供，可放弃）。  
- other: lidar 信号（next_obs[14:24] 或 obs[14:24]），可能用于感知前方地形，不强制使用。

## 8. 不确定或不可用的信号
- 真实的水平位移/位置：不可用。  
- 关节力矩限制以外的物理量（如接触力、摩擦力）：不可用。  
- 明确的成功/失败布尔标志：不可用。  
- 环境提供的原始奖励或任何额外信息字段：不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal, two legs with hip and knee joints