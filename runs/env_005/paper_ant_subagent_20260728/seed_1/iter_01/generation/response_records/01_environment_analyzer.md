# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个连续控制四足机器人运动的环境。主要目标是使四足机器人持续、稳定地向前行走或奔跑，同时保持身体在健康高度范围内（0.2 ~ 1.0），并维持身体基本直立。机器人不应仅仅满足于不倒，而是需要产生有效的向前位移。次要目标包括抑制横向滑移、垂直弹跳、过大的关节力矩或速度，以提升运动的平稳性和能效。没有预设的目标位置，成功由存活至时间截断并持续向前移动来体现，不存在显式的“到达终点”成功标志。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标是连续向前行走/奔跑，没有明确到达目标，附属高度、直立、平滑等均为约束或副产品，符合 locomotion_continuous_control 的特征。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32（推断）
- each index 的含义与 reward 可用性：

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | body_z | 身体质心垂直位置（高度） | true |
| 1 | quat_w | 身体姿态四元数实部 | true（需组合） |
| 2 | quat_x | 身体姿态四元数 x | true |
| 3 | quat_y | 身体姿态四元数 y | true |
| 4 | quat_z | 身体姿态四元数 z | true |
| 5 | joint_1_angle | 第1髋关节角度 | true |
| 6 | joint_2_angle | 第1踝关节角度 | true |
| 7 | joint_3_angle | 第2髋关节角度 | true |
| 8 | joint_4_angle | 第2踝关节角度 | true |
| 9 | joint_5_angle | 第3髋关节角度 | true |
| 10 | joint_6_angle | 第3踝关节角度 | true |
| 11 | joint_7_angle | 第4髋关节角度 | true |
| 12 | joint_8_angle | 第4踝关节角度 | true |
| 13 | body_x_velocity | 身体在世界系 x 轴的线速度（前进速度） | true |
| 14 | body_y_velocity | 身体在世界系 y 轴的线速度（横向速度） | true |
| 15 | body_z_velocity | 身体在 z 轴的速度（垂直速度） | true |
| 16 | body_roll_velocity | 身体滚转角速度 | true |
| 17 | body_pitch_velocity | 身体俯仰角速度 | true |
| 18 | body_yaw_velocity | 身体偏航角速度 | true |
| 19 | joint_1_velocity | 第1髋关节角速度 | true |
| 20 | joint_2_velocity | 第1踝关节角速度 | true |
| 21 | joint_3_velocity | 第2髋关节角速度 | true |
| 22 | joint_4_velocity | 第2踝关节角速度 | true |
| 23 | joint_5_velocity | 第3髋关节角速度 | true |
| 24 | joint_6_velocity | 第3踝关节角速度 | true |
| 25 | joint_7_velocity | 第4髋关节角速度 | true |
| 26 | joint_8_velocity | 第4踝关节角速度 | true |

注：所有观测均可用于奖励，没有禁止字段。四元数可转换为直立度量：`body_up_z = 1 - 2*(quat_x² + quat_y²)`，取值范围大约 `[-1,1]`，`1.0` 表示完美直立。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- continuous: true
- bounds: [-1.0, 1.0] 每维
- 动作维度含义：

| dim | name | meaning |
|-----|------|---------|
| 0 | hip_1_torque | 第1髋关节力矩 |
| 1 | ankle_1_torque | 第1踝关节力矩 |
| 2 | hip_2_torque | 第2髋关节力矩 |
| 3 | ankle_2_torque | 第2踝关节力矩 |
| 4 | hip_3_torque | 第3髋关节力矩 |
| 5 | ankle_3_torque | 第3踝关节力矩 |
| 6 | hip_4_torque | 第4髋关节力矩 |
| 7 | ankle_4_torque | 第4踝关节力矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无显式成功条件；时间截断（`truncated = time_limit_reached`）且未提前失败时可视为一种存活成功，但从终止信号角度不具有明确成功标志。
- **failure-like termination**: 身体高度超出 (0.2, 1.0) 范围（即跌落或弹起过高）；任意状态值变为 NaN 或 infinity。
- **ambiguous termination**: 无。
- **truncation**: 达到最大时间步，truncated 为 True，此时 episode 正常结束，不应视为失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields:  
  `reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive`, `x_position`, `y_position`, `distance_from_origin` 等均被明确禁止。  
  说明：虽然原始环境可能提供这些字段，但当前实验环境下已屏蔽，奖励函数中不得使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs`: 当前观测 (27,) 数组
- `action`: 当前动作 (8,) 数组
- `next_obs`: 下一观测 (27,) 数组
- `info`: 仅允许使用已明确声明的字段，目前 allowed_info_fields 为空，因此 info 不得用于任何奖励计算。
- `training_progress`: 仅当 prompt 明确允许使用训练进度时方可使用（当前未允许，建议忽略）。

禁止使用：
- `original_reward`：被屏蔽，严禁使用（会导致 reward hacking 或照抄官方奖励）。
- 任何未声明的 `info` 字段（包括但不限于 `reward_forward` 等官方奖励项、`x_position` 等全局位置）。
- 未在观测中明确给出的维度（如接触力，环境中不存在）。
- 任何显式或隐式的全局位置（如从 x_position 计算位移），因为已从 info 屏蔽且 obs 中不包含。

## 7. 可用于奖励函数的信号
- position: 
  - 身体高度 `body_z` (obs[0])
  - 关节角度 (obs[5:13])
- velocity:
  - 前进速度 `body_x_velocity` (obs[13])
  - 横向速度 `body_y_velocity` (obs[14])
  - 垂直速度 `body_z_velocity` (obs[15])
  - 关节速度 (obs[19:27])
- orientation:
  - 四元数 (obs[1:5])，可计算出直立程度 `body_up_z`
  - 角速度 roll/pitch/yaw (obs[16:18])，可额外作用于稳定性
- action/engine:
  - 动作向量 `action` (8,)，可用于惩罚大扭矩、动作变化等
- other:
  - 无。

## 8. 不确定或不可用的信号
- 全局位置 x, y：在 obs 和 info 均不可用，不能用于奖励。
- 接触力 / 足端触地信息：不可用。
- 成功 / 失败标志：信息中不存在，不可用。
- 摩擦 / 能耗 / 功率等指标：不可用。
- `original_reward`：被明确屏蔽，不可用。
- 时间步 / 剩余时间：info 中不提供，不建议依赖（除非接口明确允许使用 `training_progress`，但当前未授权）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped (4 legs, 8 joints)
  actuator_type: torque-controlled (effort directly commanded)
  contact_structure: four feet with ground contact, no explicit contact forces in obs
primary_objectives:
  - achieve and sustain forward velocity (body_x_velocity > 0)
  - maintain healthy body height in (0.2, 1.0) during locomotion
  - stay upright (body_up_z close to 1.0)
secondary_objectives:
  - minimize lateral velocity (body_y_velocity ≈ 0)
  - minimize vertical oscillation/velocity (body_z_velocity ≈ 0)
  - promote smooth, energy-efficient control (small joint torques and velocities)
main_failure_risks:
  - collapse (body_z <= 0.2) due to loss of balance or weak stride
  - excessive upward launch (body_z >= 1.0) due to overly energetic jumps
  - roll/pitch instability leading to inversion or large angular errors
  - numerical instability (any state → NaN/inf)
  - stationary or backward walking (body_x_velocity ≤ 0) – insufficient progress
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress  
  purpose: 激励机器人产生持续的正向前进速度。  
  why_required: 任务核心是向前移动，若无此职责，机器人可能学会原地站立而忽略运动目标。  
  usable_signals: `body_x_velocity` (obs[13])  
  risks: 过高的权重可能导致不惜代价加速而牺牲稳定性；需要与其他保持成分共同使用。

- role_id: body_height_penalty  
  purpose: 惩罚身体高度偏离健康范围（接近 0.2 或 1.0），防止跌倒或弹跳过高。  
  why_required: 高度超出范围直接导致 episode 终止；在终止前逐步惩罚可引导策略远离危险区域。  
  usable_signals: `body_z` (obs[0]), next_obs[0]  
  risks: 若只在终止时给予惩罚，稀疏性高；应设计为在整个有效范围内提供温和的边界惩罚。

- role_id: upright_stability  
  purpose: 鼓励身体保持直立，惩罚倾斜或翻转。  
  why_required: 严重倾斜可能导致摔倒或不可恢复的姿态；维持直立是稳定运动的前提。  
  usable_signals: 四元数可衍生 `body_up_z = 1 - 2*(quat_x² + quat_y²)`，仰/俯角速度也可辅助。  
  risks: 过于苛求完全直立可能压抑自然跑步的微小倾斜；可融入了 roll/pitch 容限。

### 10.2 条件职责 conditional_roles
- role_id
