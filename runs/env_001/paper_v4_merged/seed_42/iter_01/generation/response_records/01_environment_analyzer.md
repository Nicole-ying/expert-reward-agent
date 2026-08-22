# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从视口顶部出发，以最短时间和最少推力消耗到达并稳定停靠在画面中心的目标平台上。  
要求同时满足：水平与垂直位置均收敛至平台原点、速度趋近于零、身体姿态保持稳定、左右支撑腿同时与平台接触，且过程中避免坠毁、翻倾或飞出边界。  
任务核心是精准导航‑停靠；附属优化是燃料经济与快速性，两者不应混淆为主要目标。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是驱动飞行器到达空间中的一个固定目标点，并在该点完成稳定停靠。  
虽然任务描述也要求“尽可能快”“尽可能省燃料”，但这两者是典型的附属优化，主次关系明确，不属于多目标权重相当且冲突的场景。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (隐含，所有分量均为连续值或 0/1 浮点数)

各维度含义：

- obs[0]: x_position — 飞行器相对于目标平台中心的水平坐标，reward_usable: true  
- obs[1]: y_position — 飞行器相对于平台高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity — 水平线速度，reward_usable: true  
- obs[3]: y_velocity — 垂直线速度，reward_usable: true  
- obs[4]: body_angle — 身体朝向角，reward_usable: true  
- obs[5]: angular_velocity — 角速度，reward_usable: true  
- obs[6]: left_support_contact — 左支撑腿是否与表面接触 (1.0 接触，0.0 未接触)，reward_usable: true  
- obs[7]: right_support_contact — 右支撑腿是否与表面接触 (1.0 接触，0.0 未接触)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4

各动作含义：

- action 0: no_engine — 不启动任何引擎，只靠惯性运动  
- action 1: left_orientation_engine — 点燃左侧姿态引擎，用于调整角度/旋转  
- action 2: main_engine — 点燃主引擎，通常产生沿身体某方向的推力（可能包含垂直方向的一次性推力）  
- action 3: right_orientation_engine — 点燃右侧姿态引擎，与左引擎反向旋转

动作选择直接影响燃料消耗和姿态变化，奖励设计中需要跟踪动作计数来估计燃料/推力使用。

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact — 飞行器主体发生不应有的碰撞或坠毁，通常视为失败  
- horizontal_position_outside_viewport — 水平位置超出画面边界，视为失败  
- body_not_awake_or_settled — 身体进入沉睡状态或判定为已稳定停靠，可能是成功，但源码中未区分是否为正常着陆成功

无任何显式成功/失败标志传入 info 字典，因此需要**通过观测信号间接推断**终止原因。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: (空字典，没有任何字段)  
- forbidden_or_uncertain_info_fields: 禁止使用 info 读取任何字段，因为环境不提供额外信息

终止判断：
- 若 episode 终止 (not truncated) 且最终观测满足 `距离目标近、双腿接触、速度低、姿态角小`，则推断为**成功停靠** (derived_possible)  
- 若最终状态中出现任意一条不满足（如位置严重偏离、未接触、速度极大、角度过大），则推断为**失败**（坠毁、出界等）  
- 由于 termination 函数已混合了成功与失败条件，无法直接从环境获取标签，所以成功奖励必须通过 derived 推断给出

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用的输入：
- obs: 当前步观测向量 (8 维)  
- action: 当前步执行的动作 ID  
- next_obs: 下一步观测向量 (8 维)  
- info: 空字典，不可从中读取任何内容  
- training_progress: 仅当 prompt 明确要求且训练阶段相关时才使用（本环境不要求，可忽略）

严禁使用：
- original_reward 或任何形式的官方奖励  
- 任何未声明的 info 字段  
- 任何未说明的 obs 切片含义

## 7. 可用于奖励函数的信号
可直接使用的观测信号：
- position (相对于目标): `obs[0]` (x), `obs[1]` (y)  
- velocity: `obs[2]` (vx), `obs[3]` (vy)  
- orientation: `obs[4]` (angle), `obs[5]` (angular_vel)  
- contact: `obs[6]` (left contact), `obs[7]` (right contact)  
- action/engine: 当前动作 `action`，可用于检测引擎使用

间接可用信号（从观测推导）：
- distance_to_target: ‖(obs[0], obs[1])‖  
- is_crashed_or_oob: 由最终状态的位置、速度突变、接触缺失推断 (derived_possible)  
- is_successful_landing: 距离近、双腿均接触、速度低、角速度低 (derived_possible)  
- fuel_usage: 动作 1、2、3 视为消耗燃料（可加权计数）

## 8. 不确定或不可用的信号
- 任何 info 字段（环境明确返回空字典）  
- 显式成功标志、失败原因  
- 接触法向量、地面硬度等物理属性  
- 世界坐标系的绝对边界（水平超出视野的阈值未给出，但可以通过 obs 突变推断）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: planar_lander_like
  actuator_type: main_engine_with_lateral_orientation_thrusters
  contact_structure: left_and_right_landing_legs_contact_flags
primary_objectives:
  - reach_target_pad_and_settle (position near origin, low velocity, both legs in contact)
secondary_objectives:
  - minimize_fuel_consumption (reduce use of all engines)
  - minimize_time_to_settle (implicitly encouraged by sparse success bonus or small time penalty)
main_failure_risks:
  - crash_or_hard_body_contact (unsafe landing, flipping)
  - horizontal_drift_out_of_viewport
  - overshoot_target_and_oscillate
  - fail_to_kill_velocity_and_bounce_off_pad
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_progress_delta_distance**
  purpose: 保证飞行器持续朝目标平台移动。  
  why_required: 导航‑到达任务的核心是“离目标更近了吗”，避免悬停陷阱。  
  usable_signals: obs[0], obs[1], next_obs[0], next_obs[1]  
  计算方式：`distance(current) - distance(next)`，正奖励表示接近，负奖励表示远离。  
  risks: 如果 agent 长时间在目标附近悬停而不落稳，delta 会趋近于零，需要稳定停靠激励来驱动最终接触与降速。

- **role_id: stable_contact_and_low_velocity_bonus**
  purpose: 在飞行器接近目标后，鼓励双腿接触且速度量级低的“安全着陆状态”。  
  why_required: 导航到达若不要求稳定停靠，智能体会反复经过目标但不停留，无法完成最终“settle”。  
  usable_signals: obs[0], obs[1], obs[2], obs[3], obs[6], obs[7], next_obs 对应分量  
  基本形态：当 distance 小、双腿接触、速度低时给予连续的增稳奖励，促使 agent 维持稳定直到 episode 自然终止。  
  risks: 需通过距离 gate 限制生效范围，避免在远离目标时给予无意义奖励；速度阈值设置不当可能导致持续微调碰撞。

### 10.2 条件职责 conditional_roles
- **role_id: fuel_efficiency_penalty**
  condition_to_use: 任务明确要求“尽可能少用引擎推力”。应在整个 episode 中启用，但权重应低于导航与稳定奖励。  
  usable_signals: action (0,1,2,3)  
  基本形态：对动作 1,2,3 施加小的负奖励，乘以对应消耗系数（主引擎可能消耗更大）。  
  risks: 权重过大会抑制必要的姿态调整，导致任务失败。

- **role_id: inferred_terminal_success_bonus**
  condition_to_use: 当 episode 终止 (terminated=True, truncated=False) 时，基于最终 next_obs 推断是否为成功着陆。  
  usable_signals: next_obs 的坐标、接触、速度、姿态角  
  基本形态：若满足 `distance < epsilon`, `contact left & right`, `|v| < v_thresh`, `|angle| < theta_thresh`，给予大的正奖励；否则不
