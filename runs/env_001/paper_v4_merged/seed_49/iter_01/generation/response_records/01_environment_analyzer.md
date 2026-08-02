# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化任务。初始时，飞行器从视口顶部中心附近以随机初速度开始运动。核心目标是**尽快且平稳地降落在中央目标垫上**，同时**尽量减少发动机使用**。学习者必须学会接近目标、主动减速、保持机身姿态稳定，并在所有支撑腿（或接触点）安全触垫后停止动作。失败的着陆（碰撞、飞出水平边界）应彻底避免。注意：节能是次目标，不应凌驾于成功着陆之上。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**
confidence: high
reason: 核心是到达并停靠在指定目标位置（目标垫），属于典型的导航-目标到达任务。节能是附属优化，不是等价多目标。

dynamics_subtype: **goal_approach_and_soft_contact**  
说明：飞行器在重力（隐含）与推进力作用下运动，需主动控制姿态与主推力，实现**接近 → 减速 → 稳定姿态 → 软接触**的连串过程，符合接近目标并低冲击接触的动力学特征。

## 3. 观察空间 observation_space
- type: Box（连续向量）
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x_position —— 飞行器中心相对目标垫中心的水平坐标。reward_usable: true
- obs[1]: y_position —— 飞行器底部（或重心）相对垫面的垂直高度。reward_usable: true
- obs[2]: x_velocity —— 水平线速度。reward_usable: true
- obs[3]: y_velocity —— 垂直线速度。reward_usable: true
- obs[4]: body_angle —— 机身倾斜角。reward_usable: true
- obs[5]: angular_velocity —— 角速度。reward_usable: true
- obs[6]: left_support_contact —— 左支撑/腿触地标志（0/1）。reward_usable: true
- obs[7]: right_support_contact —— 右支撑/腿触地标志（0/1）。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（无任何推力，仅靠惯性/重力）  
- action 1: left_orientation_engine（启动左侧姿态发动机，产生逆时针/向右的力矩）  
- action 2: main_engine（启动主发动机，向上产生推力）  
- action 3: right_orientation_engine（启动右侧姿态发动机，产生顺时针/向左的力矩）  

主发动机提供垂直方向推力，姿态发动机用于调整倾斜角度从而改变水平推力方向，典型的**主推力+姿态控制**模式。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  终止条件中的 `body_not_awake_or_settled` 极可能对应成功着陆后的稳定状态：飞行器停在目标垫上，速度降至零，物理引擎判定其静止休眠。
- failure-like termination:  
  - `crash_or_body_contact`：飞行器主体（非支撑腿）触地或与障碍物碰撞。  
  - `horizontal_position_outside_viewport`：水平坐标超出视口范围。
- ambiguous termination:  
  理论上 `body_not_awake_or_settled` 也可能因其它原因（如卡在边界外）触发，但实际环境中通常与成功着陆强关联。
- truncation: 本环境无时间截断（step 返回 truncated 固定为 False）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空字典）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 无任何 info 字段
- 成功/失败只能**间接推断**：  
  当 episode 终止时，通过最终观测（位置、速度、角度、接触标志）综合判断是否成功着陆：
  - 推断成功条件：`|x_position| < ε_x`，`|y_position| < ε_y`，`√(vx²+vy²) < ε_vel`，`|body_angle| < ε_ang`，且至少一个接触标志为 1。
  - 否则为失败。  
  该推断路径标记为 **derived_possible**。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（动作执行前的状态）
- `action`（选择的动作索引）
- `next_obs`（动作执行后的状态）
- `info` 中明确允许的字段（当前为无，因此不可用 info）
- `training_progress` 仅有明确指示时才使用（本例无需课程式奖励，不使用）

禁止使用：
- `original_reward` / `official_reward`（已屏蔽）
- 任何未在 obs 说明中声明的 obs 切片
- 任何未在允许列表中的 info 字段

## 7. 可用于奖励函数的信号
**位置与接近度**：
- `next_obs[0]`（x 偏移）：越靠近 0 越好。
- `next_obs[1]`（y 偏移）：越靠近 0 越好，且应在安全范围内。
- 可直接计算距离 `dist = sqrt(x² + y²)` 或步间距离减少量 `delta_dist`。

**速度**：
- `next_obs[2]`（vx）、`next_obs[3]`（vy）：着陆时需极低速度，飞行中可适度引导向下减速。

**姿态**：
- `next_obs[4]`（body_angle）：应保持在安全区间内（例如 ±0.3 rad），防止侧翻。
- `next_obs[5]`（angular_velocity）：应趋近 0。

**接触**：
- `next_obs[6]`（左触地）、`next_obs[7]`（右触地）：成功着陆通常需要双接触（或至少一腿触地且速度达标）。

**动作**：
- `action` 可用来惩罚发动机使用（尤其 punishment for engine actions），但禁止奖励“无动作”。

**终端事件（derived_possible）**：
- 由最终 `next_obs` 推断的成功奖励（大正分）或失败惩罚（负分）。

## 8. 不确定或不可用的信号
- 燃料/能量消耗量：未提供。
- 目标垫的绝对坐标：未提供，只有相对位置。
- 地面高度/垫高度：未直接给出，但 y_position 已是相对垫面高度，可认为垫面在 y=0 处。
- 环境时间步长：未知，但可假设固定，无需使用。
- 任何 info 字段：不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_two_contacts (left/right support)
  actuator_type: main_engine_and_orientation_thrusters
  contact_structure: two_legs_or_pads_contact_with_ground
primary_objectives:
  - land_on_center_pad: 飞行器最终静止在目标垫上方，双接触点触垫，姿态近乎水平。
secondary_objectives:
  - minimize_engine_usage: 减少总动作次数或推力积分。
  - smooth_and_stable_approach: 避免剧烈摆动、高速冲击。
main_failure_risks:
  - crashing_body: 主体接触地面或障碍物导致坠毁。
  - drifting_out_of_viewport: 水平移动超出可观测区域。
  - hard_landing: 速度过大或角度倾斜导致侧翻，即使接触垫也视为失败（可通过身体接触垫判定失败？注意 crash_or_body_contact 可能包含了除腿以外的身体触地，所以触垫后若腿部接触而身体未触则为安全，身体触垫触发 crash）。
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: approach_progress  
  purpose: 引导飞行器向目标垫移动，每步因为距离减小而获得正反馈。  
  why_required: 要到达目标，必须持续缩短与垫的相对距离。若只有稀疏终端奖励，探索困难。  
  usable_signals: [x_position, y_position] (计算 distance)  
  risks: 若使用状态值（proximity）可能造成悬停收割（停在远处但距离固定，仍能得分）。因此应采用**步间距离减少量（delta distance）作为主信号**，鼓励每一步更近。当距离很小时可切换为接近 0 的距离接近奖励以保证接触。  
  候选算子：`delta(distance)`, `improvement`。

- role_id: soft_landing_condition  
  purpose: 确保最终接触时满足低速度、小倾斜角、双接触（或至少一腿触地）。  
  why_required: 任务要求 “make safe contact”，单纯到达位置不足以成功。  
  usable_signals: [x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]  
  risks: 在飞行过程中过早接触应该给予惩罚（如飞行中无意触地）。因此需将该职责与“非终止阶段”区分，仅当检测到接触且速度/角度符合安全着陆条件时给予正奖励，否则如果接触状态出现但未满足条件则视作 crash 惩罚。  
  候选算子：`hinge_penalty(velocity)`, `bounded_signal(body_angle)`, `sparse_terminal_bonus`。

- role_id: orientation_stability  
  purpose: 维持机身水平，防止旋转失控。  
  why_required: 大角度可能导致侧翻或触地失败，且影响主发动机推力方向。  
  usable_signals: [body_angle, angular_velocity]  
  risks: 过强限制可能限制必要的姿态调整。可采用 **hinge penalty**：只在 angle 超过安全范围（如 ±0.3 rad）时施加惩罚，或在角速度过高时惩罚。  
  候选算子：`hinge_penalty`, `angular_velocity_penalty`。

### 10.2 条件职责 conditional_roles
- role_id: engine_efficiency  
  condition_to_use: 当任务明确要求“省燃料”或 agent 开始出现频繁无效点火时启用。  
  usable_signals: [action]  
  risks: 若过早加入会抑制探索，导致 agent 不敢使用主发动机。建议在训练初期权重很低或延迟引入（curriculum）。  
  候选算子：`per_step_action_cost`（对非零动作施加小惩罚）。

- role_id: terminal_success_bonus (derived_possible)  
  condition_to_use: 仅在 episode 终止且能可靠推断成功时使用。  
  usable_signals: [final x_position, y_position, velocity, angle, contacts]  
  risks: 推断错误会奖励失败行为。必须设置严格的阈值且配合调试输出验证准确率。  
  候选算子：`sparse_terminal_bonus`, `conditional_bonus`.

- role_id: crash_penalty (derived_possible)  
  condition_to_use: 仅在 episode 终止且推断为失败（crash 或 out of bounds）时施加。  
  usable_signals: [final x_position, y_position, contacts, and any rule for crash]  
  risks: 同上，需谨慎推断。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: survival_time_reward  
  reason: 任务不是存活类型，停滞不动不能得分，时间奖励会鼓励不降落、反复悬停消耗步数。禁用。  
- role_id: constant_forward_velocity  
  reason: 没有固定前向轴，下降速度过快反而是危险。本环境应减速，而非维持高速前进。禁用。  
- role_id: path_smoothness_penalty (action rate)  
  reason: 发动机切换是离散动作，但可作为 optional 燃料效率的一部分，但非必需，且可能阻碍探索。暂不入围。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_progress | x_position, y_position (derived distance) | – | delta
