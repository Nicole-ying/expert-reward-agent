# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主任务：在二维平面中，控制一个带有两侧支撑腿的着陆器从视野上方中心附近受随机初速出发，**安全、稳定地降落到视野中央的目标平台上**，要求着陆器最终静止、姿态竖直且两侧支撑腿均与平台接触。  
次任务：在保证主目标的前提下，**尽可能快地完成着陆**，并且**尽量降低发动机推力消耗**（即少用燃料）。  
不应混淆的目标：燃料优化与快速着陆是辅助目标，它们不能凌驾于安全稳定着陆之上；本任务的最终评判是“是否成功停靠在平台上”这一离散事件，并非将能耗作为等价的主要目标。

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **confidence**: high
- **reason**: 任务明确要求到达并停止在指定目标平台（中心目标点），核心是到达特定位置并建立安全、稳定的接触（腿部着陆）。附属的能耗和速度要求是典型的辅助优化，不是冲突等价的多目标。动力学子类型选择 `goal_approach_and_soft_contact` 因为它要求着陆器先逼近目标、减速、然后依靠接触标志建立稳定着陆状态，完全符合“接近目标并低速、稳定接触/停靠”的模式。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: 根据上下文推断为 float32 (或混合了 bool 转为 float 的标志位，实际为连续量)
- **各维度含义**：
  - `obs[0]`: x_position (目标平台的相对水平坐标) —— reward_usable: true，用于控制水平对准。
  - `obs[1]`: y_position (相对于目标平台高度的垂直坐标) —— reward_usable: true，用于测量剩余高度/下降进度。
  - `obs[2]`: x_velocity (水平线速度) —— reward_usable: true，用于减速控制。
  - `obs[3]`: y_velocity (垂直线速度) —— reward_usable: true，用于控制坠落/着陆速度。
  - `obs[4]`: body_angle (机体偏转角) —— reward_usable: true，用于保持竖直姿态。
  - `obs[5]`: angular_velocity (角速度) —— reward_usable: true，姿态稳定惩罚。
  - `obs[6]`: left_support_contact (左支撑腿接触标志，1.0表示接触) —— reward_usable: true，接地判断。
  - `obs[7]`: right_support_contact (右支撑腿接触标志，1.0表示接触) —— reward_usable: true，接地判断。

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `action 0` (no_engine): 不启动任何引擎（零推力）。
  - `action 1` (left_orientation_engine): 启动左侧姿态引擎，产生旋转力矩。
  - `action 2` (main_engine): 启动主引擎，产生向上的升力。
  - `action 3` (right_orientation_engine): 启动右侧姿态引擎，产生相反的旋转力矩。
- **值域**：`[0, 1, 2, 3]`，每个动作只消耗一个离散选择的“燃料单位”。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` —— 机体静止且（可能）已与平台产生接触。根据任务描述“settle at a central target pad... make safe contact”，此条件很可能标志着成功着陆。但是否伴随 `left_support_contact` 和 `right_support_contact` 同时为真，需要观察，但终止代码中只用此单一条件判定终止，可能表示“已稳定着陆”。
- **failure-like termination**:  
  `crash_or_body_contact` —— 机体与外环境发生碰撞（非支撑腿安全着陆），通常是坠毁。  
  `horizontal_position_outside_viewport` —— 水平飘出视野边界，失败。
- **ambiguous termination**:  
  在没有显式成功标志的情况下，终止由这三种条件之一触发，其中 `body_not_awake_or_settled` 语义为“非清醒或已稳定”，我们暂时将其视为**成功候选**，但无法 100% 确认。需要后续根据经验收集证据。
- **truncation**:  
  源代码未展示任何步数限制或超时截断（`return ..., False, {}` 中没有 truncated）。如果存在，可能在上层包装器中，但当前分析不可见。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false （info 字典为空）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无（info={} 且没有声明任何可用字段）
- **forbidden_or_uncertain_info_fields**: 所有 info 字段都不允许使用，因为没有可靠信息源。`original_reward` 被屏蔽，禁止访问。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs`（当前或上一帧观察，根据实际调用约定，一般为当前步之前的 obs，也可能是上一步的 obs，但按典型用法 `compute_reward(obs, action, next_obs)`，此处 obs 为动作执行前的观察，next_obs 为执行后的观察。）
- `action`（刚刚执行的离散动作）
- `next_obs`（执行后观察）
- `info` 中明确允许的字段：**无**
- `training_progress`：仅当 prompt 明确允许使用时才可用。本任务未提及，因此**禁止使用**。

禁止使用：
- `original_reward`
- 任何未在 observation 字段中声明的、推导出的或猜测的信息
- 未在任务描述中明确可用的 info 键（如 “success”、“failure”等）

## 7. 可用于奖励函数的信号
- **position**:
  - `next_obs[0]` (x_position)：到目标平台的水平距离。
  - `next_obs[1]` (y_position)：到目标平台高度的垂直距离。
- **velocity**:
  - `next_obs[2]` (x_velocity)
  - `next_obs[3]` (y_velocity)
- **orientation**:
  - `next_obs[4]` (body_angle)
  - `next_obs[5]` (angular_velocity)
- **contact**:
  - `next_obs[6]` (left_support_contact)
  - `next_obs[7]` (right_support_contact)
- **action/engine**:
  - `action`：离散动作编号，可用于惩罚引擎使用（燃料消耗），但不能直接读出推力大小。
- **other**:  
  无。

## 8. 不确定或不可用的信号
- `info` 字典为空，不可提供任何辅助诊断信号。
- 没有直接的“着陆成功”标志位，需要从接触+静止+位置/角度条件进行启发式推断。
- 没有“剩余燃料”或“推力大小”的连续量；只能从行为离散动作推断引擎是否开启。
- 不能获得外部风速等扰动信息。
- 不能访问 `original_reward`。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_two_legs
  actuator_type: discrete_engine_select (main up, left orientation, right orientation, idle)
  contact_structure: two_legs_with_boolean_contact_signals_on_each_leg
primary_objectives:
  - land safely on the target pad (both legs contacting, body stable, within horizontal tolerance)
secondary_objectives:
  - minimize fuel consumption (discrete engine activations)
  - minimize time-to-land (implicit from trajectory, no explicit time counter)
main_failure_risks:
  - body collision during landing (crash_or_body_contact)
  - drifting horizontally out of bounds
  - hovering or oscillating without settling
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity`
  - **purpose**: 驱动着陆器向目标平台移动，减小水平距离和垂直距离。
  - **why_required**: 这是到达目标的核心驱动力；没有这个奖励，智能体缺乏方向。
  - **usable_signals**: `next_obs[0]`, `next_obs[1]`
  - **risks**: 如果只奖励接近，可能鼓励快速坠落导致硬着陆。

- **role_id**: `soft_landing_stabilization`
  - **purpose**: 在接近目标时强烈惩罚高速度，鼓励低速接触。
  - **why_required**: 任务要求“settle”和“safe contact”，速度过高会导致 crash。
  - **usable_signals**: `next_obs[2]`, `next_obs[3]`
  - **risks**: 过度惩罚速度可能导致永远不降落。

- **role_id**: `upright_attitude`
  - **purpose**: 惩罚机体偏离竖直方向以及角速度。
  - **why_required**: 安全着陆通常要求竖直姿态，且任务描述提示“keep a stable orientation”。
  - **usable_signals**: `next_obs[4]`, `next_obs[5]`
  - **risks**: 如果过强，可能限制机动性，但一般可接受。

- **role_id**: `contact_establishment`
  - **purpose**: 奖励两条腿同时与平台接触。
  - **why_required**: 只有接触才认为是“landed”，否则即使静止也可能浮空。
  - **usable_signals**: `next_obs[6]`, `next_obs[7]`
  - **risks**: 如果只奖励接触而不顾速度，可能鼓励高速冲撞获得接触后 crash。因此需要与速度惩罚联合。

### 10.2 条件职责 conditional_roles
- **role_id**: `fuel_efficiency`
  - **condition_to_use**: 当归一化进度（如接近目标）达到一定阈值，或始终使用但权重随训练阶段调整，以避免早期过度抑制探索。
  - **usable_signals**: `action`（离散动作中非 0 的动作意味着引擎开启）
  - **risks**: 如果从一开始就强惩罚引擎使用，智能体可能不敢驱动机体，导致停滞。建议在后期或接近目标时启用。

- **role_id**: `time_pressure_implicit`
  - **condition_to_use**: 一般不可直接实现，因为没有时钟信号。只能通过奖励形状鼓励快速完成（例如二次惩罚位置和速度，比线性奖励更鼓励快速结束）。不是一个独立的显式角色，而是形状设计问题。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `success_flag_bonus`
  - **reason**: 没有显式的成功标志，且不能访问 info 或原始奖励，无法实现大额稀疏成功奖励。如果基于启发式判断成功（如接触+静止）有误判风险，尤其在学习初期可能错误引导。禁用。
  - **forbidden_or_missing_signals**: info["success"] 缺失。

- **role_id**: `shaping_from_original_reward`
  - **reason**: 原始奖励被屏蔽，禁止复制或重建。不可用。
  - **forbidden_or_missing_signals**: original_reward 被禁用。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | `next_obs[0]`, `next_obs[1]` | 无 | `dense_state_signal` (e.g., -distance metric) | 可用欧氏距离或各自的分量惩罚，注意避免水平偏离和垂直高度相同尺度导致不公平。 |
| soft_landing_stabilization | `next_obs[2]`, `next_obs[3]` | 接触速度阈值 | `quadratic_penalty` 或 `negative_abs` | 速度惩罚在接近地面时权重可增大，但需从 obs 间接推断接近程度。 |
| upright_attitude | `next_obs[4]`, `next_obs[5]` | 目标角度（应为0） | `quadratic_penalty` (angle) + `quadratic_penalty` (angular vel) | 角度绝对值或平方。 |
| contact_establishment | `next_obs[6]`, `next_obs[7]` | 无 | `bounded_signal` (0..2) 或 `binary_sum` | 可直接使用 `left_contact + right_contact` 作为连续奖励。 |
| fuel_efficiency (conditional) | `action` (0: idle, others: engine) | 燃料量 | `negative_indicator` (action != 0) | 仅当训练后期或接近目标时加入惩罚。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停不下降 | `next_obs[1]` 长期为正，无减小趋势；`action` 可能总是 idle 或只点按。 | 加大高度惩罚系数；或在接近目标时不惩罚下降速度。 |
| 高速坠毁 | 接触时速度很大 (`next_obs[3]` 为较大负值) 且 episode 通过 crash 终止。 | 增强速度惩罚，尤其是垂直速度；在接近表面时增大速度惩罚权重（根据 `y_position` 缩放）。 |
| 水平漂移出界 | `next_obs[0]` 绝对值持续增大，超出边界终止。 | 增大水平位置惩罚；还可加入对水平速度的空间感知（越接近边界越惩罚）。 |
| 单腿着陆/侧翻 | 终止时只有一侧支撑腿接触，另一侧未接触。身体角度大。 | 提高 `contact_establishment` 奖励并可能结合姿态惩罚；让双腿接触成为必要高额奖励。 |
| 过度使用燃料但无法稳定 | 动作频繁切换非零，速度仍在振荡。 | 适时引入 fuel_efficiency 惩罚，并减慢引擎响应（但无法改动力学），只能通过奖励抑制高频动作。 |
