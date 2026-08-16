# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器/着陆器轨迹优化任务。智能体（飞行器）起始于视口顶部中央附近，需要通过学习控制其主引擎与姿态引擎，**到达并稳定停靠在中央目标平台上**。
**主目标：尽可能快地到达目标平台并实现稳定、安全的接触（着陆）。**
**重要次目标（效率约束）：在完成主目标的过程中，尽量减少引擎推力使用（节约燃料）。**
**不该混淆的目标：** 此任务的核心是导航到达，而非单纯的存活平衡或持续前进。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确指向“到达并稳定在中央目标平台”这个核心目标，速度与能耗要求是完成主任务时的效率约束，而非独立冲突的多目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推理假设，基于连续物理量)
- obs[0]: x_position, 飞行器相对于目标平台的水平坐标，reward_usable: true
- obs[1]: y_position, 飞行器相对于目标平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity, 飞行器的水平线性速度，reward_usable: true
- obs[3]: y_velocity, 飞行器的垂直线性速度，reward_usable: true
- obs[4]: body_angle, 飞行器的姿态角度，reward_usable: true
- obs[5]: angular_velocity, 飞行器的角速度，reward_usable: true
- obs[6]: left_support_contact, 左侧支撑腿/结构接触标志 (0.0 或 1.0)，reward_usable: true
- obs[7]: right_support_contact, 右侧支撑腿/结构接触标志 (0.0 或 1.0)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0 (no_engine): 不点火，依靠惯性滑行
- action 1 (left_orientation_engine): 点燃一个侧向/方向引擎（原文为 orientation engine），产生逆时针/顺时针力矩，用于调整姿态
- action 2 (main_engine): 点燃主引擎，产生向上的推力，用于减速和悬停
- action 3 (right_orientation_engine): 点燃另一个侧向/方向引擎，与 action 1 力矩方向相反

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`。这可能在飞行器以低速平稳地着陆在目标平台后触发。这是最接近成功的终止条件，但不能排除因卡住或静止在错误位置而触发的情况。
- failure-like termination: `crash_or_body_contact` (任何非任务期望的身体接触/坠毁)、`horizontal_position_outside_viewport`（飞出水平边界）。
- ambiguous termination: 上述"body_not_awake_or_settled"本质上带有模糊性，可能因不稳定的卡死而非成功着陆触发。
- truncation: prompt未提及，假设为`False`。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (无明确info["success"])
- explicit_failure_flag_available: false (无明确info["failure"])
- allowed_info_fields: {} (step source返回空info)
- forbidden_or_uncertain_info_fields: 任何未在step source中声明的info字段，如“success”, “failure”, “termination_reason”, “landed” 等。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs: 形状为 (8,) 的上一状态向量
- action: 上一动作 (0~3)
- next_obs: 形状为 (8,) 的当前状态向量
- info: 仅能使用空字典 {}，无可用字段
- training_progress: 仅在 prompt 明确需要课程学习时使用

禁止使用：
- original_reward（已屏蔽）
- 官方或任何仿真器内置奖励
- 任何未在 step source 的 info 中声明的字段

## 7. 可用于奖励函数的信号
- position: `next_obs[0]` (x坐标), `next_obs[1]` (y坐标)
- velocity: `next_obs[2]` (x速度), `next_obs[3]` (y速度)
- orientation: `next_obs[4]` (角度), `next_obs[5]` (角速度)
- contact: `next_obs[6]` (左脚/支撑接触), `next_obs[7]` (右脚/支撑接触)
- action/engine: `action` (可设计引擎惩罚，但必须服务于主目标，即鼓励着陆而非停滞)

## 8. 不确定或不可用的信号
- 成功/失败标志：无。
- 目标平台具体坐标/范围：未知，只能从观察的零位中心推断 `x=0` 为目标水平中心，但垂直目标高度未知。`y_position` 为零可能代表接触平台高度。
- 燃料消耗量：无直接观测。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body
  actuator_type: idealized_thrusters
  contact_structure: two_landing_legs_or_support_points
primary_objectives:
  - reach_and_settle_on_target_platform (minimize final distance to target, zero velocity, stable angle, safe contact)
secondary_objectives:
  - fuel_efficiency (minimize total engine usage)
  - time_to_land (minimize episode length, implicitly driven by early termination)
main_failure_risks:
  - crashing into ground or other structures
  - flying out of viewport boundaries
  - getting stuck in an unstable state that triggers premature termination
  - using excessive main engine and oscillating without landing
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity
  purpose: 驱动飞行器靠近目标平台。
  why_required: 核心任务，没有此职责智能体将无明确目标。
  usable_signals: [next_obs[0], next_obs[1]]
  risks: 可能导致高速飞向目标并撞毁。

- role_id: velocity_management
  purpose: 在接近目标时（或全程）抑制速度，实现软着陆。
  why_required: `body_not_awake_or_settled` 终止条件暗示需要极低速度才能触发成功着陆。
  usable_signals: [next_obs[2], next_obs[3], next_obs[5]]
  risks: 过早抑制速度可能导致飞行器悬停、不下降或油耗极高。

- role_id: orientation_stabilization
  purpose: 保持飞行器姿态水平（零角度或小角度），并抑制角速度。
  why_required: 平稳着陆需要双腿接触 (`next_obs[6]`, `[7]`)，机身倾斜可能导致坠毁。
  usable_signals: [next_obs[4], next_obs[5]]
  risks: 若接触时对角度要求过于苛刻，可能阻碍着陆。

- role_id: safe_contact
  purpose: 鼓励以两侧支撑腿接触目标平台。
  why_required: 这是成功着陆的最终标志。
  usable_signals: [next_obs[6], next_obs[7]]
  risks: 可能诱导智能体在任意位置强行接触地面以获取奖励，因此必须与 `goal_proximity` 结合，仅在接近目标区域时给予奖励。

### 10.2 条件职责 conditional_roles
- role_id: sparse_success_reward
  condition_to_use: 当且仅当`terminated`为True，且我们能高度确信是成功着陆时（例如，通过组合检查：双脚接触、位置近零、速度近零、角度近零）。由于没有明确的成功标志，此职责的实现需要高度保守。
  usable_signals: [terminated (从 gym step返回), next_obs[6], next_obs[7], next_obs[0], next_obs[1], next_obs[3]]
  risks: 判断标准设置不当会导致非成功终止被错误奖励，导致奖励作弊。必须极其谨慎。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: engine_penalty
  reason: 禁止作为独立需求，只能作为其他职责的辅助项（例如在接近目标且速度已够低时，轻度惩罚非必要引擎使用）。过度惩罚引擎会使智能体倾向于不做动作，导致无法飞向目标或硬着陆。
  forbidden_or_missing_signals: [engine_fuel_consumption]

- role_id: time_penalty
  reason: 容易与存活压力混淆，可能导致智能体过早采取不安全的着陆行为（如一头扎向地面），冲击安全和稳定职责。仅在训练后期可作为极小因子引入。
  risks: 导致崩溃式着陆。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | next_obs[0], next_obs[1] | target_absolute_coordinates | distance_based_shaping, gaussian_potential, linear_quadratic_penalty | 结合速度进行 shaping |
| velocity_management | next_obs[2], next_obs[3], next_obs[5] | - | quadratic_penalty, bounded_signal | 可设计为目标位置相关的自适应速度上限 |
| orientation_stabilization | next_obs[4], next_obs[5] | - | quadratic_penalty, absolute_penalty | 对角速度和绝对角度进行惩罚 |
| safe_contact | next_obs[6], next_obs[7] | - | conditional_bonus, step_signal | 仅当位置和速度条件满足时给予密集或稀疏奖励 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 不飞向目标，无目的徘徊 | 各episode回报不增长，`x_position` 和 `y_position` 无明显向0收敛趋势。 | 增大 goal_proximity 奖励相对 engine_penalty 的权重。 |
| 高速撞击地面 (crash) | 大量 episode 以 crash 终止，`y_velocity` 在撞击前为高负值。 | 增强 velocity_management，特别是在 `y_position` 接近0时的速度惩罚。 |
| 水平飞出边界 | 大量 episode 因 `horizontal_position_outside_viewport` 终止。 | 增强对 `x_position` 偏差和 `x_velocity` 的控制，降低姿态引擎使用时的角速度副作用。 |
| 在目标上空振荡，无法着陆 | `y_position` 在略高于0的位置震荡，`y_velocity` 在零附近正负切换。 | 调整 velocity_management 的权重或形式，允许一定下沉速度；或在接近目标时降低悬停奖励密度，增加接触奖励引导。 |
| 着陆时姿态失控倾倒 | 双脚接触时 `body_angle` 或 `angular_velocity` 较大，导致后续 body contact。 | 增大 orientation_stabilization 权重，或将其与 safe_contact 奖励关联。 |
| 过早触发“settled”导致停在平台外 | 在远离目标的位置双脚着地并静止，导致“settled”终止，但任务未成功。 | 严格化 safe_contact 奖励的发放条件，必须结合对 `x_position` 和 `y_position` 的严格检查。 |