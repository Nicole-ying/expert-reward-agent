# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主目标：从画面顶部中央附近的随机初态出发，控制二维飞行器尽快到达中心目标垫并稳定着陆（速度接近零、姿态接近水平、支撑腿接触）。  
次目标：在完成主目标的前提下，尽可能少用引擎推力（节能）。  
混淆澄清：该任务不是纯粹的速度竞赛，也不是单纯的节能悬停，核心是实现“成功、快速、节能”的精确着陆。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: high  
reason: 核心目标是到达并稳定在指定目标垫，导航到达是根本特征；节能、快速是附属优化，主次分明，不存在多目标权重相当和强烈冲突。

动力学子类型：`goal_approach_and_soft_contact`

## 3. 观察空间 observation_space
- type: `Box`
- shape: `[8]`
- dtype: `float32`（推断）
- obs[0]: `x_position`，水平坐标（相对目标垫），可用于奖励接近目标，reward_usable: true
- obs[1]: `y_position`，垂直坐标（相对垫高），可用于奖励接近目标和防越界，reward_usable: true
- obs[2]: `x_velocity`，水平线速度，可用于奖励稳定着陆和禁止硬着陆，reward_usable: true
- obs[3]: `y_velocity`，垂直线速度，同上，reward_usable: true
- obs[4]: `body_angle`，机体俯仰角，可用于奖励姿态稳定，reward_usable: true
- obs[5]: `angular_velocity`，角速度，可用于奖励姿态平滑，reward_usable: true
- obs[6]: `left_support_contact`，左支撑腿接触标志（0/1），用于判断着陆成功，reward_usable: true
- obs[7]: `right_support_contact`，右支撑腿接触标志，同上，reward_usable: true

## 4. 动作空间 action_space
- type: `Discrete`
- n: 4
- action 0: `no_engine`，不做任何推力/旋转，完全靠惯性
- action 1: `left_orientation_engine`，激发侧向方向引擎（改变机体转角，可能逆时针）
- action 2: `main_engine`，激发主引擎（沿机体朝向产生推力）
- action 3: `right_orientation_engine`，激发另一侧方向引擎（改变机体转角，可能顺时针）

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（当飞行器停稳、身体不再移动时触发，可能在正确着陆后发生）
- failure-like termination:  
  - `crash_or_body_contact`（主体非支撑部分碰地或坠毁）  
  - `horizontal_position_outside_viewport`（水平方向飞出视野）
- ambiguous termination: `body_not_awake_or_settled`在未到达目标垫时也可能触发（如坠落后停在边界外），但这时的终止不应视为成功；需要通过观察是否在目标垫上和接触确认。
- truncation: 代码中 `truncated` 恒为 `False`，无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: `{}`（空字典，无任何 info 字段）
- forbidden_or_uncertain_info_fields: 不存在 `info["success"]`、`info["failure"]`、`info["termination_reason"]`，任何推断成功/失败的逻辑必须从 `obs` / `next_obs` 的状态条件进行判定。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（长度为8的观察向量）
- `action`（离散动作索引）
- `next_obs`（同结构观察向量）
- `info` 中明确允许的字段：当前为 `{}`，无可用字段
- `training_progress` 仅在任务描述明确允许时可用，当前未声明，**不应使用**

禁止使用：
- `original_reward`
- 任何官方、掩码奖励
- 未声明的 `info` 字段
- 未声明的 `obs` 切片（不要从 obs 中额外提取未在观察空间中列出的信息）

## 7. 可用于奖励函数的信号
- **位置**：`x_position`(obs[0]), `y_position`(obs[1]) — 水平与垂直相对目标垫，可直接用于奖励接近和判断边界。
- **速度**：`x_velocity`(obs[2]), `y_velocity`(obs[3]) — 线速度，用于奖励减速、软着陆。
- **姿态**：`body_angle`(obs[4]) — 角度（可能是弧度或归一化角），用于奖励水平姿态；`angular_velocity`(obs[5]) — 用于奖励姿态平稳。
- **接触**：`left_support_contact`(obs[6]), `right_support_contact`(obs[7]) — 布尔值，标志支撑腿是否接触地面，可奖励着陆成功。
- **动作/引擎**：`action` — 离散索引，可对主引擎和方向引擎使用进行惩罚，实现节能。

## 8. 不确定或不可用的信号
- **绝对成功/失败标志**：无法从任何直接字段获得，必须通过 `next_obs` 和终止时状态的条件逻辑推断。
- **剩余燃料/油门量**：未提供。
- **目标垫精确坐标**：由 `x_position`、`y_position` 的相对坐标隐含，但无绝对位置信息。
- **外部风力/扰动**：源文件中被省略，无法使用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 单刚体飞行器（2D）
  actuator_type: 主推进引擎 + 两个侧向定向引擎
  contact_structure: 两个支撑腿，各自有独立接触传感器
primary_objectives:
  - 到达目标垫上方并稳定着陆（水平位置接近0，垂直位置接近0，速度近零，姿态水平，至少一条支撑腿接触）
secondary_objectives:
  - 尽可能少用引擎推力（节能）
  - 尽可能快地着陆（鼓励短时间的飞行）
main_failure_risks:
  - 硬着陆（速度过大导致坠毁）
  - 水平飞离视野
  - 长时间悬停浪费燃料
  - 姿态剧烈振荡导致无法稳定
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id: goal_proximity**  
  purpose: 激励飞行器尽快接近目标垫（水平、垂直方向）。  
  why_required: 到达目标是核心任务，若无此项，agent 无法学会靠近垫子。  
  usable_signals: [`x_position`, `y_position`]  
  risks: 如果只奖励接近而不限制速度，可能导致高速撞击；需配合软着陆职责。

- **role_id: soft_landing**  
  purpose: 在接近目标垫时鼓励低速、水平姿态和支撑腿接触。  
  why_required: 仅到达目标位置可能因硬着陆而失败，必须奖励稳定、安全的着陆条件。  
  usable_signals: [`x_velocity`, `y_velocity`, `body_angle`, `left_support_contact`, `right_support_contact`, `x_position`, `y_position`]（在垫子附近进行激活）  
  risks: 必须在接近目标时才着重奖励，否则早期可能误导探索。

- **role_id: fuel_efficiency**  
  purpose: 惩罚不必要的引擎使用，鼓励用最少的推力完成任务。  
  why_required: 任务明确要求使用尽量少的推力，是明确的优化目标之一。  
  usable_signals: [`action`]（判别 `main_engine` 和左右方向引擎的激发）  
  risks: 惩罚过强可能使 agent 不敢动作，导致自由落体坠毁；需与 goal_proximity 和 soft_landing 协同。

- **role_id: crash_prevention**  
  purpose: 对可能导致坠毁或飞出边界的危险状态给予负激励。  
  why_required: 防止 agent 学习到高速撞击或飞出画面的危险轨迹，提高样本效率。  
  usable_signals: [`x_position`, `y_position`, `x_velocity`, `y_velocity`]（用于判断是否即将越界或高速撞地）  
  risks: 如果危险信号过于保守可能限制探索空间；需仅对明显不安全状态施加。

- **role_id: living_penalty (时间惩罚)**  
  purpose: 每个时间步给予很小的负奖励，促使 agent 尽快完成任务。  
  why_required: 任务要求“尽可能快”，时间步惩罚是推动快速着陆的标准手段。  
  usable_signals: 常数（不需要状态信号）  
  risks: 过大的惩罚可能导致 agent 选择高速撞击以求快速终结，须配合 crash_prevention。

### 10.2 条件职责 conditional_roles
- **role_id: orientation_stabilization**  
  condition_to_use: 在 agent 已经接近目标垫且开始减速时激活，鼓励保持水平姿态。否则可能在飞行前期限制必要机动。  
  usable_signals: [`body_angle`, `angular_velocity`]  
  risks: 过早强调姿态会阻止 agent 学习必要的转向机动。

- **role_id: landing_success_bonus**  
  condition_to_use: 仅在检测到可能触发了 success-like termination 且满足所有着陆条件时一次性给予较大奖励；可在 `next_obs` 表明稳定、接触、低速度且位置靠近目标时给出。  
  usable_signals: [`next_obs` 的全部相关字段，`x_position`, `y_position`, `x_velocity`, `y_velocity`, `body_angle`, `left_support_contact`, `right_support_contact`]  
  risks: 可能被 agent 利用，在错误位置故意“停稳”骗取奖励；需严格检查位置和接触条件。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: sparse_success_only**  
  reason: 环境没有提供明确的成功标志，完全稀疏奖励（仅在最终成功时给正奖励）会导致极低的样本效率和严重的探索困难。  
  forbidden_or_missing_signals: [`explicit_success_flag` 缺失]

- **role_id: wind_perturbation_cost**  
  reason: 源文件中风的影响被省略，无法可靠估计，不应在奖励中加入抗风干扰项。  
  forbidden_or_missing_signals: 无可用风相关观测。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | `x_position`, `y_position` | none | `dense_state_signal`, `quadratic_penalty`, `bounded_signal` | 使用距离的负值或高斯型鼓励靠近；注意越远离惩罚越大 |
| soft_landing | `x_velocity`, `y_velocity`, `body_angle`, `left_support_contact`, `right_support_contact`, `x_position`, `y_position` | none | `smooth_threshold`, `step_bonus`, `product` | 可设计为速度越小、姿态越平、接触越多且位置越近时奖励越高 |
| fuel_efficiency | `action` | none | `discrete_action_cost`, `count_cost` | 对 action 2（主引擎）给予较高惩罚，对方向引擎可较小，对无动作无罚 |
| crash_prevention | `x_position`, `y_position`, `x_velocity`, `y_velocity` | none | `boundary_penalty`, `speed_limit_penalty` | 当 x_position 接近视野边缘或 y_position 负且下落速度快时给罚 |
| living_penalty | (constant) | none | `constant_penalty` | 固定小负值，如 -0.01 |
| orientation_stabilization | `body_angle`, `angular_velocity` | none | `smooth_threshold`, `abs_error` | 在接近目标并减速时激活 |
| landing_success_bonus | `next_obs` 的位置/速度/姿态/接触 | none | `logical_and`, `step_bonus` | 一次性大额奖励，条件：|x|<tol, |y|<tol, |v|<tol, |angle|<tol, contact>0 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高频悬停或微小推力摆动，不推进 | `episode_length` 过长，`action` 频繁切换 | 加强 goal_proximity 与时间惩罚，适当降低燃料惩罚 |
| 高速直接撞击目标垫 | `y_velocity` 在着陆时很大，`reward` 常为负 | 强化 soft_landing 权重，提高 crash_prevention 的惩罚幅度 |
| 水平飞出视野 | `x_position` 在 episode 末尾偏离较大 | 检查 crash_prevention 是否对水平边界覆盖不足，增加强惩罚 |
| 始终无法学会使用主引擎，只靠方向引擎乱转 | 主引擎使用率极低，`body_angle` 持续旋转 | 适度降低燃料惩罚对主引擎的抑制，或对主引擎的有效使用给予方向性奖励（需谨慎） |
| 过早判定成功骗取奖励 | 中途中止但位置不在垫上，却获得 landing_success_bonus | 严格 success 条件：必须同时满足位置、速度、姿态和接触检测 |
| 姿态极度倾斜但勉强着陆 | `body_angle` 很大时获得 soft_landing 奖励 | 收紧 soft_landing 对角度的容忍阈值 |
