# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。飞行器从视口顶部中央附近出发，受到一个初始随机力的干扰。核心目标是尽快到达中央的目标着陆板并稳定停靠（低速度、正姿态、双足接触）。次要目标是尽量节省引擎推力。智能体必须学会在有限的离散推力动作下，平顺地接近目标、减速、保持姿态稳定，并实现安全的 soft landing 接触。不应把生存、保持平衡或无关区域的探索当作主目标。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`
confidence: high
reason: 核心任务是到达指定目标位置（中心目标板）并稳定着陆，这是典型的导航与到达行为。附属的燃料节省、速度约束、姿态保持都是辅助优化项，不影响其主导目标类别。动力学层有重力、推力与旋转，但与多足移动、双足步态无关，也不是单纯的生存平衡。因此选择 navigation_goal_reaching。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: `x_position`，相对于目标板的水平坐标，reward_usable: true
- obs[1]: `y_position`，相对于目标板高度的垂直坐标，reward_usable: true
- obs[2]: `x_velocity`，水平线速度，reward_usable: true
- obs[3]: `y_velocity`，垂直线速度，reward_usable: true
- obs[4]: `body_angle`，机体倾斜角度，reward_usable: true
- obs[5]: `angular_velocity`，角速度，reward_usable: true
- obs[6]: `left_support_contact`，左足接触标志（0.0/1.0），reward_usable: true
- obs[7]: `right_support_contact`，右足接触标志（0.0/1.0），reward_usable: true

所有 obs 维度均可用于奖励计算，但必须在适当上下文中使用（例如 contact 只在接近地面时才有意义）。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` — 不启动任何引擎，仅依靠惯性/重力
- action 1: `left_orientation_engine` — 启动一个侧向姿态引擎（产生旋转推力）
- action 2: `main_engine` — 启动主引擎（产生主要向上的推力，可能同时影响旋转）
- action 3: `right_orientation_engine` — 启动相反侧的姿态引擎

动作本身是离散的，不能直接输出连续推力大小。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` 中的 `body_settled` 部分（如果环境以此表征成功着陆并稳定）. 任务的期望结束状态是低速、双足接触、位置在目标板附近且机体稳定，所以达到这种状态并触发“settled”应视为成功。
- failure-like termination:
    - `crash_or_body_contact` — 机体与障碍物碰撞或身体非正常触地（除目标板外的接触）
    - `horizontal_position_outside_viewport` — 水平位置超出允许边界
- ambiguous termination: `body_not_awake_or_settled` 中的 `body_not_awake` 可能表示机体翻转或失去平衡（无法再唤醒），也可能是安全停止；但描述中“not awake or settled”暗示两种情形，其中 `body_not_awake` 更可能是不良终止（如翻倒失去动力），而 `body_settled` 是成功。需根据实际环境验证。
- truncation: 无明确时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （空字典，没有字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，包括任何隐藏的 success、failure、termination_reason

**重要**：由于 info 为空且禁止推断其内容，奖励函数不能直接依赖终止原因；只能通过 `obs` 和 `next_obs` 中的状态来启发式地判断接近成功（如接近目标、低速、接触）或惩罚失败（如坠毁前的剧烈状态变化），但不能假定一定能访问终止原因。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（完整8维向量）
- `action`（离散动作索引 0-3）
- `next_obs`（完整8维向量）
- `info` 中明确允许的字段（当前为无，禁止使用任何 info 键）
- `training_progress` 仅在 prompt 明确允许时使用（本环境未允许，默认禁用）

禁止使用：
- `original_reward`（即 masked 官方 reward）
- `official_reward` 或任何变体
- 未声明的 info 字段
- 未声明的 obs 切片（例如假定某些维度代表未说明的意义）

## 7. 可用于奖励函数的信号
- position: `x_position`, `y_position`
- velocity: `x_velocity`, `y_velocity`
- orientation: `body_angle`
- angular_velocity: `angular_velocity`
- contact: `left_support_contact`, `right_support_contact`
- action/engine: 通过动作选择可推论引擎使用情况（动作 0 为 no_engine，其他为使用引擎）
- other: 无其他

## 8. 不确定或不可用的信号
- 没有明确的 success / failure 标志
- 没有目标板精确几何范围或接触阈值
- 没有燃料计量/剩余推力的直接读数
- 没有时间步数或剩余步数
- info 字典完全不可用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 刚性物体（可能类似月球着陆器或小型飞行器）
  actuator_type: 离散推力（主引擎＋两个姿态引擎）
  contact_structure: 双腿/双足支撑（左右 support），接触信号为二值
primary_objectives:
  - 尽可能快地从起始区到达中心目标板并稳定着陆（位置接近零、速度接近零、双足接触）
secondary_objectives:
  - 最小化引擎使用次数（节省燃料）
  - 保持机体姿态接近水平（body_angle 靠近 0）
main_failure_risks:
  - 过早碰撞导致坠毁
  - 水平漂移超出视口边界
  - 着陆时速度过高或倾斜过大而翻倒
  - 在目标区上方振荡无法稳定
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: `goal_progress`
  purpose: 引导智能体向目标板水平、垂直方向靠近，减少距离，使任务收敛到目标区域。
  why_required: 这是导航目标达成的核心驱动力，没有它智能体无法学会向目标移动。
  usable_signals: [x_position, y_position, body_angle]（距离度量，可结合）
  risks: 若权重过大可能导致强烈推力、不平稳降落。

- role_id: `soft_landing`
  purpose: 在接近目标时要求低速度、双足接触、姿态平稳，确保稳定着陆而非撞击。
  why_required: 即使到达目标位置，若速度过高或未良好接触，实际仍会坠毁或被判失败，因此必需。
  usable_signals: [x_velocity, y_velocity, left_support_contact, right_support_contact, body_angle, x_position, y_position]
  risks: 过早惩罚速度可能阻碍最初向目标移动；需用位置门控（near goal）才能生效。

- role_id: `fuel_efficiency`
  purpose: 惩罚非必要的引擎使用，促使智能体学习使用自由落体和惯性滑行。
  why_required: 任务明确要求“using as little engine thrust as possible”，没有这一职责智能体会滥用引擎，可能快速但不经济。
  usable_signals: [action]
  risks: 若惩罚过重，智能体可能不敢点火而无法到达目标，因此需要平衡。

### 10.2 条件职责 conditional_roles
- role_id: `attitude_stabilization`
  condition_to_use: 当接近目标或正在着陆阶段时（可由位置、高度判断）激活，以鼓励机身角度接近0。
  usable_signals: [body_angle, angular_velocity]
  risks: 全程施加角度惩罚可能在早期限制必要的大角度机动，应只在靠近目标时加强。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: `alive_bonus` / `time_penalty`
  reason: 无法从info获得时间信息或存活信号；仅靠步数惩罚会与快速到达的reward冲突且难以正确校准，且环境并无显式时间步限制。
  forbidden_or_missing_signals: [无自带的存活计数器或时间信号]

- role_id: `contact_early_penalty`
  reason: 左/右脚接触可能在飞行途中偶然接触（如地形误触），但这些通常已被 crash 终止捕获；无需额外惩罚接触，因为着陆本身需要接触。
  forbidden_or_missing_signals: 没有地形类型标签识别哪些接触是合法的。

- role_id: `angular_velocity_penalty` (as standalone)
  reason: 角速度本身的惩罚可能抑制必要的旋转机动；最好结合角度稳定条件使用，而非全程单独施加。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_progress | x_position, y_position, body_angle | 目标距离精确值 | negative_distance, bounded_signal, quadratic_penalty | 可使用欧式距离或分量加权奖励靠近 |
| soft_landing | x_velocity, y_velocity, left_support_contact, right_support_contact, body_angle, x_position, y_position | 接触力/渗透深度 | gate (position < threshold), penalty on | 只在接近目标时启用；需要速度阈值与接触计数 |
| fuel_efficiency | action | 推力强度连续值 | discrete_action_cost | 简单对非零动作施加负奖励 |
| attitude_stabilization | body_angle, angular_velocity | — | gate (altitude/position), quadratic_penalty on angle, abs | 可用来鼓励水平姿态 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 智能体不点火，仅自由落体并坠毁 | 多数 episode 以 crash 结束，动作频率极低 | 降低节能惩罚，或引入先鼓励移动的课程 |
| 水平漂移过大，飞出边界 | x_position 频繁超出范围 | 增加水平距离惩罚或边界惩罚常数 |
| 到达目标上空但持续振荡，无法settle | y_position, y_velocity 来回摆动，长时间不终止 | 增强 soft_landing 部分的速度阻尼或接触奖励 |
| 着陆时侧倾翻倒 | body_angle 在着陆瞬间剧烈偏离0 | 加强 attitude_stabilization 在低高度的权重 |
| 过度使用主引擎导致燃料快速耗尽但仍不稳定 | 动作2频率极高 | 提高fuel_efficiency对主引擎的惩罚，或引入更细的动作代价差异 |
| 接触过早导致 crash 惩罚误判 | 在目标板外出现 single contact 并触发 crash | 确认终止条件；若无法修复，可在 reward 中不主动惩罚 contact，仅奖励双足同时稳定接触 |
