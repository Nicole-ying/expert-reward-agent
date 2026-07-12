# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 飞行器 / 着陆器轨迹优化问题。  
主体从视野顶部中央附近出发，并带有随机初速度。核心目标是**尽快到达并稳定停靠在画面中央的目标平台上、且保持姿态平稳**；次要目标是在此过程中尽量减少引擎推力消耗。需要关键控制能力：接近目标、减速、维持直立姿态并通过左右支撑点与平台安全接触。  

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`
- **confidence**: high
- **reason**:  
  任务的首要目标是到达一个指定的固定目标位置（中央平台）并稳定停留。速度与能耗均为从属优化项，不存在目标间权重相当且不可调和的情况，因此归为“导航/到达目标”大类，而非多目标或存活类。  

- **dynamics_subtype**: `goal_approach_and_soft_contact`  
  主体需要接近平台、减速、保持姿态并最终以低速与平台柔性接触，符合“接近目标并低速、稳定接触/停靠”的特征。  

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: (8,)  
- **dtype**: `float32`  

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|----------------|
| 0 | x_position | 相对于目标平台的水平坐标 | true |
| 1 | y_position | 相对于平台高度的高度坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 主体倾角（方向角） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑接触标志（0/1） | true |
| 7 | right_support_contact | 右支撑接触标志（0/1） | true |

注：接触标志虽然在环境中可能是 `1.0` 或 `0.0`，但该维度对奖励设计有用。

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  

| 动作编号 | 名称 | 含义 | 说明 |
|----------|------|------|------|
| 0 | no_engine | 不点火 | 被动滑行 |
| 1 | left_orientation_engine | 左侧姿控引擎 | 产生角度变化 / 姿态调整 |
| 2 | main_engine | 主引擎 | 产生主要向下推力，影响垂直速度 |
| 3 | right_orientation_engine | 右侧姿控引擎 | 产生相反侧姿态调整 |

本离散动作空间没有幅度控制，每次执行一个动作施加固定强度的冲量。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` —— 当主体运动停止且可能稳定在平台上时触发。该条件**并非明确只包含成功**，可能包含因其他原因静止（如坠毁后静止），需要结合位置与接触状态判断。
- **failure-like termination**:  
  `crash_or_body_contact` —— 主体与非目标地面/结构物接触导致坠毁；  
  `horizontal_position_outside_viewport` —— 主体水平飘出视野边界。
- **ambiguous termination**:  
  `body_not_awake_or_settled` 本身未直接区分成功与失败。
- **truncation**:  
  来源于 `step` 源码中的 `terminated` 标志，`truncated` 始终为 `False`，没有超时截断。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 空（`step` 返回的 `info` 为空字典 `{}`）  
- **forbidden_or_uncertain_info_fields**: 所有 `info` 内未声明的字段，包括假设的 `success`、`failure`、`termination_reason` 等均不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs` —— 当前观测
- `action` —— 执行的动作
- `next_obs` —— 下一步观测（含变化后的位置、速度、姿态、接触信息）
- `info` 中**明确允许**的字段（当前为无字段可用）
- `training_progress` —— 仅在 prompt 明确允许时使用（当前未提及，暂不用）

**禁止使用**：
- `original_reward`（已掩码）
- `official_reward` 类似的任何内部奖励信号
- 未声明的 `info` 字段（如 `success`、`failure`）
- 未在任务说明中给出的 `obs` 切片语义

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position`（相对平台坐标，可用以引导接近与保持在目标上方）
- **velocity**: `x_velocity`, `y_velocity`（减速靠近平台）
- **orientation**: `body_angle`, `angular_velocity`（维持姿态平稳）
- **contact**: `left_support_contact`, `right_support_contact`（稳定接触检测）
- **action/engine**: 可以针对推力动作施加惩罚（如 `action != 0`）以鼓励节省燃料
- **other**: 无额外信号。

## 8. 不确定或不可用的信号
- 明确的成功/失败标志（`info` 中不存在）
- 平台绝对位置（仅提供相对位置，隐含平台位置固定于原点）
- 剩余的燃料量（未提供）
- 绝对的引擎推力值（动作仅为离散开关）
- 平台的形状、宽度信息（未显式给出，但可通过接触推断）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two support legs/skids
  actuator_type: discrete main engine + two side orientation engines
  contact_structure: left and right touch sensors, platform at center
primary_objectives:
  - reach and settle on the central target pad (i.e., x≈0, y≈0, stopped)
  - maintain upright orientation and stable contact on the pad
secondary_objectives:
  - minimize engine thrust usage (fuel efficiency)
  - minimize time to reach the goal (speed, implicitly via fast settling)
main_failure_risks:
  - crashing into ground or non-pad surfaces (body contact other than pad)
  - drifting out of the horizontal viewport
  - failing to kill velocity, leading to bounce or slide off the pad
  - over-correction of orientation causing tilt and crash
  - early termination while not actually on the pad (misinterpreted settle)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity`  
  **purpose**: 驱动主体向目标平台靠近并保持在平台附近。  
  **why_required**: 没有到达目标的引导，任务无法完成。  
  **usable_signals**: `x_position`, `y_position`（距平台的距离）  
  **risks**: 若只奖励接近而忽略速度，可能导致高速冲过或撞毁。

- **role_id**: `soft_landing_and_settling`  
  **purpose**: 在接近目标后降低速度并稳定停留在平台上。  
  **why_required**: 仅到达位置不足以成功，还需减速并稳定，否则会飞出或弹跳。  
  **usable_signals**: `x_velocity`, `y_velocity`, `x_position`, `y_position`, contact flags  
  **risks**: 过度强调低速可能使智能体过早悬停，浪费燃料与时间。

- **role_id**: `orientation_penalty`  
  **purpose**: 惩罚主体倾斜，鼓励保持垂直姿态。  
  **why_required**: 翻倒会使接触传感器失效、直接导致坠毁，且稳定平台接触需要直立。  
  **usable_signals**: `body_angle`, `angular_velocity`  
  **risks**: 若权重过大，可能阻碍必要的姿态调整机动。

### 10.2 条件职责 conditional_roles
- **role_id**: `engine_usage_penalty`  
  **purpose**: 鼓励在完成任务的前提下尽可能少使用引擎。  
  **condition_to_use**: 在任务的主要完成阶段，当接近平台后或全局均可加入，但应在到达目标前不压制动作为好；可分段加权。  
  **usable_signals**: `action` (离散动作是否为非零)  
  **risks**: 过早惩罚可能使智能体不探索；过重可能导致不敢使用引擎进行修正而坠毁。

- **role_id**: `safe_contact_bonus`  
  **purpose**: 当双腿均与平台接触且姿态良好时给予额外正奖励，强化稳定着陆。  
  **condition_to_use**: 当观察到 `left_support_contact` 和 `right_support_contact` 同时为 1 且位置在平台附近时。  
  **usable_signals**: `left_support_contact`, `right_support_contact`, `x_position`, `y_position`  
  **risks**: 可能在平台外意外触发接触（如果环境存在其他可接触物），需结合位置判断。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `time_penalty`  
  **reason**: 环境没有提供时间步计数信号，且虽可通过累计步数替代，但“尽快”已隐含在 shaping（如逼近目标）中，不需要显式时间段惩罚，反而可能引起探索不足。  
  **forbidden_or_missing_signals**: 无原生的时间步索引；可用额外内建计数器实现，但无必要且容易冲突。

- **role_id**: `progress_export_constant`  
  **reason**: 环境未提供成功率或进度反馈的额外 info 字段，无法直接依据“是否成功”给予稀疏奖励，需要依赖观察重构。  
  **forbidden_or_missing_signals**: info 中无可信 success/failure 信号。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position | — | dense_state_signal, distance_to_target, gaussian_like | 可用距离平滑奖励 |
| soft_landing_and_settling | x_velocity, y_velocity, x_position, y_position, contacts | — | dense_state_signal, bounded_signal, quadratic_penalty, contact-weighted | 结合距离与速度制作“着陆区”奖励 |
| orientation_penalty | body_angle, angular_velocity | — | dense_state_signal, quadratic_penalty | 对倾角与角速度施罚 |
| engine_usage_penalty | action | — | discrete_action_penalty, linear_penalty | 当 action ≠ 0 时给予小幅负奖励 |
| safe_contact_bonus | left_support_contact, right_support_contact, x_position, y_position | — | conditional_bonus, gating (both contact and proximity) | 双接触且 x,y 接近零时给正向奖励 |
| (avoid) time_penalty | — | elapsed steps in episode | — | 避免引入，容易与速度 shaping 冲突 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 主体经常越过平台后飘出视野 | `x_position` 绝对值持续增大，`body_angle` 大，`terminated` 因 horizontal outside viewport | 增加惩罚侧向速度和倾角，引入横向引导力 |
| 到达平台上方但弹跳后坠毁 | `y_velocity` 在接触瞬间过高，接触信号一闪即过，`terminated` 因 crash | 加强软着陆速度惩罚，或基于高度与速度的二次项 |
| 智能体只开主引擎垂直悬停而不移动 | `x_position` 偏差大，但 `y_position` 始终小，动作几乎只有主引擎 | 增加目标引导的奖励，或对无横向动作给予微小惩罚 |
| 双腿接触但姿态不稳定最终倾倒 | `body_angle` 和 `angular_velocity` 波动大，接触后仍终止 | 提高姿势惩罚，以及鼓励接触后静止的奖励 |
| 由于过度节省燃料而未能到达平台 | 奖励高但任务失败，`x_position`/`y_position` 远，`action` 很少非零 | 降低燃油惩罚权重，或改在末期才启用 |
| 在平台附近盘旋但永不降落 | `y_position` 小但不触发 settle（始终有点速度或轻微移动） | 加入基于位置与速度的“dock”奖励，当位置近、速度低时给出强正奖励 |

此环境卡片提供了完整的事实依据与奖励设计思路，后续 Reward Generator 可据此生成符合契约的奖励函数，Reflection Agent 也能用于诊断训练行为。