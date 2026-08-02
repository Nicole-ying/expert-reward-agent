# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主体是一个二维飞行器，从视口顶部中心附近受随机初始力出发，核心目标是**尽快、平稳地降落在中心目标垫上**。着陆要求在两个支撑腿均接触目标垫的同时，保持低速度、姿态稳定。附属优化目标是**尽量少用引擎推力**，即整个过程中减少不必要的姿态调整和主引擎点火次数，但必须以安全着陆为前提。任务不包括巡航、越障或抓取等无关动作。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求“reach and settle at a central target pad”，即到达指定目标位置并停靠，着陆点是唯一的、静态的。附属目标（快、省燃料）是对轨迹质量的约束，不构成与“到达”同等权重的多目标冲突。控制类型为离散动作，但底层是连续物理，符合导航目标到达 → 接近阶段与软着陆的条件组合。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: **x_position** – 机体相对于目标垫水平坐标（中心为0），reward_usable: true
- obs[1]: **y_position** – 机体相对于垫面高度，reward_usable: true
- obs[2]: **x_velocity** – 水平线速度，reward_usable: true
- obs[3]: **y_velocity** – 垂直线速度，reward_usable: true
- obs[4]: **body_angle** – 机体倾角（弧度），reward_usable: true
- obs[5]: **angular_velocity** – 角速度，reward_usable: true
- obs[6]: **left_support_contact** – 左支撑腿是否接触（0/1），reward_usable: true
- obs[7]: **right_support_contact** – 右支撑腿是否接触（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作0: no_engine – 无引擎输出（惯性漂行）
- 动作1: left_orientation_engine – 点燃左侧姿态引擎（产生角/线加速度，调节姿态）
- 动作2: main_engine – 点燃主引擎（产生主体坐标系推力，通常向上或前方）
- 动作3: right_orientation_engine – 点燃右侧姿态引擎（与左引擎相反方向）

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**: 机体与地面或垫面发生非期望碰撞（可能是高速撞击、侧翻等），视为**失败**。
- **horizontal_position_outside_viewport**: 水平位置超出视口范围，视为**失败**。
- **body_not_awake_or_settled**: 机体进入休眠状态或已稳定停止（包括成功着陆后速度归零），属于 **ambiguous termination**；需结合左右接触标志和相对位置判断是否为成功着陆。
- 本环境不设置显式最大步数截断（truncation），若 episode 自然终止前无上述触发，则由环境内部最大 step 截断，但不通过 info 提供。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 在 step 源码中返回空字典，禁止使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有字段均不存在

**重要**: 成功或失败只能通过 next_obs 间接推断。当 episode 终止时，可以根据 next_obs 的接触标志、位置、速度组合判定是否为成功。例如：左右 support 均接触且|x_position|极小、|y_position|接近0且绝对速度很低，则可认为是成功着陆。此类信号属于 derived_possible，可在奖励中使用但需谨慎组合。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
允许使用：
- obs（当前状态，shape(8,)）
- next_obs（下一时刻状态，shape(8,)）
- action（0~3）
- info 中明确允许的字段（当前为无）
- training_progress 仅当 prompt 明确允许时才用

禁止使用：
- original_reward
- official_reward
- 任何未声明的 info 字段（包括但不限于 success, failure, termination_reason 等）
- 未声明的 obs 切片含义

## 7. 可用于奖励函数的信号
- **位置**:
  - `x_position`, `y_position` → 计算与目标（0,0）的欧氏距离 `dist = sqrt(x^2 + y^2)`
  - 距离降低表示向目标靠近。
- **速度**:
  - `x_velocity`, `y_velocity` → 总体速度大小 `speed = sqrt(vx^2 + vy^2)`，可用于鼓励减速。
- **姿态**:
  - `body_angle` → 倾角，理想着陆姿态应为接近0（水平），可用角度绝对值作为 penalty。
  - `angular_velocity` → 角速度，小为宜。
- **接触**:
  - `left_support_contact`, `right_support_contact` → 可推断着陆状态。两个腿同时接触且位置在目标附近、速度低，可视为成功软着陆。
- **动作/引擎**:
  - 动作 `action` (0~3)，可区分是否使用引擎、哪个引擎，用于计算推力惩罚。
- **衍生信号 (derived_possible)**:
  - 成功着陆事件：由 `next_obs` 满足 (left_contact & right_contact) 且 `dist < 阈值` 且 `speed < 阈值` 且 episode 终止（可由环境自动截断或 stable 终止推断，但无法显式获得 terminated flag。在实践中可以通过 `original_reward` 为0或特定值不成依赖，我们只能设计奖励函数在正常步中给予奖励，而非在终止步专门奖励。我们可以利用终止时 next_obs 最后一个画面给予一次性高分，但这要求知道是否终止。由于无法获取 terminated 标志，最好**不在每一步使用“成功事件”奖励**，而是通过密集的接近、减速、姿态奖励来引导，并依靠环境终止条件自然结束，这样更安全。若实在需要，可由训练框架检测 episode 结束时最后一帧 next_obs 并给予额外奖励，但这不在标准 compute_reward 内。）

## 8. 不确定或不可用的信号
- 显式的 success/failure/termination_reason 标志：不可用，info空。
- 是否发生 crash_or_body_contact 的具体类型：不可直接获得。但可通过突然大幅位置变化或异常接触标志推断，但不可靠。
- 视口外判断：只能通过位置绝对值超过某范围（如 x > 1.5 等，需环境边界值）推断，边界可能需从初始状态范围估计。此信号可能衍生但环境未提供确切阈值，作为不确定信号。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body (lander-like), single body with two landing supports
  actuator_type: discrete thrusters (main engine + two orientation engines)
  contact_structure: left and right support contacts with ground/pad
primary_objectives:
  - Reach target pad (minimize distance to (0,0))
  - Achieve safe soft contact (both supports touching, low velocity, upright angle)
secondary_objectives:
  - Minimize engine usage (fuel efficiency)
  - Approach quickly (time efficiency)
main_failure_risks:
  - Crashing due to high impact velocity or wrong angle
  - Falling off viewport laterally
  - Oscillating uncontrollably due to overuse of orientation engines
  - Hovering near target without touching down (truncated)
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id: delta_distance_to_target**
  purpose: 鼓励连续步间向目标垫靠近，形成正梯度
  why_required: 核心目标就是到达，距离变化量是最直接的学习信号，且能避免原地悬停得分
  usable_signals: [x_position, y_position] （当前和下一步计算距离之差）
  risks: 如果同时有其他奖励（如速度奖励）可能相互干扰；若接近过快导致撞击，可能需配合安全门控

- **role_id: approach_speed_bonus_with_safety_gate**
  purpose: 在安全距离和低危险速度范围内奖励快速接近，同时通过门控防止撞击
  why_required: 任务要求“as fast as possible”，但不鼓励超速撞击；需要在接近目标时减速，故需要区分阶段
  usable_signals: [x_position, y_position, x_velocity, y_velocity]
  risks: 门控函数选择不当可能导致奖励稀疏或鼓励危险行为，应使用 hinge 型惩罚拦截高速

### 10.2 条件职责 conditional_roles
- **role_id: soft_landing_stability_penalty**
  condition_to_use: 当 agent 处于目标垫附近（dist < 阈值）或已有一腿触地时启用
  usable_signals: [y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 如果在不接近目标时惩罚角速度和垂直速度，可能阻碍正常机动。需根据到目标的距离动态加权。

- **role_id: engine_efficiency_penalty**
  condition_to_use: 贯穿全程，但权重远低于主距离奖励；任务明确要求“as little engine thrust as possible”
  usable_signals: [action]
  risks: 若设定权重过高，可能导致 agent 不使用引擎而无法到达目标；应当使引擎使用惩罚相对主奖励较小，如 -0.01 每步点燃引擎。

- **role_id: terminal_touchdown_bonus (derived)**
  condition_to_use: 若训练框架允许在 episode 结束时根据最终 `next_obs` 判定成功并给予一次性奖励
  usable_signals: [next_obs] 推导 (contact, dist, speed)
  risks: 无法在标准 `compute_reward` 中获取 terminated 信号，因此这个职责强烈依赖外部包装器，不推荐在纯奖励函数内实现；可考虑在 early stop 后手动加分，但本分析认为该职责当前环境中为 conditional 且实现复杂，不建议作为核心。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: dense_speed_penalty_global**
  reason: 持续惩罚速度会抑制向目标的机动，与“尽快到达”冲突
  forbidden_or_missing_signals: 无适应特例

- **role_id: sparse_exploration_bonus**
  reason: 本任务状态空间小，目标明确，无需额外探索奖励
  forbidden_or_missing_signals: 不需要

- **role_id: survival_time_reward**
  reason: 不适用；生存并非目标，长时间存活而不到达毫无意义
  forbidden_or_missing_signals: 无

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| delta_distance_to_target | obs[0:2], next_obs[0:2] (x,y) | None | delta = dist_prev - dist_next, linear improvement | 核心信号，需配合速降防撞 |
| approach_speed_bonus_with_safety_gate | obs[2:4], obs[0:2] | None | speed_reward = (dot(velocity, to_target_direction) * gate(dist)) / max_speed, gate can be hinge at dist_threshold | 鼓励朝目标移动，但近垫时关闭 |
| soft_landing_stability_penalty | obs[3] (vy), obs[4] (angle), obs[6:8] (contacts) | None | penalty = hinge_abs(vy, soft_limit) + hinge_abs(angle, soft_limit), scaled by proximity to target | 近垫时启用，防高速撞击 |
| engine_efficiency_penalty | action (0-3) | None | penalty = -c if action != 0 else 0 | 小常量惩罚使用引擎 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停在目标上方不下降 | y_position 保持小正值，vy≈0，接触为0，episode 被截断 | 增加接近阶段的下降奖励或调整 delta_distance 的竖直分量权重 |
| 降落后弹跳或翻倒 | 接触标志交替闪烁，body_angle 或 angular_velocity 突然大幅变化 | 加强软着陆惩罚（大垂直速度和角速度），降低下落速度限制 |
| 过度使用姿态引擎导致能量浪费并失控 | 角速度持续高，动作频繁选择1和3，而距离减少缓慢 | 提高 engine_efficiency_penalty 权重，或限制姿态引擎使用频率 |
| 直接侧向飞出视口 | x_position 迅速偏离0并超出边界，无减速 | 增加离垫时的横向速度惩罚，或在边界附近大幅惩罚 |
| 高速撞击目标垫后终止 | 触地时 y_velocity 很大，接触标志为1，episode 终止但未得高分 | 严格限制近垫时最大允许速度，应用 soft_landing_stability_penalty |
