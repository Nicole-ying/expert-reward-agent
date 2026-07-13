# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器着陆任务。智能体需要控制一个刚体（具有主发动机和两个姿控发动机），从上方某处出发，尽可能快地到达画面中央的着陆垫上并稳定停靠。主要目标是到达目标位置并实现软着陆（低速、姿态接近水平、左/右支撑腿稳定接触）。次要目标包括：最小化发动机使用（燃料效率）、缩短到达时间、保持姿态平稳。不应与纯飞行控制或仅避免碰撞的任务混淆——最终必须到达并驻留于目标垫。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**
confidence: **high**
reason: 任务最高级目标是通过导航到达指定目标点并稳定停驻。附属目标（省燃料、快速、姿态稳定）服务于主目标的质量，但不会取代导航到达性质。不存在多目标权重相当冲突的多目标情形。

## 3. 观察空间 observation_space
- type: Box（连续向量）
- shape: (8,)
- dtype: 通常为 float32
- 各维物理含义与可用性：

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | x_position | 相对于着陆垫中心的水平偏移 | true |
| 1 | y_position | 相对于垫面高度的垂直偏移 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 机体倾角（弧度） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑腿接触标志 (0/1) | true |
| 7 | right_support_contact | 右支撑腿接触标志 (0/1) | true |

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 每个动作的含义：

| action index | name | meaning |
|-------------|------|---------|
| 0 | no_engine | 所有发动机均不工作 |
| 1 | left_orientation_engine | 点火左方位发动机 |
| 2 | main_engine | 点火主发动机（产生向上推力） |
| 3 | right_orientation_engine | 点火右方位发动机 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  身体稳定停留在着陆垫上可能触发终止（terminated），但终止条件中没有显式成功标志。从任务目标推断，当`body_not_awake_or_settled` 且满足位置接近原点、低速、两腿接触时，可视为成功软着陆。
- **failure-like termination**:  
  `horizontal_position_outside_viewport`（水平出界）明确失败。`crash_or_body_contact` 部分含义为不安全的撞击（如猛烈撞击地面或非支撑部位接触），也视为失败。
- **ambiguous termination**:  
  `body_not_awake_or_settled` 可能是成功稳定着陆（我们希望），也可能是早期无动作导致的“unawake”状态。需要结合位置和接触特征区分。
- **truncation**:  
  未提到时间截断，但不排除有最大步数。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields:  
  当前 step 返回 `info={}`，因此无法使用任何 info 字段。
- forbidden_or_uncertain_info_fields:  
  任何未在声明中出现的 info 字段均不可用，禁止依赖 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- `obs`：动作前的完整观测向量（8维）
- `action`：刚执行的动作索引
- `next_obs`：动作后的观测向量（8维）
- `info`：始终为空字典，不可从中提取字段（无可用字段）
- `training_progress`：**不明确允许，不得使用**（本提示未授权）

**禁止使用：**
- `original_reward`：原始奖励被遮盖，不可训练时访问
- 任何 `info` 中未明确声明的字段（包括 success/failure 标志）
- 任何未在此文档中列出的外部状态（如真实时间、步数计数器等）
- `next_obs` 或 `obs` 中未描述的额外维度

## 7. 可用于奖励函数的信号
- **位置信号**：`x_position`、`y_position`（相对于目标垫中心和高度的偏移）。可直接用于衡量与目标的距离。
- **速度信号**：`x_velocity`、`y_velocity`。可用于鼓励减速或惩罚撞击速度。
- **姿态信号**：`body_angle`、`angular_velocity`。可用于保持机体水平。
- **接触信号**：`left_support_contact`、`right_support_contact`。两者同时为1通常表示稳定着陆，可作为成功指示或软着陆奖励。
- **动作/发动机信号**：`action` 是离散值，可构建燃料惩罚（动作非 0 则惩罚）。无直接推力值，但动作索引含义已知。
- **其他**：无。

## 8. 不确定或不可用的信号
- 无显式 `is_success` / `is_failure` 标志（info 为空）。
- 无显式剩余时间或步数。
- 无直接推力/燃料率数值（只有动作类型，无法区分推力大小，只能惩罚动作是否使用引擎）。
- 终止条件触发逻辑未暴露，不能用于奖励函数（只可在训练循环外部使用，但 compute_reward 内不可见）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body (2D, with main + two orientation thrusters)
  actuator_type: on‑off thrusters (no throttle)
  contact_structure: two-point support legs with binary contact sensors
primary_objectives:
  - reach target pad (minimize final distance)
  - achieve soft landing (low vertical speed, stable contact)
secondary_objectives:
  - minimize fuel usage (action penalties)
  - minimize time to reach (implicit in fast settling)
  - maintain upright orientation (low angle)
main_failure_risks:
  - horizontal drift out of viewport
  - high-speed impact (crash)
  - unstable or incomplete contact (only one leg on pad)
  - excessive fuel consumption without progress
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_proximity**
  purpose: 持续减少飞行器与目标垫的相对距离，引导向原点运动。
  why_required: 核心导航驱动信号，缺少则没有方向性指引。
  usable_signals: [x_position, y_position]
  risks: 仅用距离会导致“冲击”目标，而不减速，需与速度调控结合。

- **role_id: soft_landing**
  purpose: 在接近目标且两支撑腿接触时，给予正奖励；同时惩罚高速垂直撞击。
  why_required: 任务要求“settle at target pad”，仅导航无法保证安全着陆。
  usable_signals: [y_velocity, left_support_contact, right_support_contact, x_position, y_position]（可结合位置门控）
  risks: 过早给予接触奖励可能鼓励提前无控掉落；需要位置/速度门控。

- **role_id: velocity_damping**
  purpose: 随着目标距离减小，鼓励减小水平与垂直线速度。
  why_required: 避免过冲和猛烈撞击，是实现软着陆的必要条件。
  usable_signals: [x_velocity, y_velocity, x_position, y_position]
  risks: 可能过度减速导致悬停不前。

- **role_id: orientation_penalty**
  purpose: 惩罚机身倾斜（偏离水平），鼓励平稳着陆姿态。
  why_required: 倾斜可能导致单腿先触地、不稳定翻滚，损害成功着陆。
  usable_signals: [body_angle, angular_velocity]
  risks: 对于初期远距离导航，轻微惩罚即可，过强会干扰导航。

### 10.2 条件职责 conditional_roles
- **role_id: fuel_efficiency**
  purpose: 惩罚使用发动机的动作（action ∈ {1,2,3}），鼓励节能。
  condition_to_use: 始终可用，但在接近目标且需要微调时权重应适度（避免因节约而弃稳）。
  usable_signals: [action]
  risks: 过强的省燃料惩罚可能使智能体不敢点火，导致悬停失败或漂出边界。

- **role_id: early_settlement_bonus**
  purpose: 当检测到很可能成功软着陆时（两腿接触、低速度、近原点、小角度）给予一次性奖励，鼓励尽快稳定。
  condition_to_use: 需根据 next_obs 判定（非基于 info），要求多信号严格门控，避免误触。
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 门控不严可能导致假阳，智能体可能故意撞击腿部而不真正到达目标。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: time_penalty**
  reason: 无步骤数或时间剩余信号提供，无法直接计算。不可依赖 training_progress，故禁用。
  forbidden_or_missing_signals: [step_count, time_left]

- **role_id: explicit_success_reward**
  reason: info 中无可用的 `success` 标志，且无法安全推断终止性质。禁用对 info 字段的依赖。
  forbidden_or_missing_signals: [info.success, info.failure]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position | (none) | negative Euclidean distance, smooth_l1, squared distance | 可添加门控以避免在接触后仍强烈拉动 |
