# 匿名环境理解卡片

## 1. 任务目标
本环境是一个二维飞行器/着陆器轨迹优化问题。  
智能体从靠近画面顶部中心的位置出发，受到一个随机初始力影响。  
**主要目标**：尽可能快地到达并平稳降落在中央的目标平台上（位置接近目标、速度趋近于零、姿态稳定、双支撑腿安全接触）。  
**次要目标**：最小化发动机推力使用，从而节省燃料。  
**不应混淆的目标**：不要求复杂机动或避障，核心是精准、节能的最终软着陆。

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`
- **confidence**: high
- **reason**: 任务的核心是到达并稳定着陆在中央平台，这是典型的“导航目标到达”问题。  
  虽然附带快速、省燃料的偏好，但它们不是与“是否到达”同等权重的并行目标，因此不属于多目标任务。  
  动力学上该任务具有**接近目标、减速、调整姿态、实现安全软接触**的显著特征，因此进一步匹配 `dynamics_subtype: goal_approach_and_soft_contact`。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: float32 (推断，通常连续观测用浮点数)
- 各维度含义（均为 reward usable）：
  - obs[0] **x_position**: 相对目标平台中心的水平坐标 → `reward_usable: true`
  - obs[1] **y_position**: 相对平台高度的垂直坐标 → `reward_usable: true`
  - obs[2] **x_velocity**: 水平线速度 → `reward_usable: true`
  - obs[3] **y_velocity**: 垂直线速度 → `reward_usable: true`
  - obs[4] **body_angle**: 机体朝向角度 → `reward_usable: true`
  - obs[5] **angular_velocity**: 角速度 → `reward_usable: true`
  - obs[6] **left_support_contact**: 左支撑腿接触标志（0/1） → `reward_usable: true`
  - obs[7] **right_support_contact**: 右支撑腿接触标志（0/1） → `reward_usable: true`

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- 各动作含义：
  - action 0: **no_engine** —— 不点火（无推力）
  - action 1: **left_orientation_engine** —— 点燃左侧姿态发动机（产生旋转力矩，推左）
  - action 2: **main_engine** —— 点燃主发动机（产生向上推力）
  - action 3: **right_orientation_engine** —— 点燃右侧姿态发动机（产生相反旋转力矩，推右）

## 5. step 与终止条件分析
### 5.1 终止模式
根据源码中 `terminated` 的逻辑：
- **success-like termination**:  
  - `body_not_awake_or_settled`（机体不再活跃或已稳定下来）——在目标平台稳定着陆时通常触发，可视为潜在成功终止。  
  - `crash_or_body_contact` 中的一部分：如果两条腿都与平台接触且速度、角度满足安全条件，可能触发终止并成功，但任务源码未区分成功/失败，故不能直接当作成功标志。
- **failure-like termination**:  
  - `horizontal_position_outside_viewport`（水平位置超出画面边界）——明确失败。  
  - `crash_or_body_contact` 中的非平台接触（例如撞地、侧翻）——明确失败。
- **ambiguous termination**:  
  - `body_not_awake_or_settled` 也可能在失败状态（如翻转昏迷）出现，因此单独依赖此条件不可靠。
- **truncation**:  
  - 源码中未出现 `truncated`，仅 `terminated` 被返回，`info` 为空，因此无其他截断信息。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  `info` 为空，`terminated` 本身也未分解为成功/失败。
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 无（info 固定为 `{}`）
- **forbidden_or_uncertain_info_fields**: 所有未声明的 info 字段均不可用；尤其 **不能假设存在 `success`、`failure`、`termination_reason` 等字段**。

## 6. reward 函数接口契约
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
### 允许使用
- `obs` (当前观测，通常不使用，但可用)
- `action` (当前动作)
- `next_obs` (下一时刻观测，**强烈推荐作为奖励计算的基础**)
- `info` (仅空字典，无可靠字段)
- `training_progress` (仅当 prompt 明确说明需要时才用，此处一般不依赖)

### 禁止使用
- `original_reward` —— 被明确标记为 `MASKED`，必须忽略
- `official_reward` 或任何来自环境的直接奖励信号
- 任何未在观测空间中声明的 info 键值
- 基于 `terminated` 或 `truncated` 的外部信号（它们不是 `compute_reward` 的参数）

## 7. 可用于奖励函数的信号
以下信号均从 `next_obs` 获取，部分可利用 `obs` 进行 delta 计算（如速度变化）：

- **位置信号**:
  - `next_obs[0]` (x 距目标)
  - `next_obs[1]` (y 距平台)
- **速度信号**:
  - `next_obs[2]` (vx)
  - `next_obs[3]` (vy)
- **姿态信号**:
  - `next_obs[4]` (角度)
  - `next_obs[5]` (角速度)
- **接触信号**:
  - `next_obs[6]` (左腿接触)
  - `next_obs[7]` (右腿接触)
- **动作/推力信号**:
  - `action` (可判断是否使用主发动机或姿态发动机，用于推力惩罚)
- **其他可能衍生信号**:
  - 综合距离 `sqrt(x_pos^2 + y_pos^2)`
  - 速率 `sqrt(vx^2 + vy^2)`
  - 角度绝对值 `abs(angle)`

## 8. 不确定或不可用的信号
- **不可用**：
  - 明确的“成功着陆”标志（未提供）
  - 明确的“坠毁”标志
  - 剩余燃料或推力积分（未提供）
  - 环境提供的原始奖励
- **不确定**：
  - 身体稳定性/清醒状态（`body_not_awake_or_settled` 仅用于终止，未作为观测传入，不可在奖励中使用）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_lander
  actuator_type: main_thruster + lateral_orientation_thrusters
  contact_structure: two_leg_contacts (left + right pads)
primary_objectives:
  - 最终水平位置接近 0，垂直位置接近 0（降落在平台中心）
  - 着陆时速度接近 0（软着陆）
  - 机体角度接近 0（竖直）
  - 双腿与平台稳定接触
secondary_objectives:
  - 减少主发动机和姿态发动机的使用（省燃料）
  - 在尽可能短的时间内完成着陆（隐含快速性，可通过每步微小负奖励鼓励）
main_failure_risks:
  - 高速垂直撞击导致坠毁
  - 过大倾角导致侧翻并触发 `crash_or_body_contact` 失败
  - 水平漂移超出视口边界
  - 长时间悬停消耗过多燃料但未完成着陆（可能导致后续环境截断）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity_and_settling`
  - **purpose**: 引导智能体移动至平台正上方并降低到接触高度，同时确保稳态（低速度）
  - **why_required**: 任务核心是到达并安稳着陆，无可替代
  - **usable_signals**: `next_obs[0]`, `next_obs[1]`, `next_obs[2]`, `next_obs[3]`, `next_obs[4]`, `next_obs[5]`
  - **risks**: 如果仅奖励接近而忽略速度，可能导致高速撞击；必须结合速度惩罚
- **role_id**: `soft_landing_and_contact`
  - **purpose**: 在最终接触阶段奖励双腿接触、低速度和竖直姿态
  - **why_required**: 终止条件中“安全接触”是关键，缺乏此信号可能导致智能体不学习精准着陆
  - **usable_signals**: `next_obs[6]`, `next_obs[7]`, `next_obs[2]`, `next_obs[3]`, `next_obs[4]`
  - **risks**: 若过早给予大量接触奖励，可能鼓励提前拍地而不减速
- **role_id**: `orientation_stabilization`
  - **purpose**: 保持小角度、小角速度
  - **why_required**: 倾角过大会导致碰撞、腿无法同时触地，进而失败
  - **usable_signals**: `next_obs[4]`, `next_obs[5]`
  - **risks**: 太强可能过度抑制机动能力

### 10.2 条件职责 conditional_roles
- **role_id**: `thrust_penalty`
  - **condition_to_use**: 当希望鼓励燃料效率时启用；对于初始训练阶段建议先使用温和惩罚或后期逐步强化
  - **usable_signals**: `action`（判断是否为 1,2,3）
  - **risks**: 过大的惩罚可能导致智能体不敢使用主发动机，无法减速坠落；需平衡 goal_proximity 与 thrust_penalty
- **role_id**: `survival_bonus_or_time_penalty`
  - **condition_to_use**: 若期望加快完成任务，可在每一步给予小的负奖励（或正奖励翻转）；但需注意过早终止可能造成训练不稳定
  - **usable_signals**: none (只需每步固定量)
  - **risks**: 可能导致智能体过早尝试“自杀”以结束轨迹；因此必须在着陆接触奖励足够大时才能引入

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `explicit_success_bonus`
  - **reason**: 环境未提供成功标志，且 `terminated` 信息不传入奖励函数，即使通过观测间接判断成功状态也可能引入错误信号（如在失败沉睡时误给成功奖励）
  - **forbidden_or_missing_signals**: `success flag` (missing)
- **role_id**: `crash_penalty_from_termination`
  - **reason**: 无法在 `compute_reward` 中获取终止原因，不能根据是否终止给予立即惩罚
  - **forbidden_or_missing_signals**: `terminated` flag, `failure_reason` (not accessible)

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity_and_settling | `next_obs[0]`, `next_obs[1]`, `next_obs[2]`, `next_obs[3]` | none | `distance_penalty`, `velocity_penalty` (e.g., `-dist - k*|v|`) | 核心驱动靠近并减速 |
| soft_landing_and_contact | `next_obs[6]`, `next_obs[7]`, `next_obs[2]`, `next_obs[3]`, `next_obs[4]` | none | `contact_bonus` when `both legs contact`, `bonus` when `vy` small & `|angle|` small | 鼓励最终软着陆姿态，可仅在接近时给予 |
| orientation_stabilization | `next_obs[4]`, `next_obs[5]` | none | `quadratic_penalty` on angle and angular velocity | 持续稳定 |
| thrust_penalty | `action` | fuel consumption | `per_step_cost` for actions 1,2,3 (possibly weighted by type) | 条件性强，可逐渐增大 |
| survival_bonus_or_time_penalty | none (step count) | episode length | `small_positive_reward` (alive bonus) | 谨慎使用，易误导 |
| explicit_success_bonus | – | `success_flag` | – | 禁用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 始终悬停在高空，不敢下降 | 平均 y 位置远离 0，主发动机几乎不使用 | 加大目标接近奖励，降低推力惩罚，或增加负高度惩罚 |
| 高速垂直撞击平台 | 终止时 vy 很大，双腿接触但倾角大 | 强化速度惩罚（尤其在接近平台时