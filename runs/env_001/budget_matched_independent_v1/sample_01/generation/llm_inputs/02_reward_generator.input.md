# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器着陆轨迹优化任务。飞行器从视野上方的随机初始位置出发，受到初始随机力扰动。核心目标是**控制飞行器安全、平稳地降落在画面中央的目标平台上**。具体而言：
- **主要目标**：使飞行器到达平台正上方并实现稳定着陆（水平与垂直速度趋于零、姿态竖直、双腿接触平台）。
- **附属目标**：尽可能快地完成着陆，同时**最小化发动机推力使用**（节省燃料）。
- **不可混淆**：这不是纯粹的导航任务——着陆后的稳定性（速度、姿态、接触）是任务成功的关键组成部分；不是典型的持续运动控制任务，因为它以到达并停止在固定目标点为终点。

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`
- **confidence**: high
- **reason**: 任务的核心是到达指定的固定目标位置（目标平台）并稳定接触，附属有速度、姿态、能耗要求。这与“核心是到达指定目标位置”的导航‑目标到达族最为吻合，且后续动力学子类型进一步细化了“软接触”特性。

**动力学子类型 dynamics_subtype**: `goal_approach_and_soft_contact`

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: float32 （连续值，接触标志为 0.0/1.0 浮点）
- 各索引含义：
  - `obs[0]`: `x_position` —— 飞行器相对于目标平台中心的水平距离，reward_usable: true
  - `obs[1]`: `y_position` —— 飞行器相对于目标平台高度的垂直距离，reward_usable: true
  - `obs[2]`: `x_velocity` —— 水平线速度，reward_usable: true
  - `obs[3]`: `y_velocity` —— 垂直线速度，reward_usable: true
  - `obs[4]`: `body_angle` —— 飞行器躯干朝向与竖直方向的夹角，reward_usable: true
  - `obs[5]`: `angular_velocity` —— 角速度，reward_usable: true
  - `obs[6]`: `left_support_contact` —— 左支撑腿与平台或地面发生接触的标志（1.0 接触 / 0.0 未接触），reward_usable: true
  - `obs[7]`: `right_support_contact` —— 右支撑腿接触标志，reward_usable: true

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- 动作含义：
  - `action 0`：无引擎 —— 不施加任何推力，仅靠惯性运动，reward_usable: true（可用于燃料惩罚）
  - `action 1`：左姿态发动机 —— 向左上方喷射，产生逆时针旋转力矩，reward_usable: true
  - `action 2`：主发动机 —— 向正下方喷射，产生向上的推力，同时也是消耗燃料的主要动作，reward_usable: true
  - `action 3`：右姿态发动机 —— 向右上方喷射，产生顺时针旋转力矩，reward_usable: true

## 5. step 与终止条件分析
### 5.1 终止模式
终止条件组合：`crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled`
- **success‑like termination**：`body_not_awake_or_settled` 在飞行器双腿接触目标平台且速度、姿态足够稳定时，由物理引擎自动进入休眠/稳定状态触发。这很可能对应成功着陆。
- **failure‑like termination**：
  - `crash_or_body_contact`：飞行器主体与地面或平台以外物体发生了不安全的接触（如坠毁、侧翻等）。
  - `horizontal_position_outside_viewport`：飞行器水平飞出视口边界。
- **ambiguous termination**：`body_not_awake_or_settled` 也可能由其他原因（如翻倒后无动能）触发，无法从终止本身直接区分成功与失败。
- **truncation**：未提到 episode 长度限制。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**：从 masked step source 可知 `info` 为一个空字典 `{}`，因此奖励函数内部 **不可依赖 info**
- **forbidden_or_uncertain_info_fields**：`success`, `failure`, `termination_reason`, 任何与环境成功/失败有关的预定义标志均不存在或不可靠

## 6. reward 函数接口契约
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用**：
- `obs`：上一步观测（np.ndarray）
- `action`：刚执行的动作（int）
- `next_obs`：新观测（np.ndarray）
- `training_progress` 仅在 prompt 明确允许时方可使用（当前无此类说明，**禁用**）
- `info`：仅可使用任务说明中明确声明的字段（当前为空，故不可用）

**禁止使用**：
- `original_reward`（官方奖励已掩蔽，不得参照或泄露）
- 任何未在观测空间说明中声明的 obs 切片（环境可能返回额外噪声，但不应依赖）
- 任何未明确允许的 info 字段

## 7. 可用于奖励函数的信号
所有信号均从 `obs` 或 `next_obs`（以及 `action`）中获得。

- **position**：
  - `x_pos = next_obs[0]` （或 `obs[0]`）
  - `y_pos = next_obs[1]`
  - 可构造到目标点的**二维距离**：`dist = sqrt(x_pos² + y_pos²)`
- **velocity**：
  - `x_vel = next_obs[2]`
  - `y_vel = next_obs[3]`
  - 可用于速度惩罚、软着陆条件
- **orientation**：
  - `angle = next_obs[4]`
  - 可构造角度惩罚（目标接近 0）
- **angular velocity**：
  - `ang_vel = next_obs[5]`
  - 可辅助评估旋转稳定性
- **contact**：
  - `left_contact = next_obs[6]`
  - `right_contact = next_obs[7]`
  - 用于判断是否已接触平台/地面，作为稳定着陆的标志
- **action/engine**：
  - `action` 值可用于**燃料消耗惩罚**：动作 1/2/3 表示使用发动机，2（主发动机）消耗最大。

## 8. 不确定或不可用的信号
- **明确的成功/失败标志**：不存在，无法用于 sparse 奖励。
- **接触类型**：`left_contact`、`right_contact` 无法区分是与目标平台接触还是与地面/障碍物接触。因此仅靠接触标志不能完全保证“正确着陆”。
- **地形/地面高度**：观测不包含地面绝对位置，



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

