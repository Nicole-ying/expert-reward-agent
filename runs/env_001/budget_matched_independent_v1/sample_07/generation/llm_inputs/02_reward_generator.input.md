# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 飞行器轨迹优化问题。一个刚体从视口顶部中心附近出发，带有初始随机扰动。
核心目标是**到达并稳定停留在中心目标平台上（settle）**，要求飞行器最终停在目标垫上，
速度趋于零，姿态稳定，并且两个接触标志均为有效接触。
次要目标是最小化到达时间和发动机推力使用量，即“尽快、尽量省燃料”。
任务**不能混淆**为单纯推进（locomotion）或仅保持平衡，着陆成功是最终判定标准。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 核心目标是到达指定目标位置（目标垫）并稳定停驻，附属有时间/能效等优化但不改变主要目的。该任务不属于持续行走、操作、多目标冲突等类别。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（连续量 + 接触标志 0.0/1.0）  
- 各分量含义与可奖励性：  

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | x_position | 飞行器质心相对于目标垫中心的水平偏移量 | true |
| 1 | y_position | 飞行器质心相对于目标垫高度的垂直偏移量 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 身体倾斜角 | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑接触标志（1.0 有接触，0.0 无） | true |
| 7 | right_support_contact | 右支撑接触标志（1.0 有接触，0.0 无） | true |

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：

| action | name | meaning |
|--------|------|---------|
| 0 | no_engine | 不启动任何引擎，仅依靠惯性、重力、风等 |
| 1 | left_orientation_engine | 启动左侧姿态引擎，提供偏航/扭矩 |
| 2 | main_engine | 启动主引擎，提供主要推力（通常向下或向后） |
| 3 | right_orientation_engine | 启动右侧姿态引擎，与左引擎反向，调节姿态 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` — 身体已经静止或安定。若此时飞行器位于目标垫上方（x、y 接近 0）、速度极低、双接触点有效，且姿态水平，则很可能成功着陆。该模式是唯一可能代表成功的终止，但无法直接确认。
- **failure-like termination**:  
  - `crash_or_body_contact` — 身体与地面或非目标区域发生碰撞（例如机腹触地、侧翻等）。  
  - `horizontal_position_outside_viewport` — 水平位置超出可视范围，飞行器脱离可控制区域。
- **ambiguous termination**:  
  上述 `body_not_awake_or_settled` 也可在不安全状态下触发（如悬停在空中但静止，或只有一个接触点触地），因此为模糊终止。
- **truncation**:  
  源步骤中 `return` 的第四个参数为 `False`，表示无截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **false**  
- allowed_info_fields: 无（`info` 为 `{}`）  
- forbidden_or_uncertain_info_fields: 任何依赖 `info` 的 `success`、`failure`、`termination_reason` 等字段均不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（当前观测）
- `action`（离散动作索引）
- `next_obs`（下一步观测）
- `info` 中明确允许的字段（这里 `info` 为空，**无可用的额外信息**）
- `training_progress` 仅在 prompt 明确允许时使用（当前未开放，不建议依赖）

禁止使用：
- `original_reward`（官方奖励已掩膜）
- 任何未明确声明的 `obs` 切片含义外的原始值
- 假定的 `info` 字段（如 `success`、`landing_quality` 等）

## 7. 可用于奖励函数的信号
- 位置：`obs[0]` (x), `obs[1]` (y)，均为相对目标垫中心的偏移。  
- 速度：`obs[2]` (vx), `obs[3]` (vy)。  
- 姿态：`obs[4]` (angle), `obs[5]` (angular_vel)。  
- 接触：`obs[6]` (left contact), `obs[7]` (right contact)。  
- 动作：`action`（0~3 离散值，可判断是否用引擎）。  
- 可在 `next_obs` 中获取以上信号的变化，例如速度变化、位置变化等。

## 8. 不确定或不可用的信号
- 官方奖励：完全掩膜，不可获得。  
- 着陆成功 / 失败标识：无。  
- 剩余燃料 / 时间步计数：无法从当前接口获取。  
- 目标垫的绝对坐标或尺寸：未知，但 obs 已提供相对位置，足够。  
- `info` 中的任何字段：均为空，不可用。  
- `training_progress` 的具体含义与范围未知，当前不应依赖。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: single_rigid_body
  actuator_type:
    - main_engine (垂直 / 倾斜推力)
    - two_orientation_engines (左右旋转控制)
  contact_structure:
    - two_support_contacts (left, right)
primary_objectives:
  - reach_and_settle_on_target_pad (到达目标垫并稳定停驻)
secondary_objectives:
  - minimize_time_to_settle (最小化着陆时间)
  - minimize_engine_usage (最小化引擎使用)
main_failure_risks:
  - crash_into_ground (与地面/非目标碰撞)
  - out_of_horizontal_bound (水平漂移出界)
  - unstable_or_incomplete_landing (着陆时翻滚、单脚着地、速度未消除等)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
这些职责直接服务于核心目标，必须在所有训练阶段存在或占据主导。

- **role_id**: `approach_to_target`  
  purpose: 引导飞行器朝目标垫移动，使相对位置 (x, y) 趋近于 (0,0)。  
  why_required: 不朝目标移动则永远无法抵达。  
  usable_signals: [obs[0] x_position, obs[1] y_position, next_obs[0], next_obs[1]]  
  risks: 过分简单的负距离奖励可能导致震颤；需要使用平滑、有界的信号。

- **role_id**: `soft_landing`  
  purpose: 在接近目标时促使



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

