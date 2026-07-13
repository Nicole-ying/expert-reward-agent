# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
主要目标：控制一个2D飞行器从顶部中心区域出发，最终平稳降落在画面中央的目标平台上。要求平台接触发生在两条支撑腿上、身体方向水平、速度趋近于零并且不发生 crash（如身体其他部位触碰地面或墙壁）。  
次要目标：在达成安全着陆的前提下，尽可能缩短飞行时间，并尽量减少引擎推力（节省燃料）。  
不应混淆的目标：任务的核心是精准软着陆，而非单纯穿越地形或保持平衡；快速与节能只是锦上添花，不能以牺牲安全降落为代价。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达指定的目标位置（平台中心）并稳定驻留，属于典型的终点导航/到达任务；附属的时间、燃料优化是次要约束，不构成权重相当的冲突目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测默认）
- obs[0]: x_position，含义：相对于目标平台中心的水平坐标，reward_usable: true
- obs[1]: y_position，含义：相对于目标平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，含义：水平线速度，reward_usable: true
- obs[3]: y_velocity，含义：垂直线速度，reward_usable: true
- obs[4]: body_angle，含义：身体朝向角度（可能以“水平为0”），reward_usable: true
- obs[5]: angular_velocity，含义：角速度，reward_usable: true
- obs[6]: left_support_contact，含义：左支撑腿是否接触平台（1.0接触，0.0未接触），reward_usable: true
- obs[7]: right_support_contact，含义：右支撑腿是否接触平台（1.0接触，0.0未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，含义：不点火，仅靠惯性运动
- action 1: left_orientation_engine，含义：启动左侧姿态发动机，产生某个方向的力矩/推力以调整朝向
- action 2: main_engine，含义：启动主引擎，提供向上的推力
- action 3: right_orientation_engine，含义：启动右侧姿态发动机，产生与左侧发动机相反的力矩/推力

（引擎的具体推力方向、大小未给出，但可以从名称推断：主引擎用于减速/悬浮，姿态引擎用于旋转身体。）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` —— 当身体静止/稳定时触发。极可能对应成功着陆并停稳的场景，因为在平台上稳定后不再移动，环境认为任务结束。
- **failure-like termination**:  
  - `crash_or_body_contact` —— 身体某些非支撑部位（如底部、侧边）发生碰撞，通常意味着坠毁。  
  - `horizontal_position_outside_viewport` —— 水平坐标超出可视范围，表示飞行器飞离并丢失。
- **ambiguous termination**: `body_not_awake_or_settled` 理论上也可能因为卡在某处不动而触发，但在该环境的典型设计中它大概率与成功绑定。
- **truncation**: 当前 step source 未展示时间截断，但实际环境可能存在 episode 长度限制（unknown）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {}，禁止使用任何 info 字段）
- forbidden_or_uncertain_info_fields: info 所有字段均不可用；original_reward 被屏蔽，禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`（当前观测）
- `action`（当前动作）
- `next_obs`（下一观测）
- `info` 目前为空，不得依赖任何内容
- `training_progress` 仅在 prompt 明确要求时使用，否则禁止

禁止使用：
- `original_reward`（官方奖励，已被屏蔽）
- 任何未声明或凭空出现的 `info` 字段
- 未在 observation_space 中声明的 obs 切片

## 7. 可用于奖励函数的信号
- **位置信号**：`obs[0]`（x偏移）、`obs[1]`（y偏移）及其变化量（可由 `next_obs - obs` 近似）
- **速度信号**：`obs[2]`（x速度）、`obs[3]`（y速度）
- **方向信号**：`obs[4]`（身体角度）、`obs[5]`（角速度）
- **接触信号**：`obs[6]`（左腿接触）、`obs[7]`（右腿接触）
- **动作/引擎信号**：`action`（0/1/2/3），可间接反映燃料消耗
- **其他**：可通过位置/速度组合计算距离、靠近速率等；时间代价可隐式用每步常数惩罚表示。

## 8. 不确定或不可用的信号
- **明确的 crash 标志**：无。仅能从 `obs` 中推测（如位置极度异常、速度突然反向等），但边界未知，初期不可靠。
- **精确的视野边界**：未提供值，无法预设阈值。
- **真实剩余燃料或推力大小**：观测空间中无此类信号，只能通过动作近似。
- **时间戳**：未提供步数或剩余时间，无法构建时间直接惩罚（但可间接用常数惩罚）。
- **身体其他部位碰撞信息**：除两条支撑腿外，无其它接触传感器，因此无法精确判断 crash_or_body_contact，只能靠经验间接推断。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D lander/flyer
  actuator_type: main engine (vertical thrust) + two lateral orientation engines
  contact_structure: two symmetric support legs for touchdown
primary_objectives:
  - Reach the central target pad (x≈0, y≈0)
  - Achieve soft touchdown with both legs contacting and low velocities
  - Keep body angle near zero (horizontal)
secondary_objectives:
  - Minimize flight time
  - Minimize engine usage (fuel)
main_failure_risks:
  - Crashing body into ground/walls
  - Exceeding horizontal boundaries
  - Touching down with non‑leg parts
  - Bouncing



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

