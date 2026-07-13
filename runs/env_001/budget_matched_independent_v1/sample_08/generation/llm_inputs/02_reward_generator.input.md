# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
主目标：控制一个 2D 飞行器从上方初始位置出发，尽快、稳定地降落在正下方的目标平台上。  
次目标：在保证安全着陆的前提下，尽量减少引擎推力（节省燃料）。  
不应混淆的目标：不要将稳定姿态与到达目标割裂；着陆是“到达 + 稳定减速 + 姿态小 + 安全接触”的复合目标，并非单纯到达某一坐标。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达中央目标垫并稳定停下，属于目标导向的导航任务；其他目标（省燃料、快速）为附属优化，不是多个同等重要的冲突目标，不构成 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float32
- obs[0]: x_position — 水平坐标（相对目标垫），reward_usable: true  
- obs[1]: y_position — 垂直坐标（相对垫面高度），reward_usable: true  
- obs[2]: x_velocity — 水平速度，reward_usable: true  
- obs[3]: y_velocity — 垂直速度，reward_usable: true  
- obs[4]: body_angle — 机体朝向角，reward_usable: true  
- obs[5]: angular_velocity — 角速度，reward_usable: true  
- obs[6]: left_support_contact — 左支撑触地标志（1/0），reward_usable: true  
- obs[7]: right_support_contact — 右支撑触地标志（1/0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine — 不点火，维持当前速度与姿态  
- action 1: left_orientation_engine — 点火左姿态引擎（产生旋转力矩）  
- action 2: main_engine — 点火主引擎（产生向上的推力）  
- action 3: right_orientation_engine — 点火右姿态引擎（产生相反方向的旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 可能对应成功稳定着陆（机体静止且不再活动），但该条件也可能由碰撞导致的“不活跃”触发，需结合观察判断。  
- failure-like termination: crash_or_body_contact（不安全接触）和 horizontal_position_outside_viewport（飞离边界）明确属于失败。  
- ambiguous termination: body_not_awake_or_settled 的成功/失败边界模糊，无法直接用作奖励依据。  
- truncation: 源码未出现，可能无 episode 步数限制（默认无）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: `{}` 无任何字段  
- forbidden_or_uncertain_info_fields: 任何 info 字段均不可使用，终止原因仅能从 done 标志与最后一帧观测推测。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 上一状态（8 维向量）
- action: 上一步采取的动作（0-3 整数）
- next_obs: 当前状态（8 维向量），可用于构建密集奖励与终止状态判断
- info: 当前为 `{}`，无任何可用字段
- training_progress: 即使 prompt 提及，当前任务未允许使用，应避免依赖

禁止使用：
- original_reward（被 mask，不得泄露、模拟或间接重建）
- official_reward / env 原始返回值
- 未声明的 info 字段（如 “success”、“failure”、“termination_reason” 等）
- 未声明的 obs 切片（仅可使用已公布的 8 维字段）

奖励函数不应内部调用 env 的任何方法，也不应通过 next_obs 推断 hidden 内部状态（如引擎推力数值）来重建 official reward。

## 7. 可用于奖励函数的信号
- position: next_obs 的 x, y 坐标
- velocity: next_obs 的 vx, vy
- orientation: next_obs 的 angle, angular_velocity
- contact: next_obs 的 left/right contact 标志
- action/engine: 当前 action 选择（可判别是否使用主引擎/姿态引擎），可从奖励函数参数 `action` 直接获取
- other: 无额外显式信号（如燃料量、距离阈值等）

## 8. 不确定或不可用的信号
- 精确的“着陆成功”标志：info 中不存在，只能从接触标志 + 速度/姿态阈值推断，且可能因碰撞误判
- 机体与目标垫的确切碰撞方向/强度：未给出
- 剩余燃料量或推力消耗：未在 obs 中直接体现，只能通过 action 间接推算使用频率
- 时间步数：未提供，truncation 未知，不能显式构建时间惩罚（除非使用 training_progress 近似，但这不应作为主要依赖）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid vehicle
  actuator_type: main thruster + two orientation thrusters
  contact_structure: two support legs (left/right) with binary contact
primary_objectives:
  - reach and settle on the central target pad
  - minimize crash / out-of-bound events
secondary_objectives:
  - use as little engine thrust as possible (fuel efficiency)
  - land quickly (implicit time pressure)
main_failure_risks:
  - crashing due to high vertical / horizontal speed at touchdown
  - tilting over (excessive angle) and contacting ground with body
  - drifting out of the horizontal viewport
  - hovering indefinitely without attempting to land (avoidance of progress)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: target_approach_proximity  
  purpose: 引导飞行器向目标垫（x≈0, y≈0）移动  
  why_required: 核心目标，没有该引导无法学习到达  
  usable_signals: [x_position, y_position] from next_obs  
  risks: 如果奖励单调递减至目标，可能忽略减速需求，导致撞击。需配合速度控制。

- role_id: velocity_norm_penalty  
  purpose: 惩罚过大的速度（尤其是接近地面时），迫使减速  
  why_required: 着陆必须低速度，否则 crash  
  usable_signals: [x_velocity, y_velocity] from next_obs  
  risks: 若不加位置条件，可能使 agent 不敢移动；需随高度缩减或与高度耦合。

- role_id: orientation_stability  
  purpose: 在着陆阶段鼓励小角度（接近直立）和低角速度  
  why_required: 安全接触需要两个支撑腿同时着地，大角度导致侧翻  
  usable_signals: [body_angle, angular_velocity] from next_obs  
  risks: 过度关注可能导致 agent 迟迟不下降；应在接近地面时加强。

- role_id: safe_contact_encouragement  
  purpose: 当左右支撑腿均接触地面时，额外奖励，若同时满足低速度与直立则给出最大奖励  
  why_required: 终点事件信号缺失，需要靠状态组合模拟成功着陆  
  usable_signals: [left_support_contact, right_support_contact, x_velocity, y_velocity, body_angle]  
  risks: 可能被 agent 利用：刻意在非目标位置制造接触但速度/角度很差仍获得部分奖励，需严格阈值。

### 10.2 条件职责 conditional_roles
- role_id: engine_usage_regularization  
  purpose: 微惩罚主引擎或姿态引擎的点火，促进节约推力  
  condition_to_use: 在已接近目标且速度较低时施加更强的惩罚（避免过早惩罚抑制探索），或在全局以极小的系数启用  
  usable_signals: [action]  
  risks: 若系数过高，agent 可能选择不动的策略（什么都不做就不耗油），导致永远不降落；需与 proximity reward 平衡。

- role_id: soft_time_pressure  
  purpose: 通过每步微小负奖励隐式鼓励快速完成，替代缺失的时间惩罚  
  condition_to_use: 仅当有 episode 步长上限或希望加速收敛时考虑；默认环境下 truncation 未知，此角色可选择性加入。  
  usable_signals: 无直接信号，需依赖 reward 函数外的步数计数（如使用 training_progress 间接加入）  
  risks: 可能导致 agent 牺牲安全性换取速度；若加入必须更重视安全性角色。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: boundary_safety_penalty  
  reason: 水平超出视口会直接终止 episode，惩罚意义不大；更有效的方法是依靠 proximity 把 agent 拉回中心。  
  forbidden_or_missing_signals: 边界位置阈值未明确给出；需要人为设定 hard 值，可能与环境真实边界不一致。

- role_id: crash_specific_penalty  
  reason: 环境终止时 crash 由 low-level 判定，但不提供显式标志。试图从终止前的观测推断 crash 并给予强负奖励容易误判（尤其是 body_not_awake 也可能为成功）。  
  forbidden_or_missing_signals: 无 `crash_flag` 或 `contact_force` 等信息。

- role_id: exact_landing_zone_incentive  
  reason: 目标只是中央目标垫，无多区域区分，不需要针对特定子区域给予分级奖励。简单 proximity 已足够。  
  forbidden_or_missing_signals: 无细粒度着陆区标识。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| target_approach_proximity | next_obs x_position, y_position | 无 | quadratic_penalty, bounded_signal (smooth distance) | 可使用平方向量距离或平滑平方距离（如 -sqrt(dx^2+dy^2) 或 -(dx^2+dy^2)） |
| velocity_norm_penalty | next_obs x_velocity, y_velocity | 无 | quadratic_penalty, conditional scaling (e.g. scale with height) | 结合高度衰减系数降低高空速度惩罚，避免抑制下降 |
| orientation_stability | next_obs body_angle, angular_velocity | 无 | quadratic_penalty, threshold_penalty | 当高度低于某值时强激活，或使用 angle * (1 - normalized_height) |
| safe_contact_encouragement | next_obs left/right contact, velocity, angle | 明确 success flag | conditional_bonus (if both contacts and low speed & angle) | 阈值需保守：contact==1 and abs(vx)<0.1, abs(vy)<0.2, abs(angle)<0.1 rad |
| engine_usage_regularization | action | 燃料消耗量 | constant_per_step (minimal), action_penalty | 仅惩罚 main engine (2), orientation engine (1,3) 可能因旋转需要不宜过度惩罚 |
| soft_time_pressure | (无直接信号) | episode steps | constant small negative per step (fixed), depends on training_progress logic | 仅在确实需要且不破坏安全目标时加入 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 不下降，悬停在高空 | y_position 不减小，velocity 接近 0，angle 稳定 | 加强 target_approach_proximity 权重，或放松高空速度惩罚；增加对接近目标的诱导 |
| 接近目标但未充分减速，撞击 | 接近 (0,0) 时速度仍较大，然后 episode 终止 | 强化 velocity_norm_penalty 在低高度的强度，或使用非线性更陡峭的惩罚 |
| 侧翻（单腿着地，角度大） | angle 较大，仅一侧 contact 为 1 | 加强 orientation_stability 尤其是低高度时的角度惩罚；提高 safe_contact 阈值 |
| 摇摆震荡



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

