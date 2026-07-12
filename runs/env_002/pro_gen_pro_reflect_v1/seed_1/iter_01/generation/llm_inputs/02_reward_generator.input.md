# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 双足行走任务。智能体控制一个拥有两条腿（各含髋、膝两个关节）的身体，在不平坦地形上向前行走。核心目标是 **走得尽可能远、尽可能快**，同时 **最小化能量消耗**。要获得好成绩，必须通过协调四肢关节力矩生成稳定、高效的双足步态；一旦身体倾斜过度、摔倒，回合即告失败。任务**不存在明确的到达点**，过程终止于身体摔倒（失败）或到达地形末端（可视为成功完成路程）。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务明确要求持续前进通过地形，核心判断依据是推进距离与速度，而非导航到固定目标点；能耗作为次要优化项，不构成新的主任务类型。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float（具体为 float32/64，下同）
- 各分量含义及 reward_usable 标记：

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | hull_angle | 躯干相对于竖直方向的倾角（弧度） | true |
| 1 | hull_angular_velocity | 躯干的角速度 | true |
| 2 | horizontal_velocity | 前方的线速度（正值表示向前） | true |
| 3 | vertical_velocity | 上下方向线速度 | true |
| 4 | hip1_angle | 腿1髋关节角度 | true |
| 5 | hip1_speed | 腿1髋关节角速度 | true |
| 6 | knee1_angle | 腿1膝关节角度 | true |
| 7 | knee1_speed | 腿1膝关节角速度 | true |
| 8 | leg1_contact | 腿1触地标志（1.0 触地，0.0 离地） | true |
| 9 | hip2_angle | 腿2髋关节角度 | true |
|10 | hip2_speed | 腿2髋关节角速度 | true |
|11 | knee2_angle | 腿2膝关节角度 | true |
|12 | knee2_speed | 腿2膝关节角速度 | true |
|13 | leg2_contact | 腿2触地标志（1.0 触地，0.0 离地） | true |
|14 | lidar_0 | LIDAR 距离传感器 0（前方地形距离） | true（但需谨慎使用） |
|15 | lidar_1 | LIDAR 距离传感器 1 | true |
|16 | lidar_2 | LIDAR 距离传感器 2 | true |
|17 | lidar_3 | LIDAR 距离传感器 3 | true |
|18 | lidar_4 | LIDAR 距离传感器 4 | true |
|19 | lidar_5 | LIDAR 距离传感器 5 | true |
|20 | lidar_6 | LIDAR 距离传感器 6 | true |
|21 | lidar_7 | LIDAR 距离传感器 7 | true |
|22 | lidar_8 | LIDAR 距离传感器 8 | true |
|23 | lidar_9 | LIDAR 距离传感器 9 | true |

> 注：所有字段在理论上都可参与奖励计算。LIDAR 读数可用于感知前方地形起伏，但作为连续控制任务，直接奖励地形平整度可能引入噪声；除非明确需要地形自适应，否则奖励函数不一定要使用它们。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- dtype: float  
- 各维度含义：

| index | name | meaning |
|-------|------|---------|
| 0 | hip_torque_leg1 | 施加到腿1髋关节的力矩，范围 [-1, 1] |
| 1 | knee_torque_leg1 | 施加到腿1膝关节的力矩，范围 [-1, 1] |
| 2 | hip_torque_leg2 | 施加到腿2髋关节的力矩，范围 [-1, 1] |
| 3 | knee_torque_leg2 | 施加到腿2膝关节的力矩，范围 [-1, 1] |

动作是连续的关节力矩，智能体需生成协调的关节动作以形成稳定步态。

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**：`reached_end_of_terrain` — 身体抵达地形终点，说明已走完全程。由于无明确成功标志，该终止仅暗示路径被完整覆盖，必须谨慎使用，不能直接作为奖励。
- **failure-like termination**：`body_fallen_over` — 躯干倾斜过度导致摔倒，这是明确的失败。
- **ambiguous termination**：无其它模式。
- **truncation**：源码中仅返回 `terminated`，无 `truncated` 标记，因此默认所有终止都视为 episode 结束（可能由最大步数截断，但未指明，需假设环境可能包含基于步数的截断）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
  （无 `info` 输出，无法直接得知 `reached_end_of_terrain` 的具体值）
- explicit_failure_flag_available: **false**  
  （同样，`body_fallen_over` 没有作为独立标志提供给 reward 函数）
- allowed_info_fields: 无（`info` 为 `{}`）
- forbidden_or_uncertain_info_fields: 任何需要从 `info` 读取的字段均禁止

> 奖励函数**必须**基于观测序列隐式推断失败风险（例如通过 `hull_angle` 绝对值过大），**绝对不能**依赖 `done` 或任何终止标志。

## 6. reward 函数接口契约

函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

- **允许使用**：
  - `obs`（当前观测数组，shape (24,)）
  - `action`（当前动作数组，shape (4,)）
  - `next_obs`（下一时刻观测数组）
  - `training_progress`（浮点数，范围 [0,1]，仅在 prompt 明确允许时使用）
- **禁止使用**：
  - `original_reward`（被屏蔽的官方奖励）
  - 任何未声明的 `info` 字段（本环境中 `info` 为空，故禁止）
  - 未出现在上述列表中的额外全局变量

## 7. 可用于奖励函数的信号

- **姿态/平衡**：
  - `hull_angle`（obs[0]）：躯干偏离竖直的程度，越小越平衡。
  - `hull_angular_velocity`（obs[1]）：倾斜变化快慢，可用于惩罚剧烈晃动。
- **推进速度**：
  - `horizontal_velocity`（obs[2]）：正向速度，越大前进越快。
  - `vertical_velocity`（obs[3]）：可辅助检测颠簸或跳跃，但不一定是主要奖励源。
- **关节状态**：
  - 髋/膝角度与角速度（obs[4..7], obs[9..12]）：可用于限制关节极限、惩罚过大动作或鼓励平滑运动。
- **接触状态**：
  - `leg1_contact`, `leg2_contact`（obs[8], [13]）：反映脚是否着地，可用于检测步态交替或防止双腿同时离地。
- **动作/能量**：
  - `action` 四维力矩：直接反映控制能量，平方和或绝对值和可用来惩罚低效动作。
- **地形感知**：
  - LIDAR 传感器（obs[14..23]）：10个距离读数，描述前方地形轮廓，可用于奖励对地形的适应性（如避免过陡坡），但使用需谨慎，会大幅增加奖励设计复杂度。

## 8. 不确定或不可用的信号

- **明确的成功/失败标志**：不可用，`info` 为空。
- **地形结束信号**：无法在奖励计算时获取。
- **地形确切形状**：LID



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

