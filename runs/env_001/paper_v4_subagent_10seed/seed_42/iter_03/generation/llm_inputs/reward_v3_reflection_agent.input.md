# ⚠️ REBUILD MODE
系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。
参考 #6 完整公式算子库选新的主信号框架，基于 #3 累积记录避开已失败的路径。
不要受上一轮代码结构约束。


# 1. Search objective
- target_score: 200.000000
- current_score: -30.299078
- gap_to_target: 230.299078
- target_achievement_ratio: -15.150%

# 2. 上一轮奖励函数代码（该轮得分: -30.299078）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Convexified progress reward to incentivize faster approach.
    """
    # ---- Unpack observations ----
    px0, py0 = obs[0], obs[1]
    px1, py1 = next_obs[0], next_obs[1]
    vx1, vy1 = next_obs[2], next_obs[3]
    angle1  = next_obs[4]
    angvel1 = next_obs[5]
    left_leg  = next_obs[6]
    right_leg = next_obs[7]

    # ---- 1. Progress to target: convex combination of linear + quadratic ----
    dist_prev = (px0**2 + py0**2) ** 0.5
    dist_next = (px1**2 + py1**2) ** 0.5
    raw_progress = dist_prev - dist_next       # positive when approaching
    progress = max(0.0, raw_progress)           # only reward net progress
    progress_reward = progress + 2.0 * progress**2   # convex -> bigger steps get more

    # ---- 2. Orientation / stability soft constraints (unchanged) ----
    angle_penalty  = -0.01 * (angle1 ** 2)
    angvel_penalty = -0.005 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ---- 3. Soft landing guidance (unchanged) ----
    speed1 = (vx1**2 + vy1**2) ** 0.5
    proximity_threshold = 0.2
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor = 1.0 / (1.0 + 10.0 * speed1)
        soft_landing = contact_factor * speed_factor
    else:
        soft_landing = 0.0

    # ---- Combine components ----
    total_reward = (
        1.0 * progress_reward
        + 1.0 * orientation_penalty
        + 1.0 * soft_landing
    )

    components = {
        "progress_delta": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_landing": soft_landing
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 436.40 | 163.33 | ✅ |
| 2 | 二次项使大步前进的奖励显著更高，agent 将更早到达 proximity 区域并仍由 `soft_landing... | 二次项使大步前进的奖励显著更高，agent 将更早到达 proximity 区域并仍由 `soft_landing... | 95.85 | -30.30 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-30.299078, len=95.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-121.924873, 60.681461]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_delta | 1.283470 | 68.9% | 68.9% | 93.9% |
| soft_landing | 0.495858 | 26.6% | 26.6% | 3.4% |
| orientation_penalty | -0.082683 | -4.4% | 4.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 9/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Eval score -30.3, all episodes terminate early (len 95.85). Shaped reward mean 0.028 vs original env -1.27 (misaligned). Progress_delta dominates (69% share, 94% active); soft_landing sparse (3.4% active) but contributes 27%.

**Component Anomalies**: soft_landing: very low active rate yet high share (27%), indicating large spikes. orientation_penalty: always active but negligible magnitude. progress_delta: dominant positive driver.

**Training Dynamics**: No temporal snapshots; cannot assess trends.

**Signal Quality**: No dead components. soft_landing unreliable due to sparsity. Shaped reward not aligned with true objective, likely encouraging fast progress at cost of early crashes. No attractor for stable landing.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个 2D 飞行器/车辆类似的任务：智能体从视口顶部中心附近受初始随机力开始，需要尽可能快地飞抵并稳定停靠在中央目标平台上，同时尽量减少发动机推力使用。核心目标是 **快速、稳定地完成着陆（到达并停留）**，次要目标是 **节省燃料、保持姿态平稳**。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推断)
- obs[0]: `x_position` – 相对于目标平台中心的水平坐标，越小越接近，reward_usable: true
- obs[1]: `y_position` – 相对于目标平台高度的垂直坐标，reward_usable: true
- obs[2]: `x_velocity` – 水平线速度，reward_usable: true
- obs[3]: `y_velocity` – 垂直线速度，reward_usable: true
- obs[4]: `body_angle` – 身体倾斜角度（假设水平为 0），reward_usable: true
- obs[5]: `angular_velocity` – 角速度，reward_usable: true
- obs[6]: `left_support_contact` – 左支撑腿是否接触（0 或 1），reward_usable: true
- obs[7]: `right_support_contact` – 右支撑腿是否接触（0 或 1），reward_usable: true

（注意：所有字段均可用，但需小心接触信号的语义，任务目标中“接触”指的是安全着陆在目标平台，而非与地面或障碍物的碰撞）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` – 不点火，依靠当前动量漂移
- action 1: `left_orientation_engine` – 点燃左姿态发动机（调整姿态或水平推力）
- action 2: `main_engine` – 点燃主发动机（主要提供垂直或前进推力）
- action 3: `right_orientation_engine` – 点燃右姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
根据掩码源码，存在三种终止触发：
- `crash_or_body_contact` – 坠毁或部分身体接触（可能包括与地面/障碍物的不当接触）
- `horizontal_position_outside_viewport` – 水平位置超出视口边界（失败）
- `body_not_awake_or_settled` – 身体不再活跃或已经稳定（可能为成功，若发生在目标平台上）

成功意义上的终止并没有显式分离，只能通过观测状态间接判别：当智能体接近目标( x ≈ 0, y ≈ 0 )，速度极小，且两侧支撑腿接触（可能），触发 `body_not_awake_or_settled` 可视为 soft landing success；而其他终止条件（crash、出界）则对应失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空字典，无额外字段）
- forbidden_or_uncertain_info_fields: 所有未在 `observation_space` 中列出的字段均不可用（包括 `terminated` 标记、`success` 等）

## 7. 可用于奖励函数的信号
- **位置**：`x_position`，`y_position`（相对目标，直接表征进度）
- **速度**：`x_velocity`，`y_velocity`（绝对值或矢量和可用于判断稳定、能耗）
- **姿态**：`body_angle`，`angular_velocity`（衡量晃动，违反稳定着陆）
- **接触**：`left_support_contact`，`right_support_contact`（区分接触/非接触，可用于软着陆推断）
- **动作/引擎**：动作类别 0-3，可用于燃油消耗惩罚（action != 0 视为使用引擎）
- **其他衍生信号**：
  - 距离目标：`dist = sqrt(x^2 + y^2)`（可直接计算）
  - 速度大小：`speed = sqrt(vx^2 + vy^2)`
  - 距离减少：`delta_dist = dist_prev - dist_next`
  - 姿态偏离：`angle_deviation`（假设水平为0）

# 7. Formula Operator Library（完整版，用于 Level 3 重建）
# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

---

## 1. Expert Schema 使用规则

- environment_card.md 中的任务画像和可用信号优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role（任务需要什么类型的奖励信号），再选 signal（哪个观测维度承载这个 role），再选 formula operator（用什么数学形式表达），最后写代码。
- 如果某个 role 需要的信号在观测空间中不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- reward_v1 以主学习信号和必要的稳定/安全约束为重点。效率、能耗、复杂门控和动态权重可以在后续迭代中按需加入，但不应因"模板没列"而排除合理的设计。

---

## 2. 信号完备性自查清单

在完成初始设计后，逐一检查以下信号类型是否被覆盖——不是每个任务都需要全部，但每一项的缺失应是有意选择：

- **主进展信号**：agent 朝任务目标前进时是否获得正向反馈？该信号是否每步都有梯度？
- **灾难性失败信号**：是否存在明确的终止惩罚（如摔倒、飞出边界）？如果观测中可推断失败状态，是否给予了足够强的负向信号？
- **效率/代价信号**：连续动作空间中是否有能量消耗或控制代价约束？离散动作空间中是否有不必要的动作惩罚？
- **任务完成信号**：终止条件中是否包含 success-like 条件？相应的观测是否可被用来构造任务完成的软近似信号？
- **健康/稳定约束**：agent 是否因缺少姿态/速度/位置约束而产生不安全行为？

---

## 3. Formula Operator Library

每个算子包含：数学形式、使用条件、适用证据。

### 3.1 dense_state_signal
数学形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2`
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
使用条件：该状态信号每步可观测，且与某项任务职责直接相关。
适用证据：
  - 凸化 → episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小（agent 满足于低水平稳态）。
  - hinge → 约束组件的 active_rate≈100%（全时惩罚）但 terminated 率仍高，说明 agent 在安全范围内也被持续惩罚，需要只在越界时生效的 hinge。
风险：线性正奖励在信号平台期无梯度；凸化权重过大可能诱导极端行为；hinge 的 threshold 需根据环境卡片的观测范围设定。

### 3.2 improvement_delta
数学形式：`old_measure - new_measure`（期望减少时）或 `next_value - current_value`（期望增加时）
使用条件：obs 和 next_obs 中存在可比较的标量度量，该度量沿最优路径应单调变化。
适用证据：有明确的进展度量（位置、距离、高度、角度等），且该度量的变化比瞬时速率更能反映真实进展。
与 dense_state_signal 的选择：如果要鼓励"处于某种好状态"，用 `w * signal`。如果要鼓励"朝好方向改变"，用 delta。delta 的优势是 agent 无法在好状态上停滞不前，必须持续改善。适合：agent 当前的绝对状态值不能完全反映进展（如位置——站在原点不动 vs. 走到终点但位置绝对值可能相同）。
注意：对观测中直接给出的速度信号（如 `horizontal_velocity`）不要做 delta——速度本身已经是变化率。对观测中的位置/角度/距离类信号优先考虑 delta。

### 3.3 potential_based_shaping
数学形式：`potential(next_obs) - potential(obs)`
使用条件：(1) 任务有一个可量化的进展度量（如位置、距离、高度）；(2) 该度量沿最优路径应单调变化；(3) 能从观测中构造一个标量的 potential function。
如何构造 potential：从观测中选择一个在任务完成时达到极值、且沿最优路径单调变化的信号（或信号组合）。potential 的计算只能依赖观测，不能依赖环境内部状态。
与 improvement_delta 的关系：两者数学上等价。potential_based_shaping 的优势在于允许将多个信号编码到一个 potential 中（如同时考虑位置和姿态），而 improvement_delta 通常用于单个度量。
风险：potential 若与任务目标不一致会系统性地误导策略。reward_v1 中如果存在天然的进展度量，优先使用 improvement_delta 的简单形式；当需要组合多个信号构造进展度量时，使用 potential_based_shaping。

### 3.4 quadratic_penalty
数学形式：`-w * error**2` 或 `-w * sum(action_i**2)`
使用条件：约束信号连续可观测，惩罚不应压制主学习信号。用于轻量抑制——需要约束但不至于触发终止的行为。
适用证据：某维度出现高频大幅波动或极端值但未触发终止。
与 hinge 的选择：如果约束有明确的安全边界（如身体倾角超过 X 度必摔），用 hinge（3.1）。如果只是希望"越小越好"没有硬边界（如控制代价、小幅抖动），用 quadratic。
风险：权重过大导致 agent 不敢行动。

### 3.5 soft_health_gate
数学形式：`main_reward * gate_factor`，gate_factor ∈ [0, 1] 在身体状态恶化时平滑衰减。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (safe_bound - current) / margin))`
使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
适用证据：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍 >0——agent 在"先冲后死"，需要在健康恶化时切断主奖励而非额外加罚。
风险：gate 太严格抑制探索；衰减区间应设在"接近危险但尚未终止"的范围内。

### 3.6 terminal_event
数学形式：`if failure_condition: reward = -PENALTY`（硬覆盖 per-step 奖励），或 `if success_condition: reward = +BONUS`
使用条件：(1) 存在可从观测推断的灾难性失败状态（如身体倾角超过阈值 + 接触地面）或任务完成状态；(2) 环境 info 为空因此无法直接读取终止原因。
如何构造：不要依赖 info 字段判断终止原因。可从观测推断：摔倒 → hull_angle 突然偏转 + 身体位置急剧下降；到达终点 → 持续前进中 episode 突然终止（truncated）；出界 → 位置坐标超出有效范围。
适用证据：agent 频繁触发某种终止模式，但当前奖励没有针对该模式提供差异化信号——比如所有终止回合 reward 都一样，agent 无法区分成功和失败。
与 hinge/gate 的区别：hinge 在越界前提供连续梯度，gate 在恶化时衰减主信号。terminal_event 在事件发生的那一刻提供硬信号——没有梯度，但语义明确（"这就是你应该避免/追求的结果"）。

### 3.7 action_efficiency
数学形式：`-w * sum(|action_i|)` 或 `-w * sum(action_i**2)`
使用条件：动作空间 ≥ 2 维连续控制，且任务包含隐含的效率需求（如 locomotion、manipulation）。
适用证据：agent 学会完成任务但动作幅度异常大、能耗高——说明缺效率约束。通常系数较小（主信号 per-step 的 1-5%），避免压制探索。
注意：离散动作空间通常不需要此算子，因为离散动作的选择隐含了代价。首次迭代可不加入，后续迭代若观察到无效动作频繁出现再考虑。

### 3.8 joint_condition_proxy
数学形式：`factor_1 * factor_2 * ...`（每个 factor 为连续 bounded 形式）或 `(f1 + f2 + ...) / n` 或 `(f1 * f2 * ...) ** (1/n)`
使用条件：没有显式 success flag，但有连续信号可构造任务完成的软近似。
适用证据：agent 能在各子条件分别取得进展但无法同时满足。
风险：乘积塌缩（一个 factor→0 则整体→0）；用几何平均或算术平均可缓解。

### 3.9 bounded_signal
数学形式：`x / (1 + abs(x))` 或 `1 / (1 + k * abs(error))` 或 `max(0, 1 - abs(error) / threshold)`
使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。用于压缩极端值而非施加约束。
与 hinge 的区别：bounded 是从两端压缩信号范围，hinge 是只在超出阈值时施加惩罚。如果目标是"值不应超过 X"，用 hinge；如果目标是"值不应该爆炸但无所谓具体范围"，用 bounded。

### 3.10 preview_conditioned_reward
数学形式：`main_reward * preview_factor`，preview_factor 基于观测中能反映**未来状态**的信号（如距离传感器、高度采样、前方地形探测），在不利前景下从 1 平滑衰减到下限。
使用条件：(1) 观测中存在提供前方/未来信息的维度；(2) 该维度可以映射到"前景好/坏"的连续度量；(3) agent 的失败模式与"无法提前调整行为以应对即将到来的状态变化"相关。
如何构造：从提供未来信息的观测中选择一个标量信号，设计一个在安全前景下接近 1、危险前景下接近下限（如 0.3-0.5）的衰减函数。下限不为零以避免完全抑制探索。
适用证据：agent 在相似的瞬时状态下表现差异大（同样的速度/姿态，有时成功有时失败），说明当前状态本身不足以区分好坏——缺少关于"接下来会发生什么"的信息。
与 soft_health_gate 的区别：gate 用当前的**身体状态**乘主奖励（"我已经歪了，别冲了"——被动响应）。preview 用**未来信息**乘主奖励（"前面是坑，别冲了"——主动预判）。两者可以共存：`main_reward * health_gate * preview_factor`。
风险：preview 信号若有噪声会导致主奖励波动；衰减下限设太低会抑制必要探索。

---

## 4. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"定位到合适的算子变换。
以数学语义和训练表现证据为准，不要求组件名完全匹配。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |



# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | orientation_penalty + progress_delta + soft_landing | 163.33 | 163.33 | 0.00 | 436.40 | orientation_penalty=-0.001 progress_delta=0.011 soft_landing=0.118 | new_best |
| 2 | orientation_penalty + progress_delta + soft_landing | -30.30 | 163.33 | -193.63 | 95.85 | orientation_penalty=-0.001 progress_delta=0.016 soft_landing=0.014 | no_meaningful_improvement |
