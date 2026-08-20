# ⚠️ REBUILD MODE
系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。
参考 #6 完整公式算子库选新的主信号框架，基于 #3 累积记录避开已失败的路径。
不要受上一轮代码结构约束。


# 1. Search objective
- target_score: 200.000000
- current_score: -124.390904
- gap_to_target: 324.390904
- target_achievement_ratio: -62.195%

# 2. 上一轮奖励函数代码（该轮得分: -124.390904）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nl_contact, nr_contact = next_obs

    # compute distances to target (0,0)
    dist = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5

    # 1. main progress signal: improvement_delta on distance
    delta_dist = dist - dist_next  # positive when approaching target
    delta_dist_clipped = max(-0.5, min(delta_dist, 0.5))  # bound extreme jumps
    progress_reward = 1.0 * delta_dist_clipped

    # 2. speed penalty when near the target to encourage gentle approach
    close_threshold = 0.3
    if dist_next < close_threshold:
        speed_penalty = -0.5 * (abs(nvx) + abs(nvy))
    else:
        speed_penalty = 0.0

    # 3. attitude stability: hinge penalty on body angle
    safe_angle = 0.2  # radians
    angle_excess = max(0.0, abs(nangle) - safe_angle)
    angle_penalty = -0.1 * angle_excess

    # 4. fuel efficiency: small penalty for main engine usage
    fuel_cost = 0.0
    if action == 2:  # main engine
        fuel_cost = -0.02

    # 5. soft landing bonus: proxy success condition using observable signals
    success_dist_thresh = 0.1
    success_speed_thresh = 0.2
    success_angle_thresh = 0.1
    soft_landing_bonus = 0.0
    if (dist_next < success_dist_thresh and
        abs(nvx) < success_speed_thresh and
        abs(nvy) < success_speed_thresh and
        abs(nangle) < success_angle_thresh and
        nl_contact == 1 and nr_contact == 1):
        soft_landing_bonus = 10.0

    total_reward = progress_reward + speed_penalty + angle_penalty + fuel_cost + soft_landing_bonus
    components = {
        "progress_reward": progress_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "fuel_cost": fuel_cost,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.30 | -117.88 | ✅ |
| 2 | 骨架变化: action_cost + angle_hinge + danger_penalty + progr | — | 68.35 | -117.48 | ✅ |
| 3 | 骨架变化: action_cost + angle_hinge + landing_contact_reward | — | 68.30 | -122.17 | ❌ |
| 4 | 骨架变化: action_cost + landing_contact_reward + landing_spe | — | 143.70 | -87.19 | ✅ |
| 5 | 骨架变化: action_cost + landing_contact_reward + progress_sh | — | 143.70 | -87.19 | ❌ |
| 6 | 骨架变化: action_cost + angle_hinge_penalty + landing_contac | — | 68.35 | -114.35 | ❌ |
| 7 | 骨架变化: action_cost + angle_hinge_penalty + landing_contac | — | 71.20 | -105.53 | ❓ |
| 8 | 骨架变化: angle_penalty + fuel_cost + progress_reward + soft | — | 84.45 | -124.39 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-124.390904, len=84.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-166.363526, -96.224772]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 1.029374 | 53.5% | 56.4% | 100.0% |
| speed_penalty | -0.619512 | -32.2% | 32.2% | 0.8% |
| fuel_cost | -0.218000 | -11.3% | 11.3% | 12.9% |
| angle_penalty | -0.001613 | -0.1% | 0.1% | 0.2% |
| soft_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: mean_eval_reward=-124.4, all 20 episodes early terminal (length=84.5, score<-50). Shaped total ~0.0; progress_reward +1.03/sum but offset by penalties.

**Component Anomalies**: soft_landing_bonus dead (0% active, 0.0 sum). speed_penalty rarely active (0.8%) but contributes -32.2% signed share, large magnitude when triggered near target.

**Training Dynamics**: no temporal snapshots; final checkpoint only.

**Signal Quality**: soft_landing_bonus never crosses thresholds; speed_penalty threshold rarely met (dist<0.3). Total shaped reward ~0.0, no success attractor; all episodes crash early.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器/着陆器从视口上方出发，尽快降落到画面中央的水平目标垫上并稳定停靠。主目标是精确到达并停稳在目标垫中心（位置误差趋于零，速度接近零，两支撑脚着垫）。次要目标是尽量减少引擎使用（节能），快速完成任务。注意不要与此类任务可能混淆的纯飞行姿态控制、单纯前进速度优化或仅存活不要求停稳的任务混淆。

## 3. 观察空间 observation_space
- **type:** Box  
- **shape:** (8,)  
- **dtype:** 通常为 float64（环境默认），可视为连续浮点数。  

各索引含义：  
- `obs[0]`：`x_position`，飞行器相对目标垫中心的水平距离（向右为正），reward_usable: true  
- `obs[1]`：`y_position`，飞行器相对目标垫高度的垂直距离（向上为正，0 表示与垫面等高），reward_usable: true  
- `obs[2]`：`x_velocity`，水平线速度，reward_usable: true  
- `obs[3]`：`y_velocity`，垂直线速度，reward_usable: true  
- `obs[4]`：`body_angle`，机身倾角（弧度，0 为水平），reward_usable: true  
- `obs[5]`：`angular_velocity`，角速度，reward_usable: true  
- `obs[6]`：`left_support_contact`，左侧支撑脚触地标志（0.0 或 1.0），reward_usable: true  
- `obs[7]`：`right_support_contact`，右侧支撑脚触地标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- **type:** Discrete  
- **n:** 4  
- **动作说明：**  
  - `action 0`：“no_engine” — 所有引擎关闭，无推力。  
  - `action 1`：“left_orientation_engine” — 点燃左侧姿态引擎，产生偏航/旋转力矩。  
  - `action 2`：“main_engine” — 点燃主引擎，产生主体推力（通常向上或沿机身轴线）。  
  - `action 3`：“right_orientation_engine” — 点燃右侧姿态引擎，产生反方向旋转力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled`，三种触发情景：
- **crash_or_body_contact**：飞行器主体（非支撑脚）与地面或环境障碍碰撞，通常表示失败。  
- **horizontal_position_outside_viewport**：飞行器水平超出视口范围，失败。  
- **body_not_awake_or_settled**：物理体进入休眠状态或因稳定停靠而“settled”。根据任务目标，在目标垫上稳定停靠后应触发此条件，属于成功结果；但也可能因坠毁后体僵硬休眠触发，因此需要结合其他观测才能确定是成功还是失败。  

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** false  
- **explicit_failure_flag_available:** false  
- **allowed_info_fields:** `info` 当前为空字典 `{}`，无法直接获得任何结果标志。  
- **forbidden_or_uncertain_info_fields:** 任何未声明的字段（如 `success`、`failure`、`termination_reason` 等）均不可信。  

成功/失败只能通过 **derived_possible** 方式从观测序列中推断：  
- 成功终端（目标垫稳定停靠）：`episode` 结束时，`x_position`≈0, `y_position`≈0, `|x_velocity|` 和 `|y_velocity|` 很小，`left_support_contact`==1, `right_support_contact`==1，且未发生 `horizontal_out` 现象。  
- 坠毁终端：`episode` 结束时，倾角 `|body_angle|` 很大，或 `y_position` 异常低（地面以下），或只有一只脚接触物且位置远离目标垫。  
- 出界终端：`episode` 结束时，`x_position` 绝对值超出合理范围（范围需通过环境运行中观测到的边界估计，如 |x| > 1.5，或从 rollouts 中统计）。

## 7. 可用于奖励函数的信号
- **位置相关：**  
  - `x_position`, `y_position`（可直接计算到目标垫中心的欧氏距离 `dist = sqrt(x² + y²)`）  
  - 可衍生：`dist_to_target`，上一时刻距离与当前距离之差（delta progress）：`progress = dist(obs) - dist(next_obs)`，正值表示靠近。  
- **速度相关：**  
  - `x_velocity`, `y_velocity` 可用于惩罚接近时的剩余动能，或构建稳定条件。  
- **姿态相关：**  
  - `body_angle` 可用于 hinge penalty（防止倾斜过大）；`angular_velocity` 用于抑制快速旋转。  
- **接触信号：**  
  - `left_support_contact`, `right_support_contact` 可判断双脚是否着垫，是成功停靠的必要条件。  
- **动作相关：**  
  - `action` 值可用于计算动作成本（action ≠ 0 时轻微惩罚）。  
- **衍生信号（derived_possible，需与环境边界参数拟合）：**  
  - **终端成功事件：** 当 `terminated` 且 `dist_to_target` 小于阈值 (如 0.1)，速度幅值低于阈值，且 `left_support_contact` 和 `right_support_contact` 均为 1。  
  - **坠毁事件：** 当 `terminated` 且不满足成功条件，同时 `|body_angle|` 过大或 `y_position` 偏离过大。  
  - **出界事件：** 当 `terminated` 且 `x_position` 超出可靠运行范围。

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
| 1 | action_cost + angle_hinge + progress_shaping | -117.88 | -117.88 | 0.00 | 68.30 | action_cost=-0.001 angle_hinge=-0.001 progress_shaping=0.015 | new_best |
| 2 | action_cost + angle_hinge + danger_penalty + progress_shaping | -117.48 | -117.48 | 0.00 | 68.35 | action_cost=-0.002 angle_hinge=-0.001 danger_penalty=-0.002 progress_shaping=0.015 | new_best |
| 3 | action_cost + angle_hinge + landing_contact_reward + progress_shaping | -122.17 | -117.48 | -4.69 | 68.30 | action_cost=-0.001 angle_hinge=-0.001 landing_contact_reward=0.003 progress_shaping=0.015 | no_meaningful_improvement |
| 4 | action_cost + landing_contact_reward + landing_speed_gate + progress_shaping + shaped_progress | -87.19 | -87.19 | 0.00 | 143.70 | action_cost=-0.002 landing_contact_reward=0.007 landing_speed_gate=0.879 progress_shaping=0.014 shaped_progress=0.011 | new_best |
| 5 | action_cost + landing_contact_reward + progress_shaping + shaped_progress | -87.19 | -87.19 | 0.00 | 143.70 | action_cost=-0.002 landing_contact_reward=0.007 progress_shaping=0.014 shaped_progress=0.011 | no_meaningful_improvement |
| 6 | action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress | -114.35 | -87.19 | -27.16 | 68.35 | action_cost=-0.001 angle_hinge_penalty=-0.000 landing_contact_reward=0.003 progress_shaping=0.015 shaped_progress=0.012 | no_meaningful_improvement |
| 7 | action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress | -105.53 | -87.19 | -18.34 | 71.20 | action_cost=-0.001 angle_hinge_penalty=-0.000 landing_contact_reward=0.003 progress_shaping=0.008 shaped_progress=0.007 | unsolved_stagnation_fresh_restart |
| 8 | angle_penalty + fuel_cost + progress_reward + soft_landing_bonus + speed_penalty | -124.39 | -87.19 | -37.20 | 84.45 | angle_penalty=-0.002 fuel_cost=-0.005 progress_reward=0.009 soft_landing_bonus=0.004 speed_penalty=-0.015 | no_meaningful_improvement |
