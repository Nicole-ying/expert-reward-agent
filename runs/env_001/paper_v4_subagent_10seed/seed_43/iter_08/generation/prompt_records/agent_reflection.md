# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。正常模式下每次做一个可验证的修改。重建模式（用户 prompt 明确标注 REBUILD MODE）下可以更换主信号框架。

# 你收到的数据（按顺序）

1. **Search objective** — 目标分数、当前分数、差距。
2. **上一轮奖励函数代码** — 刚被训练过的 reward 源码。
3. **累积迭代记录** — 每轮"做了什么→预期什么→实际发生什么"的因果链表。预判列连续 ❌ 意味着当前方向大概率错误。
4. **训练反馈** — Final-policy outcome（score, len, terminated/truncated）、组件表格（episode_sum_mean 是每回合有符号累计量，active_rate 是非零触发率）。
5. **环境事实** — 任务目标（§1）、观测空间（§3）、动作空间（§4）、终止条件（§5）。声明的 obs/action 维度是唯一可用接口。
6. **Formula Operator Library** — 正常模式给算子切换表；重建模式给完整公式算子库（§2.1-2.8），用于选全新骨架。
7. **历史记忆** — 迭代历史表（iter, skeleton, score, len, decision）。

# 决策流程

## 0. 信号覆盖审计（先于诊断，必须逐项完成）

**在诊断现有组件之前，首先判断失败是因为信号缺失还是信号校准问题。** 这个区分决定后续所有方向。

### 0.1 终止模式分析

从 #4 的 terminated/truncated 数量和 episode length 分布，推断 agent 主要以什么方式结束 episode：
- 如果大部分 episode 是 truncated（超时）→ agent 存活但未完成任务目标
- 如果大部分 episode 是 terminated 且长度短 → agent 触发了某种终止条件
- 如果 terminated 的 episode 中有长有短 → 可能存在多种终止原因

结合 #5 §5 声明的终止条件列表，推断当前 episode 的终止主要是哪种条件触发的，以及是否有证据表明 agent 已经接近任务完成。

### 0.2 观测使用扫描

逐项检查 #5 中声明的观测维度在 #2 代码中的使用情况：
- 哪些观测维度被使用了？（列出索引和含义）
- 哪些观测维度未被使用？（列出索引和含义）
- 未使用的观测中，是否有维度能提供关于"agent 为什么会以当前模式终止"的信息？
- 未使用的观测中，是否有维度能提供关于"接下来会发生什么"的预判信息？

### 0.3 信号缺口判断

综合 0.1 和 0.2，判断当前奖励函数的信号覆盖状态：
- **信号齐全但校准问题**：所有相关观测已被使用，终止模式与组件激活模式一致 → 问题在权重/阈值/数学形式。走 §1 行为诊断。
- **信号缺失**：存在未使用的观测维度，且该维度可能解释当前终止模式 → 优先考虑新增组件使用该维度。走 §2 的"第0步发现信号缺口 → add 新组件"路径。
- **不确定**：在 §1 诊断中同时保留两种可能性。

### 0.4 僵尸组件检查

#4 组件表中 active_rate < 2% 的组件 → 该组件设计意图未实现，应删除、替换或改造其触发条件。

## 1. 行为诊断

综合第 0 步结论、#3 累积记录、#4 训练反馈：

1. **agent 在做什么？** 快速失败 / 慢速徘徊 / 刷分 exploit？若 #3 累积记录中 len 从高位断崖暴跌且至今未恢复 → 暴跌那轮的修改大概率是根因。

2. **干预哪个目标？** 结合第 0 步缺口判断和组件证据。只干预一个目标。

3. **这个方向还值得继续吗？** 看 #3 累积记录。若同一方向的改动连续 ≥ 3 轮预判 ❌ → 这些修补在治标。**考虑 Level 3 重建而非继续修。**

## 2. 选择干预层级

**Level 1 — 尺度修复**：职责完备、数学形态合理，只是系数/阈值异常。
- `|penalty per-step| / |progress per-step| > 0.5` 且 active_rate ≈ 100% → 降系数至 0.1~0.3x。

**Level 2 — 结构变换**：缺职责、active_rate 接近 0、数学形态塌缩。每轮只改一个组件。

| 证据 | 变换 |
|---|---|
| active_rate < 5% | 二值 → 连续 bounded factor |
| 极端值支配 reward | 无界 → 有界 |
| 占据好状态即持续获奖 | 绝对值 → 改善量 `next - cur` |
| 约束在无关阶段妨碍探索 | 全局惩罚 → 局部门控 |
| 独立目标可互相补偿 | 加权和 → 乘积或几何平均 |
| 乘积经常塌缩为 0 | 乘积 → 几何平均 |
| proxy 提高但外部分数不升 | proxy → 对齐任务完成 |
| 第 0 步发现信号缺口 | **add 新组件** |

**Level 3 — 重建骨架**：
- #3 累积记录中连续 ≥ 3 轮预判 ❌，len 长期未恢复，或同一骨架族已迭代 ≥ 4 轮未刷新 best。
- 重建时：根据 #6 完整公式算子库选不同于已尝试过的主信号框架，基于 #3 累积记录避开已失败的路径。#3 记录了所有历史尝试和它们的因果——用它来决定新骨架应该有什么、不应该有什么。

## 正常模式 vs 重建模式

- **正常模式**：修改一个组件。输出 Level 1 或 Level 2 的诊断。
- **重建模式**（用户 prompt 标有 REBUILD MODE）：你不是在修改上一轮代码——你是在基于全部历史设计新骨架。可以参考 #2 代码中的可用信号声明，但不要受其结构约束。输出 Level 3 的诊断。

# 设计校准（写代码前检查）

1. **新惩罚系数**：目标 per-step ≤ 主信号 per-step 的 0.3x。主信号 per-step ≈ episode_sum_mean / len。
2. **hinge 阈值**：设在终止边界的 60-80% 处。
3. **gate 不塌缩**：在"不理想但安全"区域 gate ≥ 0.3。
4. **单组件 ≤ 2x 主信号**。
5. **总惩罚负担**：所有惩罚的 per-step 合计 ≤ 主信号 per-step 的 0.5x。若 #3 累积记录中 len 自某轮常驻惩罚加入后暴跌且未恢复 → 优先削弱它而非加新东西。

# 代码约束

- 只用 #5 环境事实声明的 obs/action 维度和索引。
- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止 import、class、try/except、eval/exec/open。
- 平方根 `** 0.5`；指数 `2.718281828 ** exponent`。
- 正常模式每轮只改一个组件；重建模式可以重写。
- 签名 `def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`

# 输出

```markdown
# 设计理由
（正常模式：改了什么组件、为什么、数学形式、系数校准）
（重建模式：为什么以前都失败了、新骨架选了什么算子、和已尝试过的有什么本质不同）

```python
def compute_reward(...):
    ...
```

# 诊断摘要
- **audit**: （第 0 步的一句话结论）
- **behavior**: （agent 在做什么）
- **signal**: （缺什么或什么过强）
- **level**: Level 1 / Level 2 / Level 3（系统会据此决定是否进入重建模式）
- **hypothesis**: （为什么这个修改应改善）
- **risk**: （最可能的副作用）
```

```

## User Prompt

```markdown
# ⚠️ REBUILD MODE
系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。
参考 #6 完整公式算子库选新的主信号框架，基于 #3 累积记录避开已失败的路径。
不要受上一轮代码结构约束。


# 1. Search objective
- target_score: 200.000000
- current_score: -80.853856
- gap_to_target: 280.853856
- target_achievement_ratio: -40.427%

# 2. 上一轮奖励函数代码（该轮得分: -80.853856）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Soft landing proxy reward (main learning signal)
    landing_reward = 0.0
    if next_left > 0.5 and next_right > 0.5:
        pos_factor = 2.718281828 ** (-(next_x ** 2) / (2 * 0.0025))
        speed_n = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
        spd_factor = 2.718281828 ** (-(speed_n ** 2) / (2 * 0.04))
        ang_n = abs(next_angle)
        ang_factor = 2.718281828 ** (-(ang_n ** 2) / (2 * 0.01))
        landing_reward = 10.0 * pos_factor * spd_factor * ang_factor

    # 2. Progress reward: reduction in distance to target
    dist_now = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist_now - dist_next

    near_target = dist_now < 0.5
    gate = 1.0
    if near_target:
        gate = 1.0 / (1.0 + 10.0 * (y_vel ** 2) + 5.0 * (angle ** 2))
    progress_reward = delta_dist * gate

    # 3. Action efficiency penalty
    action_cost = -0.01 if action != 0 else 0.0

    # 4. Safety penalty (replaces boundary_penalty)
    # Penalise dangerous descent: too fast downward speed when close to ground,
    # amplified by body tilt.
    height_limit = 0.3
    v_limit = 0.2          # safe downward speed threshold (negative means down, so -y_vel positive)
    proximity = max(0.0, 1.0 - y_pos / height_limit)  # [0,1] when y_pos < 0.3
    danger_speed = max(0.0, -y_vel - v_limit)         # >0 when downward speed exceeds limit
    attitude = 1.0 + 2.0 * abs(angle)                 # tilt penalty multiplier
    safety_penalty = -0.2 * danger_speed * proximity * attitude

    # 5. Light angle/angular-velocity penalty
    angle_penalty = -0.01 * abs(angle) - 0.001 * abs(ang_vel)

    total_reward = (landing_reward + progress_reward +
                    action_cost + safety_penalty + angle_penalty)

    components = {
        "landing_soft_reward": landing_reward,
        "progress": progress_reward,
        "action_cost": action_cost,
        "safety_penalty": safety_penalty,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 775.10 | -115.30 | ✅ |
| 2 | 新奖励在高度低、速度小、姿态好的区间提供密集梯度，引导 agent 进入着陆走廊；active_rate 将从 0... | 新奖励在高度低、速度小、姿态好的区间提供密集梯度，引导 agent 进入着陆走廊；active_rate 将从 0... | 1000.00 | -18.80 | ✅ |
| 3 | 用稀疏但高额的接触成功奖励替换 exploit 源，使 agent 只有真正安全双腿着陆才能获得最大回报，prog... | 用稀疏但高额的接触成功奖励替换 exploit 源，使 agent 只有真正安全双腿着陆才能获得最大回报，prog... | 501.05 | -112.84 | ❌ |
| 4 | 移除惩罚并将其转化为 progress 的门控因子，消除“越快死越赚”的激励，agent 将恢复生存并逐步改善姿态... | 移除惩罚并将其转化为 progress 的门控因子，消除“越快死越赚”的激励，agent 将恢复生存并逐步改善姿态... | 131.15 | -115.49 | ❌ |
| 5 | 恢复密集的接近‑安全奖励（landing_approach_reward）将为 agent 提供连续梯度，引导其靠... | 恢复密集的接近‑安全奖励（landing_approach_reward）将为 agent 提供连续梯度，引导其靠... | 1000.00 | -55.78 | ❓ |
| 6 | 骨架变化: action_cost + angle_penalty + boundary_penalty + l | — | 68.30 | -117.78 | ❌ |
| 7 | 添加连续 safety_penalty 迫使 agent 在低空减速并保持竖直，延长生存时间，进而有机会触发 la... | 添加连续 safety_penalty 迫使 agent 在低空减速并保持竖直，延长生存时间，进而有机会触发 la... | 103.55 | -80.85 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-80.853856, len=103.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.190127, 16.648258]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_soft_reward | 4.059274 | 54.9% | 54.9% | 1.1% |
| safety_penalty | -1.475376 | -19.9% | 19.9% | 9.5% |
| progress | 0.936338 | 12.7% | 15.6% | 100.0% |
| action_cost | -0.483500 | -6.5% | 6.5% | 46.7% |
| angle_penalty | -0.228760 | -3.1% | 3.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: mean_eval_reward=-80.85, mean_ep_len=103.55, terminated=20/20. landing_soft_reward active 1.1% but 54.9% signed_share; safety_penalty active 9.5% with -19.9% share. Progress always on but only 12.7% share.

**Component Anomalies**: landing_soft_reward is dead (1.1% active) yet dominates positive share. safety_penalty rare (9.5%) with large negative share. angle_penalty constant tiny. action_cost moderate.

**Training Dynamics**: no temporal snapshots; unable to assess trends.

**Signal Quality**: landing reward threshold rarely crossed (next_left>0.5 & next_right>0.5). Sparse signals cause high variance. Missing consistent attractor for landing behavior.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
该任务是一个 2D 飞行器（或着陆器）的精确软着陆问题。一个带两条支撑腿的飞行器从上方某个位置开始，施加一个随机初始力。核心目标是在最短时间内以最少的发动机推力安全降落在中心目标平台上，实现两条支撑腿同时接触平台、姿态接近垂直、速度几乎为零的稳定停靠。Agent 必须学会高效地靠近目标区域、减速、保持姿态稳定并建立安全接触。次要目标是降低发动机使用频率和总动作数，以节省燃料。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float (likely float32)
- obs[0] (`x_position`): 水平坐标，相对于目标平台的中心。可用于计算到目标的水平距离。
  - reward_usable: true
- obs[1] (`y_position`): 垂直坐标，相对于目标平台的高度。可用于高度/距离计算。
  - reward_usable: true
- obs[2] (`x_velocity`): 水平线速度。用于速度惩罚或接触条件。
  - reward_usable: true
- obs[3] (`y_velocity`): 垂直线速度。用于着陆软硬判定。
  - reward_usable: true
- obs[4] (`body_angle`): 身体朝向角度（以弧度计，0 表示竖直）。用于姿态稳定性约束。
  - reward_usable: true
- obs[5] (`angular_velocity`): 角速度。用于姿态变化惩罚。
  - reward_usable: true
- obs[6] (`left_support_contact`): 左支撑腿接触目标平台标志（1.0 接触，0.0 未接触）。关键着陆信号。
  - reward_usable: true
- obs[7] (`right_support_contact`): 右支撑腿接触目标平台标志。关键着陆信号。
  - reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` – 不启动任何发动机，滑行。
- action 1: `left_orientation_engine` – 启动左侧姿态发动机，产生向左旋转的力矩，调整身体角度。
- action 2: `main_engine` – 启动主发动机，提供向上的推力（对抗重力或减速）。
- action 3: `right_orientation_engine` – 启动右侧姿态发动机，产生向右旋转的力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 可能由 `body_not_awake_or_settled` 触发，当飞行器两条腿都接触目标平台、速度极低、姿态稳定时，身体被判定为“settled”，随后 episode 终止。虽然环境没有提供显式成功标志，但这一条件可作为成功完成的代理。
- failure-like termination: `crash_or_body_contact`（身体其他部位撞击地面或平台）和 `horizontal_position_outside_viewport`（水平飞出视野）都是明显的失败终止。
- ambiguous termination: `body_not_awake_or_settled` 也可能在不稳定或仅单腿接触的情况下触发，因而单独不代表成功，需要结合其他观测区分。
- truncation: 无截断（环境未设定最大步数

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
| 1 | landing_bonus + progress + soft_landing_penalty | -115.30 | -115.30 | 0.00 | 775.10 | landing_bonus=7.699 progress=0.040 soft_landing_penalty=0.203 | new_best |
| 2 | landing_approach_reward + progress + soft_landing_penalty | -18.80 | -18.80 | 0.00 | 1000.00 | landing_approach_reward=2.424 progress=0.036 soft_landing_penalty=0.198 | new_best |
| 3 | contact_success_reward + progress + soft_landing_penalty | -112.84 | -18.80 | -94.04 | 501.05 | contact_success_reward=42.715 progress=0.051 soft_landing_penalty=0.281 | no_meaningful_improvement |
| 4 | contact_success_reward + landing_gate + progress | -115.49 | -18.80 | -96.69 | 131.15 | contact_success_reward=0.722 landing_gate=0.145 progress=0.117 | no_meaningful_improvement |
| 5 | contact_success_reward + landing_approach_reward + progress | -55.78 | -18.80 | -36.97 | 1000.00 | contact_success_reward=50.158 landing_approach_reward=0.424 progress=0.029 | unsolved_stagnation_fresh_restart |
| 6 | action_cost + angle_penalty + boundary_penalty + landing_soft_reward + progress | -117.78 | -18.80 | -98.98 | 68.30 | action_cost=-0.003 angle_penalty=-0.001 boundary_penalty=0.000 landing_soft_reward=0.013 progress=0.013 | no_meaningful_improvement |
| 7 | action_cost + angle_penalty + landing_soft_reward + progress + safety_penalty | -80.85 | -18.80 | -62.05 | 103.55 | action_cost=-0.006 angle_penalty=-0.002 landing_soft_reward=0.033 progress=0.010 safety_penalty=-0.019 | no_meaningful_improvement |

```
