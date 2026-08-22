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
# 1. Search objective
- target_score: 200.000000
- current_score: 146.416659
- gap_to_target: 53.583341
- target_achievement_ratio: 73.208%

# 2. 上一轮奖励函数代码（该轮得分: 146.416659）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Scale-fix: orientation_penalty weakened 10× to unbind approach capability.
    Keeps safe_progress and soft_landing unchanged; active-rate drop expected.
    """

    # ── Unpack observations ──────────────────────────────────────────
    px0, py0 = obs[0], obs[1]          # last position
    px1, py1 = next_obs[0], next_obs[1]  # current position
    vx1, vy1 = next_obs[2], next_obs[3]  # current velocity
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ──────────────────────────────────────────────
    dist_prev  = (px0**2 + py0**2) ** 0.5
    dist_next  = (px1**2 + py1**2) ** 0.5
    speed      = (vx1**2 + vy1**2) ** 0.5

    # ── 1. Safe progress (speed-gated advancement) ───────────────────
    raw_progress = dist_prev - dist_next   # positive when approaching
    progress     = max(0.0, raw_progress)

    k_target      = 1.5
    gate_strength = 3.0

    expected_speed = k_target * dist_next
    excess_speed   = max(0.0, speed - expected_speed)
    speed_gate     = 1.0 / (1.0 + gate_strength * excess_speed**2)

    safe_progress  = progress * speed_gate

    # ── 2. Orientation / stability penalties (weakened 10×) ─────────
    angle_penalty  = -0.01 * (angle1 ** 2)
    angvel_penalty = -0.005 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ── 3. Soft landing guidance (unchanged) ─────────────────────────
    proximity_threshold = 0.3
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor   = 1.0 / (1.0 + 10.0 * speed)
        angle_factor   = 1.0 / (1.0 + 5.0 * (angle1**2))
        soft_landing   = contact_factor * speed_factor * angle_factor
    else:
        soft_landing = 0.0

    # ── Combine ──────────────────────────────────────────────────────
    total_reward = (
        1.0 * safe_progress
        + 1.0 * orientation_penalty
        + 2.0 * soft_landing
    )

    components = {
        "safe_progress":       safe_progress,
        "orientation_penalty": orientation_penalty,
        "soft_landing":        soft_landing,
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 436.40 | 163.33 | ✅ |
| 2 | 二次项使大步前进的奖励显著更高，agent 将更早到达 proximity 区域并仍由 `soft_landing... | 二次项使大步前进的奖励显著更高，agent 将更早到达 proximity 区域并仍由 `soft_landing... | 95.85 | -30.30 | ❌ |
| 3 | 骨架变化: orientation_penalty + safe_progress + soft_landing | — | 1000.00 | -11.06 | ❌ |
| 4 | 降低 penalty 系数 10 倍后，agent 将敢于倾斜和转动，从而有效接近目标，触发 soft_landi... | 降低 penalty 系数 10 倍后，agent 将敢于倾斜和转动，从而有效接近目标，触发 soft_landi... | 812.95 | 146.42 | ➖ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=146.416659, len=812.950000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[36.722380, 212.679957]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 69.212113 | 98.0% | 98.0% | 12.5% |
| safe_progress | 1.382024 | 2.0% | 2.0% | 70.3% |
| orientation_penalty | -0.055110 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: mean_eval_reward=146.4, ep_len=812.95, terminated=16/20. Reward dominated by soft_landing (98% signed share, active only 12.5% of steps).

**Component Anomalies**: soft_landing: dominating (98% share, mean=69.2) but sparse (12.5% active). orientation_penalty: 100% active, negligible magnitude (share -0.1%). safe_progress: 70.3% active, tiny share (2%).

**Training Dynamics**: No temporal checkpoint data available.

**Signal Quality**: Dead signal: orientation_penalty always present but effectively zero. Sparse gate: soft_landing fires rarely, creating sparse reward attractor. safe_progress provides weak signal despite frequent activation. Missing dense progress incentive.

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

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | orientation_penalty + progress_delta + soft_landing | 163.33 | 163.33 | 0.00 | 436.40 | orientation_penalty=-0.001 progress_delta=0.011 soft_landing=0.118 | new_best |
| 2 | orientation_penalty + progress_delta + soft_landing | -30.30 | 163.33 | -193.63 | 95.85 | orientation_penalty=-0.001 progress_delta=0.016 soft_landing=0.014 | no_meaningful_improvement |
| 3 | orientation_penalty + safe_progress + soft_landing | -11.06 | 163.33 | -174.39 | 1000.00 | orientation_penalty=-0.003 safe_progress=0.003 soft_landing=0.474 | no_meaningful_improvement |
| 4 | orientation_penalty + safe_progress + soft_landing | 146.42 | 163.33 | -16.91 | 812.95 | orientation_penalty=-0.000 safe_progress=0.005 soft_landing=0.373 | unsolved_high_achievement_continue_from_best |

```
