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
- current_score: -113.405732
- gap_to_target: 313.405732
- target_achievement_ratio: -56.703%

# 2. 上一轮奖励函数代码（该轮得分: -113.405732）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- unpack observations -------------------
    x,  y  = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle      = obs[4]
    angvel     = obs[5]
    left_leg   = obs[6]
    right_leg  = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle  = next_obs[4]
    n_angvel = next_obs[5]
    n_left   = next_obs[6]
    n_right  = next_obs[7]

    # ------------------- helper quantities -------------------
    dist      = (x**2  + y**2)  ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    vel_abs       = (vx**2 + vy**2) ** 0.5
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 1.0
    w_proximity = 10.0
    w_fuel = 0.2   # new: light per-step fuel penalty

    th_angle  = 0.5
    th_vel    = 1.0
    th_angvel = 2.0
    th_dist   = 0.5

    gate_min = 0.1
    gate_min_stab = 0.2

    # ------------------- 1. progress signal (distance delta) -------------------
    delta_dist = max(0.0, dist - next_dist)

    gate_angle  = max(gate_min, 1.0 - abs(angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - vel_abs      / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * delta_dist * gate

    # ------------------- 2. proximity + stability reward -------------------
    prox_factor = max(0.0, 1.0 - next_dist / th_dist)

    a_stab  = max(gate_min_stab, 1.0 - abs(n_angle)  / th_angle)
    v_stab  = max(gate_min_stab, 1.0 - next_vel_abs   / th_vel)
    av_stab = max(gate_min_stab, 1.0 - abs(n_angvel)  / th_angvel)
    stab = a_stab * v_stab * av_stab

    contact_flag = 1.0 if (n_left + n_right) >= 1.0 else 0.0
    contact_mult = 1.0 + 0.5 * contact_flag

    proximity_stability_reward = w_proximity * prox_factor * stab * contact_mult

    # ------------------- 3. fuel penalty (new) -------------------
    engine_on = 1.0 if action != 0 else 0.0   # any engine usage
    fuel_penalty = -w_fuel * engine_on

    # ------------------- total reward -------------------
    total_reward = progress_gated + proximity_stability_reward + fuel_penalty

    components = {
        'progress_gated':   progress_gated,
        'proximity_stability': proximity_stability_reward,
        'fuel_penalty':      fuel_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.45 | -110.63 | ✅ |
| 2 | 密集的 proximity_stability 信号（预期 ~2.0/step）将对抗环境惩罚，引导 agent ... | 密集的 proximity_stability 信号（预期 ~2.0/step）将对抗环境惩罚，引导 agent ... | 372.45 | 98.82 | ✅ |
| 3 | 加入轻量 per-step 燃料消耗惩罚（‑0.2/step）将梯度引导策略减少不必要的引擎点火，从而提升环境内置... | 加入轻量 per-step 燃料消耗惩罚（‑0.2/step）将梯度引导策略减少不必要的引擎点火，从而提升环境内置... | 68.40 | -113.41 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-113.405732, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-140.041657, -96.181510]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_stability | 10.757301 | 94.2% | 94.2% | 16.5% |
| fuel_penalty | -0.510000 | -4.5% | 4.5% | 3.7% |
| progress_gated | 0.151162 | 1.3% | 1.3% | 91.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Eval score=-113.41, all 20 episodes terminated early (mean len=68.4). Original env reward=-1.635/step dominates; generated reward=0.143/step is an order of magnitude too small to compensate. The lander crashes quickly in all episodes.

**Component Anomalies**: proximity_stability: 94.2% signed share but only 16.5% active rate — high value when it fires but gates rarely open. progress_gated: 91.9% active but only 1.3% share (ep sum=0.15) — gate is permissive but delta_dist is near-zero. fuel_penalty: effectively dead (3.7% active).

**Training Dynamics**: No component dynamics snapshots available — temporal trends across checkpoints could not be inspected. Static picture only: the three generated components sum to ~0.14/step vs env's -1.64/step.

**Signal Quality**: Generated reward cannot reach the agent: it's ~11x smaller than the native negative signal. proximity_stability gate thresholds (th_angle=0.5, th_vel=1.0, th_angvel=2.0, gate_min_stab=0.2) restrict activation to 16.5%. progress_gated is open but progress (delta_dist) is negligible — the lander doesn't move toward origin.

**Evidence Confidence**: `high`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
任务核心是控制一个具有两条支撑腿的 2D 飞行器（启动时带有随机初始扰动），从视口顶部中央附近出发，安全、平稳地降落到画面中央的水平目标平台上，并稳定停靠。主要目标是抵达目标位置并实现“软着陆”（低速、姿态竖直、支撑腿接触），尽量减少发动机使用量（燃料消耗），同时鼓励快速完成。附属目标为姿态保持、节能及时间效率，但不应与安全降落冲突，也不应被误认为单纯的点对点导航或纯粹的平衡维持任务。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: float32（根据 Box 推断）
- **各维含义与 reward_usable 属性**：
  - **obs[0]**: x_position — 水平坐标，相对于目标平台的水平偏移，reward_usable: **true**
  - **obs[1]**: y_position — 垂直坐标，相对于目标平台高度的偏移，reward_usable: **true**
  - **obs[2]**: x_velocity — 水平线速度，reward_usable: **true**
  - **obs[3]**: y_velocity — 垂直线速度，reward_usable: **true**
  - **obs[4]**: body_angle — 机体倾斜角度，reward_usable: **true**
  - **obs[5]**: angular_velocity — 机体角速度，reward_usable: **true**
  - **obs[6]**: left_support_contact — 左支撑腿接触标志（1.0 表示接触，0.0 表示未接触），reward_usable: **true**
  - **obs[7]**: right_support_contact — 右支撑腿接触标志，reward_usable: **true**

所有观测字段均可直接用于奖励计算。

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **具体动作与含义**：
  - **action 0**: no_engine — 不启动任何引擎，自由滑行
  - **action 1**: left_orientation_engine — 启动左侧姿态引擎，产生使机体逆时针（或对应方向）旋转的力矩
  - **action 2**: main_engine — 启动主引擎，产生垂直向上的推力（减速或悬停）
  - **action 3**: right_orientation_engine — 启动右侧姿态引擎，产生与左引擎反向的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：身体稳定停靠（body_not_awake_or_settled）且至少有一只支撑腿接触地面，且没有发生 crash 或出界。这是期望的成功状态，表现为速度极小、姿态接近竖直、接触信号为 1，但无法从 info 直接读取，必须通过观测信号间接推断。
- **failure-like termination**：
  - crash_or_body_contact：身体主体（非支撑腿）接触地面或其他碰撞导致坠毁，通常与高速、大角度撞击有关。
  - horizontal_position_outside_viewport：水平位置超出可显示边界，即机体飞离有效区域。
- **ambiguous termination**：body_not_awake_or_settled 但左右支撑腿均未接触——可能代表机体已倒地且静止，本质上属于失败。
- **truncation**：未提及显式 step 限制，但可能存在隐式最大步数（环境未披露），此时 info 为空字典，无法直接识别。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false （info 为空字典，无任何成功标志）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无（info 为空）
- **forbidden_or_uncertain_info_fields**: 所有通常可能存在于 info 中的字段如 "success"、"failure"、"termination_reason"、"reward_components" 等均不存在，且不得假设它们可用。终止条件只能通过观测组合（位置、速度、角度、接触）以及是否在达到稳定/边界时 episode 结束来间接推断，标记为 **derived_possible**。

## 7. 可用于奖励函数的信号
- **位置信号**：`obs[0] x_position`、`obs[1] y_position`、`next_obs[0]`、`next_obs[1]`。可用于计算到目标 (0,0) 的距离、高度误差等。
- **速度信号**：`obs[2] x_velocity`、`obs[3] y_velocity`、`next_obs` 中对应项。可用于惩罚高速撞击或奖励低速软着陆。
- **姿态信号**：`obs[4] body_angle`、`obs[5] angular_velocity`、`next_obs` 对应项。可用于鼓励竖直姿态和减少旋转。
- **接触信号**：`obs[6] left_support_contact`、`obs[7] right_support_contact`、`next_obs` 对应项。可用于奖励支撑腿接触，表示着陆成功。
- **动作/引擎信号**：`action` 取值可用于计算燃料消耗（若 action ≠ 0 则为引擎启用）。
- **衍生推断信号（derived_possible）**：
  - 邻近成功：当 `next_obs` 中支撑腿接触为 1，且 `next_obs` 的 `x_velocity`、`y_velocity`、`body_angle` 接近 0，`y_position` 接近 0，可推断为成功软着陆。虽然无法从 info 获得标识，但在连续奖励中可通过组合条件给出额外奖励。
  - 坠毁推断：`next_obs` 中 `body_angle` 突然大幅偏离 0 或 `y_position` 突变（被重置），可间接推测崩溃，但不要用于奖励，仅用于诊断。
  - 出界推断：`x_position` 超出合理范围（如 >1 或 <-1），可用于惩罚，但此时环境已终止，一般不需要奖励。

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
| 1 | progress_gated + soft_landing | -110.63 | -110.63 | 0.00 | 68.45 | progress_gated=0.002 soft_landing=0.008 | new_best |
| 2 | progress_gated + proximity_stability | 98.82 | 98.82 | 0.00 | 372.45 | progress_gated=0.001 proximity_stability=3.735 | new_best |
| 3 | fuel_penalty + progress_gated + proximity_stability | -113.41 | 98.82 | -212.22 | 68.40 | fuel_penalty=-0.017 progress_gated=0.002 proximity_stability=0.158 | no_meaningful_improvement |

```
