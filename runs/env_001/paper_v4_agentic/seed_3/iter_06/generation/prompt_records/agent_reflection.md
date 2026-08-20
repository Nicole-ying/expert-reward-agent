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
- current_score: 241.723383
- gap_to_target: -41.723383
- target_achievement_ratio: 120.862%

# 2. 上一轮奖励函数代码（该轮得分: 241.723383）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆分
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft_contact = next_obs[6]
    nright_contact = next_obs[7]

    w_progress = 20.0
    w_landing = 0.5            # 降低系数，防止瞬时奖励过载
    w_land_vel = 10.0
    w_angle = 0.5
    w_angvel = 0.5
    engine_cost = 0.02

    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5

    # 1. 距离改进（保持不变）
    progress = w_progress * (dist - ndist)

    # 2. 着陆质量软信号 —— 仅在双腿同时接触时激活
    if nleft_contact > 0.5 and nright_contact > 0.5:
        altitude_factor = max(0.0, 1.0 - abs(ny) / 0.5)
        align_factor    = max(0.0, 1.0 - abs(nx) / 0.5)
        vx_factor       = max(0.0, 1.0 - abs(nvx) / 0.3)
        vy_factor       = max(0.0, 1.0 - abs(nvy) / 0.5)
        angle_factor    = max(0.0, 1.0 - abs(nangle) / 0.2)
        product = (altitude_factor * align_factor * vx_factor *
                   vy_factor * angle_factor)
        if product > 0.0:
            landing_quality = w_landing * (product ** (1.0 / 5.0))
        else:
            landing_quality = 0.0
    else:
        landing_quality = 0.0

    # 3. 着陆速度惩罚（仅在双腿接触时）
    fcontact = float(nleft_contact * nright_contact)
    if fcontact > 0.5:
        vel_pen = -w_land_vel * (nvx**2 + nvy**2)
    else:
        vel_pen = 0.0

    # 4. 姿态稳定惩罚（全程）
    att_penalty = -w_angle * (nangle**2) - w_angvel * (nangvel**2)

    # 5. 引擎使用惩罚
    eng_pen = -engine_cost if action != 0 else 0.0

    total_reward = progress + landing_quality + vel_pen + att_penalty + eng_pen
    components = {
        "progress": progress,
        "landing_quality": landing_quality,
        "landing_velocity_penalty": vel_pen,
        "attitude_penalty": att_penalty,
        "engine_cost": eng_pen
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 817.15 | 99.05 | ✅ |
| 2 | 连续化的 landing_quality（含接触偏置）将在接近平台时提供稳定梯度，引导 agent 在最终阶段减速... | 连续化的 landing_quality（含接触偏置）将在接近平台时提供稳定梯度，引导 agent 在最终阶段减速... | 682.10 | 96.28 | ❌ |
| 3 | 将 progress 系数提升 4 倍，使其每步贡献与持续惩罚相当或更高，恢复 progress 作为主导引力，让... | 将 progress 系数提升 4 倍，使其每步贡献与持续惩罚相当或更高，恢复 progress 作为主导引力，让... | 363.30 | 170.64 | ✅ |
| 4 | 将位置因子容忍半径从 0.2 拓宽到 0.5，使着陆信号提前 2.5 倍范围激活，为 agent 提供连续的接近梯... | 将位置因子容忍半径从 0.2 拓宽到 0.5，使着陆信号提前 2.5 倍范围激活，为 agent 提供连续的接近梯... | 990.55 | -12.98 | ❌ |
| 5 | 门控使徘徊的净收益消失，agent 被迫完成双腿着陆才能获得正向信号，episode 长度将缩短，得分回升。 | 门控使徘徊的净收益消失，agent 被迫完成双腿着陆才能获得正向信号，episode 长度将缩短，得分回升。 | 443.45 | 241.72 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=241.723383, len=443.450000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[139.580747, 291.345925]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 50.283116 | 56.6% | 56.6% | 23.4% |
| progress | 27.652928 | 31.1% | 31.8% | 97.2% |
| engine_cost | -7.417000 | -8.3% | 8.3% | 83.6% |
| attitude_penalty | -2.431167 | -2.7% | 2.7% | 100.0% |
| landing_velocity_penalty | -0.431218 | -0.5% | 0.5% | 21.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Automatic fallback after 5 turns without submit. Raw data: [inspect_component_dynamics]: (no monitor snapshots — training may not have completed)
[inspect_previous_reward]: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0

**Component Anomalies**: Subagent exhausted turns without explicit submission.

**Training Dynamics**: No temporal analysis available.

**Signal Quality**: No signal quality assessment available.

**Evidence Confidence**: `low`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：控制一个 2D 飞行器从初始位置（通常靠近视口顶部中央）出发，尽可能快地降落到场景中央的目标平台上，并以低速度、稳定姿态安全停稳，使两条支撑腿同时接触平台。次要目标：在完成任务的过程中，尽量减少引擎使用量（节省燃料、减少推力）。不应将姿态摆动最小化或单纯的速度最小化作为独立目标，这些只是达成安全着陆的附属约束。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 通常为 float32（匿名环境未明确，但符合连续观测惯例）
- 各维度含义：
  - obs[0]: x_position（水平坐标，相对于目标平台中心的偏移）—— reward_usable: true
  - obs[1]: y_position（垂直坐标，相对于平台高度基准的偏移）—— reward_usable: true
  - obs[2]: x_velocity（水平线速度）—— reward_usable: true
  - obs[3]: y_velocity（垂直速度）—— reward_usable: true
  - obs[4]: body_angle（机身倾斜角度）—— reward_usable: true
  - obs[5]: angular_velocity（角速度）—— reward_usable: true
  - obs[6]: left_support_contact（左侧支撑腿接触标志，1.0表示接触）—— reward_usable: true
  - obs[7]: right_support_contact（右侧支撑腿接触标志，1.0表示接触）—— reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 各动作含义：
  - action 0: no_engine —— 不启动任何引擎（惯性运动）
  - action 1: left_orientation_engine —— 启动左侧方向引擎（产生逆时针或顺时针力矩，改变姿态）
  - action 2: main_engine —— 启动主引擎（产生向上的推力，通常用于减速或上升）
  - action 3: right_orientation_engine —— 启动右侧方向引擎（产生与左侧引擎相反的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：`body_not_awake_or_settled` 为真，并且可以通过观测信号交叉验证：两条支撑腿均接触平台（obs[6] 和 obs[7] 都为 1.0）、水平位置接近 0（obs[0] ≈ 0）、垂直速度接近 0、姿态角接近水平。这种情况暗示飞行器已稳定停靠在目标平台上。
- **failure-like termination**：`crash_or_body_contact`（主体与地形或其他物体发生不期望的接触，导致损毁）、`horizontal_position_outside_viewport`（水平位置超出屏幕边界，飞行器脱离有效区域）。
- **ambiguous termination**：`body_not_awake_or_settled` 为真，但两条支撑腿未同时接触平台，或者位置不在平台附近。这可能是飞行器在平台外静止但未悬空（例如已经坠毁但引擎关闭或卡在地形中），需通过位置和接触信号判别。在初始学习阶段，部分此类终止可视为失败。
- **truncation**：无显式截断逻辑，`info` 为空，`truncated` 返回 False，即 episode 仅在触发上述终止条件时结束。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（`info` 内无任何成功标记）
- explicit_failure_flag_available: false（`info` 内无任何失败标记）
- allowed_info_fields: 无（`info` 返回空字典）
- forbidden_or_uncertain_info_fields: 所有 `info` 字段（因为没有声明任何可用字段，且原环境可能将奖励或终止原因隐藏在 `info` 中，但根据要求不能假设其存在，因此全部禁止使用）

## 7. 可用于奖励函数的信号
- **位置类**：`x_position` (obs[0])，`y_position` (obs[1])。可计算与目标平台的水平距离和垂直距离，用于引导接近。
- **速度类**：`x_velocity` (obs[2])，`y_velocity` (obs[3])。可用于惩罚着陆时的冲击速度，或在飞行阶段鼓励平滑性。
- **姿态类**：`body_angle` (obs[4])，`angular_velocity` (obs[5])。用于要求安全着陆时的姿态稳定性（尽量接近水平）。
- **接触类**：`left_support_contact` (obs[6])，`right_support_contact` (obs[7])。两腿同时接触平台是成功着陆的必要条件，可据此构造着陆奖励。
- **动作/引擎类**：`action`。可惩罚引擎使用（no_engine 不惩罚，其余动作惩罚）以鼓励节省燃料。
- **派生推断信号（derived_possible）**：
  - 成功着陆指示器：可从 `body_not_awake_or_settled` 导致 episode 结束，且 `obs[6]` 和 `obs[7]` 均为 1.0、obs[0] 接近 0、obs[3] 接近 0 间接推断。可在奖励函数中结合 next_obs 构造着陆成功奖励，但需谨慎使用，因为无法直接读取终止原因。
  - 失败着陆指示器：可从 episode 结束时 `crash_or_body_contact` 或出界未接触双足推断，但同样无法在奖励计算时直接获取，只能通过观测模式判断（如 `next_obs` 中位置突变、速度极大等）。

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
| 1 | attitude_penalty + engine_cost + landing_quality + landing_velocity_penalty + progress | 99.05 | 99.05 | 0.00 | 817.15 | attitude_penalty=-0.042 engine_cost=-0.013 landing_quality=0.305 landing_velocity_penalty=-0.046 progress=0.033 | new_best |
| 2 | attitude_penalty + engine_cost + landing_quality + landing_velocity_penalty + progress | 96.28 | 99.05 | -2.76 | 682.10 | attitude_penalty=-0.059 engine_cost=-0.012 landing_quality=0.594 landing_velocity_penalty=-0.044 progress=0.032 | no_meaningful_improvement |
| 3 | attitude_penalty + engine_cost + landing_quality + landing_velocity_penalty + progress | 170.64 | 170.64 | 0.00 | 363.30 | attitude_penalty=-0.065 engine_cost=-0.014 landing_quality=0.525 landing_velocity_penalty=-0.047 progress=0.133 | new_best |
| 4 | attitude_penalty + engine_cost + landing_quality + landing_velocity_penalty + progress | -12.98 | 170.64 | -183.62 | 990.55 | attitude_penalty=-0.031 engine_cost=-0.014 landing_quality=1.061 landing_velocity_penalty=-0.026 progress=0.061 | no_meaningful_improvement |
| 5 | attitude_penalty + engine_cost + landing_quality + landing_velocity_penalty + progress | 241.72 | 241.72 | 0.00 | 443.45 | attitude_penalty=-0.043 engine_cost=-0.014 landing_quality=0.191 landing_velocity_penalty=-0.033 progress=0.109 | target_solved_new_best |

```
