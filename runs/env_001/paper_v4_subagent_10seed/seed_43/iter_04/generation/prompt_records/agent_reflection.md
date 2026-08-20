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
- current_score: -112.844162
- gap_to_target: 312.844162
- target_achievement_ratio: -56.422%

# 2. 上一轮奖励函数代码（该轮得分: -112.844162）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on distance to origin)
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_curr - dist_next)  # positive when getting closer

    # 2. Soft landing constraints
    k = 0.5
    desired_y_vel = -k * y_next
    vertical_error = y_vel_next - desired_y_vel
    penalty_y_vel = vertical_error**2
    penalty_x_vel = x_vel_next**2
    angle_error = max(0.0, abs(angle_next) - 0.2)
    penalty_angle = angle_error**2
    ang_vel_error = max(0.0, abs(ang_vel_next) - 0.5)
    penalty_ang_vel = ang_vel_error**2

    w_y_vel = 1.0
    w_x_vel = 1.0
    w_angle = 2.0
    w_ang_vel = 0.5

    soft_landing_penalty = (w_y_vel * penalty_y_vel +
                            w_x_vel * penalty_x_vel +
                            w_angle * penalty_angle +
                            w_ang_vel * penalty_ang_vel)

    # 3. Contact-based success reward (replaces landing_approach_reward)
    # Both legs must be in contact (binary) and conditions must be safe
    contact_flag = min(left_contact, right_contact)  # 0.0 or 1.0

    # Gaussian factors for safe landing: small velocity, upright, low angular velocity
    sigma_vy = 0.2
    sigma_vx = 0.2
    sigma_angle = 0.1
    sigma_angvel = 0.2

    safe_vy = 2.718281828 ** (- (y_vel_next**2) / (sigma_vy**2))
    safe_vx = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx**2))
    safe_angle = 2.718281828 ** (- (angle_next**2) / (sigma_angle**2))
    safe_angvel = 2.718281828 ** (- (ang_vel_next**2) / (sigma_angvel**2))

    contact_success_reward = 200.0 * contact_flag * safe_vy * safe_vx * safe_angle * safe_angvel

    total_reward = progress - soft_landing_penalty + contact_success_reward

    components = {
        'progress': progress,
        'soft_landing_penalty': soft_landing_penalty,
        'contact_success_reward': contact_success_reward
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 775.10 | -115.30 | ✅ |
| 2 | 新奖励在高度低、速度小、姿态好的区间提供密集梯度，引导 agent 进入着陆走廊；active_rate 将从 0... | 新奖励在高度低、速度小、姿态好的区间提供密集梯度，引导 agent 进入着陆走廊；active_rate 将从 0... | 1000.00 | -18.80 | ✅ |
| 3 | 用稀疏但高额的接触成功奖励替换 exploit 源，使 agent 只有真正安全双腿着陆才能获得最大回报，prog... | 用稀疏但高额的接触成功奖励替换 exploit 源，使 agent 只有真正安全双腿着陆才能获得最大回报，prog... | 501.05 | -112.84 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-112.844162, len=501.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-215.460508, 155.948084]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_penalty | 52.427033 | 82.0% | 82.0% | 100.0% |
| progress | 4.140557 | 6.5% | 18.0% | 100.0% |
| contact_success_reward | 0.000732 | 0.0% | 0.0% | 2.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Eval score -112.8, terminated 20/20, ep_len ~501. soft_landing_penalty dominates (signed_share 82%). Progress small (6.5%). contact_success_reward virtually zero (ep_sum_mean 0.0007).

**Component Anomalies**: contact_success_reward dead (active_rate 2.6%, magnitude_share 0.0%). soft_landing_penalty >70% share; penalty overwhelms progress, yielding negative total reward.

**Training Dynamics**: No temporal snapshots available.

**Signal Quality**: contact_success_reward gate never crossed; missing safe-landing attractor. soft_landing_penalty always active, driving avoidance but no target condition.

**Evidence Confidence**: `high`

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
| 1 | landing_bonus + progress + soft_landing_penalty | -115.30 | -115.30 | 0.00 | 775.10 | landing_bonus=7.699 progress=0.040 soft_landing_penalty=0.203 | new_best |
| 2 | landing_approach_reward + progress + soft_landing_penalty | -18.80 | -18.80 | 0.00 | 1000.00 | landing_approach_reward=2.424 progress=0.036 soft_landing_penalty=0.198 | new_best |
| 3 | contact_success_reward + progress + soft_landing_penalty | -112.84 | -18.80 | -94.04 | 501.05 | contact_success_reward=42.715 progress=0.051 soft_landing_penalty=0.281 | no_meaningful_improvement |

```
