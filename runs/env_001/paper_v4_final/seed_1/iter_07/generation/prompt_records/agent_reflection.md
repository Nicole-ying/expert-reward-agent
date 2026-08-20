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
- current_score: 266.371030
- gap_to_target: -66.371030
- target_achievement_ratio: 133.186%

# 2. 上一轮奖励函数代码（该轮得分: 266.371030）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Progress: distance reduction ---
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = dist - next_dist

    # --- Landing incentive: only when legs touch ground ---
    leg_contact = 1.0 if (left_contact > 0.5 or right_contact > 0.5) else 0.0
    speed = (next_vx ** 2 + next_vy ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 3.0 * speed)
    w_landing = 1.0
    landing_incentive = leg_contact * w_landing / (1.0 + next_dist * 5.0) * speed_factor

    # --- Angular velocity penalty (replaces body angle penalty) ---
    w_angvel = 0.05
    safe_angvel = 0.5
    angvel_error = abs(next_angvel) - safe_angvel
    angvel_penalty = -w_angvel * angvel_error if angvel_error > 0 else 0.0

    # --- Total reward ---
    total_reward = w_progress * progress + landing_incentive + angvel_penalty

    components = {
        "progress_reward": w_progress * progress,
        "landing_incentive": landing_incentive,
        "angvel_penalty": angvel_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 985.25 | -70.92 | ✅ |
| 2 | 全局势场 `1/(1+10d)` 使 agent 在所有距离上都能感知方向——靠近原点直接获得更高奖励，不再只依赖... | 全局势场 `1/(1+10d)` 使 agent 在所有距离上都能感知方向——靠近原点直接获得更高奖励，不再只依赖... | 1000.00 | 146.77 | ✅ |
| 3 | 引入接触门控后，悬停收益降至 1/10，agent 有动力降低到足以触发支撑腿接触的高度并完成着陆。收紧角度惩罚使... | 引入接触门控后，悬停收益降至 1/10，agent 有动力降低到足以触发支撑腿接触的高度并完成着陆。收紧角度惩罚使... | 973.80 | 149.81 | ✅ |
| 4 | 将速度因子乘入 landing_incentive 后，高速移动的奖励大幅缩水，迫使 agent 减速至静止以获取... | 将速度因子乘入 landing_incentive 后，高速移动的奖励大幅缩水，迫使 agent 减速至静止以获取... | 847.70 | 180.66 | ✅ |
| 5 | 移除无接触奖励后，agent 必须降落才能获得主奖励，从而加快成功着陆、缩短 episode 并提高 task c... | 移除无接触奖励后，agent 必须降落才能获得主奖励，从而加快成功着陆、缩短 episode 并提高 task c... | 416.70 | 251.57 | ✅ |
| 6 | 通过轻微抑制高角速度，可让着陆姿态更平稳，减少无用震荡，且不会损害已学会的快速着陆策略。 | 通过轻微抑制高角速度，可让着陆姿态更平稳，减少无用震荡，且不会损害已学会的快速着陆策略。 | 289.10 | 266.37 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=266.371030, len=289.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[239.746436, 298.721438]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_incentive | 36.527932 | 96.3% | 96.3% | 15.5% |
| progress_reward | 1.379438 | 3.6% | 3.7% | 95.0% |
| angvel_penalty | -0.002546 | -0.0% | 0.0% | 0.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个2D飞行器从顶部出发，尽快且尽可能少地用引擎推力降落到中央目标垫上，并稳定停靠。主体要求是到达并安全着陆，附属要求是推进效率和姿态平稳。不要把纯粹的时间最短或燃料最少当成独立主目标，它们只是附属优化。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float（推测）
- obs[0]: x_position, 相对目标垫的水平坐标，reward_usable: true
- obs[1]: y_position, 相对目标垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity, 水平线速度，reward_usable: true
- obs[3]: y_velocity, 垂直线速度，reward_usable: true
- obs[4]: body_angle, 机体倾斜角，reward_usable: true
- obs[5]: angular_velocity, 角速度，reward_usable: true
- obs[6]: left_support_contact, 左支撑腿接触标志（0或1），reward_usable: true
- obs[7]: right_support_contact, 右支撑腿接触标志（0或1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，无推力
- action 1: left_orientation_engine — 点燃左姿态引擎，产生顺时针转动效果（具体方向取决于坐标系）
- action 2: main_engine — 点燃主引擎，通常产生向上推力以减速或提供升力
- action 3: right_orientation_engine — 点燃右姿态引擎，产生逆时针转动效果

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 极可能表示机体已稳定停靠并进入休眠，结合观测中位置接近原点、速度极小、至少一个支撑接触时，可判定为成功着陆。
- failure-like termination: crash_or_body_contact（可能与障碍物或地面猛烈碰撞）、horizontal_position_outside_viewport（水平出界）
- ambiguous termination: 如果 body_not_awake_or_settled 发生时位置偏离目标垫或姿态异常，则为失败（如侧翻冻住）。需通过观测信号区分。
- truncation: 无明确最大步数截断说明，但可能存在时间上限；该截断不属于任务成功或失败。

### 5.2 success/failure

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
| 1 | angle_penalty + contact_bonus + progress_reward | -70.92 | -70.92 | 0.00 | 985.25 | angle_penalty=-0.017 contact_bonus=4.073 progress_reward=0.005 | new_best |
| 2 | angle_penalty + landing_incentive + progress_reward | 146.77 | 146.77 | 0.00 | 1000.00 | angle_penalty=-0.001 landing_incentive=0.179 progress_reward=0.002 | new_best |
| 3 | angle_penalty + landing_incentive + progress_reward | 149.81 | 149.81 | 0.00 | 973.80 | angle_penalty=-0.003 landing_incentive=0.270 progress_reward=0.003 | new_best |
| 4 | angle_penalty + landing_incentive + progress_reward | 180.66 | 180.66 | 0.00 | 847.70 | angle_penalty=-0.003 landing_incentive=0.249 progress_reward=0.003 | new_best |
| 5 | angle_penalty + landing_incentive + progress_reward | 251.57 | 251.57 | 0.00 | 416.70 | angle_penalty=-0.003 landing_incentive=0.433 progress_reward=0.003 | target_solved_new_best |
| 6 | angvel_penalty + landing_incentive + progress_reward | 266.37 | 266.37 | 0.00 | 289.10 | angvel_penalty=-0.001 landing_incentive=0.425 progress_reward=0.003 | target_solved_new_best |

```
