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
- current_score: 232.573910
- gap_to_target: -32.573910
- target_achievement_ratio: 116.287%

# 2. 上一轮奖励函数代码（该轮得分: 232.573910）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态（用于距离计算）
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    l_contact = next_obs[6]
    r_contact = next_obs[7]

    # ---------- 1. 进度奖励：向目标 (0,0) 靠近 ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next
    progress_reward = 1.0 * progress

    # ---------- 2. 着陆预备与接触奖励 ----------
    prox = 1.0 / (1.0 + 10.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 5.0 * (abs(nx_v) + abs(ny_v)))
    angle_factor = 1.0 / (1.0 + 3.0 * (abs(n_angle) + abs(n_ang_v)))

    # 双腿接触作为连续因子（二值乘积，0 或 1）
    contact_factor = l_contact * r_contact

    # 混合奖励：未接触时保留一半引导，接触时获得完整奖励
    approach_bonus = 2.0 * prox * speed_factor * angle_factor * (0.5 + 0.5 * contact_factor)

    # ---------- 3. 着陆安全性惩罚 ----------
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)

    gate_safety = 1.0 / (1.0 + 5.0 * dist_next)
    landing_safety_penalty = (0.03 * vel_pen + 0.02 * ang_pen + 0.03 * tilt_pen) * gate_safety

    # ---------- 总奖励 ----------
    total_reward = progress_reward + approach_bonus - landing_safety_penalty

    components = {
        "progress_reward": float(progress_reward),
        "approach_bonus": float(approach_bonus),
        "landing_safety_penalty": float(landing_safety_penalty)
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 211.25 | 39.61 | ✅ |
| 2 | 新增接触奖励让agent有动力在接近目标时保持双腿着地姿态，配合landing_safety_penalty抑制速... | 新增接触奖励让agent有动力在接近目标时保持双腿着地姿态，配合landing_safety_penalty抑制速... | 872.65 | 97.70 | ✅ |
| 3 | 尖锐门控与降低系数将迫使 agent 必须驶向目标附近才能获得有效的接触奖励，进而恢复 progress 的主导地... | 尖锐门控与降低系数将迫使 agent 必须驶向目标附近才能获得有效的接触奖励，进而恢复 progress 的主导地... | 1000.00 | 146.75 | ✅ |
| 4 | 用 proximity、speed、angle、contact 四因子乘积取代原有的接触×距离门控，将奖励范围压缩... | 用 proximity、speed、angle、contact 四因子乘积取代原有的接触×距离门控，将奖励范围压缩... | 68.40 | -117.67 | ❌ |
| 5 | 恢复连续接触 × 距离门控的正向奖励，同时将安全惩罚降至主信号的 0.5× 以下，agent 将重新获得向目标移动... | 恢复连续接触 × 距离门控的正向奖励，同时将安全惩罚降至主信号的 0.5× 以下，agent 将重新获得向目标移动... | 1000.00 | -8.04 | ❌ |
| 6 | 骨架变化: approach_bonus + landing_safety_penalty + progress | — | 446.40 | 219.70 | ✅ |
| 7 | 在 approach_bonus 中引入双腿接触因子，使未接触时奖励减半、接触后完整发放，将鼓励 agent 主动... | 在 approach_bonus 中引入双腿接触因子，使未接触时奖励减半、接触后完整发放，将鼓励 agent 主动... | 460.10 | 232.57 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=232.573910, len=460.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[200.292128, 264.009958]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_bonus | 139.370649 | 98.8% | 98.8% | 100.0% |
| progress_reward | 1.380049 | 1.0% | 1.0% | 96.7% |
| landing_safety_penalty | 0.235785 | 0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Final policy achieves mean_eval_reward=232.57, all 20 episodes terminate successfully at mean length 460.1. However, approach_bonus dominates reward composition at 98.8% share (episode sum 139.37), while progress_reward (1.0%, sum=1.38) and landing_safety_penalty (0.2%, sum=0.236) are negligible. Per-step means confirm the imbalance: approach_bonus=0.657 vs progress=0.002 vs safety=0.0007 — a ~329:1 ratio.

**Component Anomalies**: approach_bonus is pathologically dominant (98.8% share), starving the other two components of any learning signal. progress_reward is present (96.7% active) but contributes only 1% of total reward magnitude. landing_safety_penalty is effectively invisible at 0.2% share — far too small to shape safe landing behavior.

**Mechanism Hypothesis**: The 2.0× multiplier on approach_bonus combined with the multiplicative prox-speed-angle structure creates a reward surface where the shaping bonus overwhelms task-aligned progress. The agent can achieve high scores by optimizing approach_bonus alone, bypassing both progress toward target and landing safety constraints. This is a reward-scale dominance problem, not a sparsity problem.

**Decision Implication**: PATCH approach_bonus: reduce its magnitude (e.g., lower the 2.0 coefficient or remove the 0.5 floor on contact_factor) so progress_reward and landing_safety_penalty can contribute meaningfully. Also consider scaling up landing_safety_penalty coefficients (currently 0.02–0.03) by at least 5–10× so it registers against the dominant bonus. Keep the component — don't rebuild.

**Confidence**: `high`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境要求一个 2D 飞行器（带主引擎和两个姿态引擎）从视口顶部中心附近出发，以随机初始速度开始，尽快到达视口中心的着陆平台，并以极低的速度、稳定的姿态安全接触并停稳。主目标是**精确到达目标位置并稳定停靠**；次要目标是**快速完成**和**尽可能少地使用引擎推力**。任务的核心是导航与精确着陆，不应与纯粹的生存或持续前进任务混淆。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测）
- 各维含义：
  - obs[0]（x_position）：水平坐标，相对于目标着陆点的水平偏移，reward_usable: true
  - obs[1]（y_position）：垂直坐标，相对于平台高度的垂直偏移（平台高度处为 0），reward_usable: true
  - obs[2]（x_velocity）：水平线速度，reward_usable: true
  - obs[3]（y_velocity）：垂直线速度，reward_usable: true
  - obs[4]（body_angle）：机体朝向角（弧度），reward_usable: true
  - obs[5]（angular_velocity）：角速度，reward_usable: true
  - obs[6]（left_support_contact）：左侧支撑杆与地面/平台的接触标志（1.0 表示接触），reward_usable: true
  - obs[7]（right_support_contact）：右侧支撑杆接触标志（1.0 表示接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作列表：
  - action 0: no_engine，不开启任何引擎
  - action 1: left_orientation_engine，开启左姿态引擎（产生角加速度，可能向左旋转）
  - action 2: main_engine，开启主引擎（产生向上的推力）
  - action 3: right_orientation_engine，开启右姿态引擎（产生相反方向的角加速度）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  机体在平台上稳定停靠，触发 **body_not_awake_or_settled**（休眠/静止）。此时通常同时满足：两腿接触 flag 均为 1，x_position 和 y_position 接近 0，线速度与角速度均很小，且未发生 crash 或出界。
- failure-like termination:  
  - **horizontal_position_outside_viewport**：机体水平飞出视口边界，直接失败。  
  - **crash_or_body_contact**（非着陆接触）：机体以过大速度、过大角度或接触到非平台区域（如地面以外）触发终止，属于失败。需要结合接触标志和速度判断。
- ambiguous termination:  
  **crash_or_body_contact** 在某些情况下也可能是成功着陆，因为着陆时也会发生身体接触并可能触发该条件。需要进一步通过双腿是否都接触、速度是否低、是否在目标附近来区分。  
  **body_not_awake_or_settled** 也可能是碰撞后卡住不动导致的静止，但碰撞后通常接触标志不会全为 1 且位置会偏离目标，因此可通过位置与接触标志排除模糊性。
- truncation:  
  无显式截断（源码中返回的 truncated 恒为 False）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 始终为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用
- 推断成功/失败的间接路径（derived_possible）：
  - **成功**：episode 终止且满足以下条件 → (left_support_contact == 1.0) and (right_support_contact == 1.0) and (|x_position| 很小) and (|y_position| 很小) and (|body_angle| 很小) and 线速度/角速度很低。  
  - **失败（crash）**：episode 终止但上述条件不成立（例如双腿未同时接触、位置大幅偏离、角度或速度很大）。  
  - **出界**：可通过终止时 |x_position| 显著大于视口半宽推断。

## 7. 可用于奖励函数的信号
- position: x_position, y_position（均可直接获得，表示相对于目标的位置）
- velocity: x_velocity, y_velocity, angular_velocity
- orientation: body_angle
- contact: left_support_contact, right_support_contact
- action/engine: 离散动作 id 可映射为推力状态（无推力、左旋、主推、右旋）；可用于估计燃料消耗、避免无用点火
- other: 从上述信号可派生的距离（euclidean distance, |x|+|y| 等）、接近速度、朝向对齐程度、双腿是否均接触、是否在目标附近等

所有信号均为可直接从 obs 或 next_obs 读取的数值，无量纲但具有物理意义。

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
| 1 | landing_safety_penalty + progress_reward + x_boundary_penalty | 39.61 | 39.61 | 0.00 | 211.25 | landing_safety_penalty=0.003 progress_reward=0.007 x_boundary_penalty=0.000 | new_best |
| 2 | landing_contact_bonus + landing_safety_penalty + progress_reward + x_boundary_penalty | 97.70 | 97.70 | 0.00 | 872.65 | landing_contact_bonus=0.283 landing_safety_penalty=0.003 progress_reward=0.003 x_boundary_penalty=0.000 | new_best |
| 3 | landing_contact_bonus + landing_safety_penalty + progress_reward + x_boundary_penalty | 146.75 | 146.75 | 0.00 | 1000.00 | landing_contact_bonus=0.100 landing_safety_penalty=0.003 progress_reward=0.002 x_boundary_penalty=0.000 | new_best |
| 4 | landing_safety_penalty + precise_landing_bonus + progress_reward + x_boundary_penalty | -117.67 | 146.75 | -264.42 | 68.40 | landing_safety_penalty=0.016 precise_landing_bonus=0.066 progress_reward=0.016 x_boundary_penalty=0.000 | no_meaningful_improvement |
| 5 | landing_contact_bonus + landing_safety_penalty + progress_reward | -8.04 | 146.75 | -154.79 | 1000.00 | landing_contact_bonus=0.847 landing_safety_penalty=0.002 progress_reward=0.004 | no_meaningful_improvement |
| 6 | approach_bonus + landing_safety_penalty + progress_reward | 219.70 | 219.70 | 0.00 | 446.40 | approach_bonus=0.716 landing_safety_penalty=0.001 progress_reward=0.002 | target_solved_new_best |
| 7 | approach_bonus + landing_safety_penalty + progress_reward | 232.57 | 232.57 | 0.00 | 460.10 | approach_bonus=0.657 landing_safety_penalty=0.001 progress_reward=0.002 | target_solved_new_best |

```
