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
- current_score: -119.080845
- gap_to_target: 319.080845
- target_achievement_ratio: -59.540%

# 2. 上一轮奖励函数代码（该轮得分: -119.080845）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations: both obs and next_obs are 8-dim vectors
    x,        y        = obs[0], obs[1]
    vx,       vy       = obs[2], obs[3]
    angle,    ang_vel  = obs[4], obs[5]
    # next_obs
    nx,       ny       = next_obs[0], next_obs[1]
    nvx,      nvy      = next_obs[2], next_obs[3]
    nangle,   nang_vel = next_obs[4], next_obs[5]
    lcon,     rcon     = next_obs[6], next_obs[7]  # contact flags at next state

    # ----- potential function (smaller values are better) -----
    def potential(px, py, pvx, pvy, pa):
        dist = (px**2 + py**2) ** 0.5
        vel  = (pvx**2 + pvy**2) ** 0.5
        return -(2.0 * dist + 1.0 * vel + 1.0 * abs(pa))

    # Main progress signal: improvement in potential
    pot_old = potential(x, y, vx, vy, angle)
    pot_new = potential(nx, ny, nvx, nvy, nangle)
    progress = pot_new - pot_old
    # Scale factor can be tuned, keep raw for now. Usually we want reward per step in range ~1.0
    main_progress = progress   # expected range roughly [-?..+?], but typical improvement gives ~0.1-1.0

    # ----- fuel efficiency (action cost) -----
    # action 0 = no engine, 1/2/3 = use engine
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ----- extreme tilt hinge (hard safety) -----
    tilt = abs(nangle)
    tilt_limit = 0.5   # radians, strongly tilted
    if tilt > tilt_limit:
        extreme_tilt_penalty = -0.5 * (tilt - tilt_limit)
    else:
        extreme_tilt_penalty = 0.0

    # ----- soft contact encouragement (only when close to target) -----
    dist_to_target = (nx**2 + ny**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + dist_to_target)   # close → 1, far → 0
    contact_bonus = 0.2 * lcon * rcon * proximity_factor

    # ----- total reward -----
    total_reward = main_progress + fuel_penalty + extreme_tilt_penalty + contact_bonus

    components = {
        "potential_delta": main_progress,
        "fuel_penalty": fuel_penalty,
        "extreme_tilt_penalty": extreme_tilt_penalty,
        "stable_contact_bonus": contact_bonus
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-119.080845, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-143.524791, -98.176274]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| potential_delta | 2.060402 | 82.2% | 91.4% | 100.0% |
| stable_contact_bonus | 0.149898 | 6.0% | 6.0% | 1.3% |
| fuel_penalty | -0.066000 | -2.6% | 2.6% | 4.8% |
| extreme_tilt_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境为一个二维轨迹优化任务：代理人（飞行器）初始位于视口顶部中央附近，携带随机初始力。核心任务是**尽快到达并稳定停靠在画面中央的目标着陆垫上**，同时**尽可能减少发动机推力使用**。代理人需要学会接近目标、减速、保持姿态稳定并实现安全接触。

主要目标：在尽量短的时间内，让身体相对着陆垫的位置（x, y）趋近于零，同时速度降为零，姿态保持竖直（body_angle≈0），并使左右两个支撑脚同时接触。  
次要目标：最小化燃料消耗（动作中使用发动机的次数）。  
不应混淆为仅需靠近即可得分的悬浮任务——单纯悬浮不应获得持续奖励，且最终必须实现有接触的稳定停靠。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float64（推断为 float，实际代码中可能 float32/float64，对奖励无影响）  
- 各维度含义（均基于 next_obs 视角，无历史滑动窗口）：

| 索引 | 名称                     | 含义                                                                 | reward_usable |
|------|--------------------------|----------------------------------------------------------------------|---------------|
| 0    | x_position               | 身体水平坐标，相对于着陆垫中心的偏移                                   | true          |
| 1    | y_position               | 身体垂直坐标，相对于着陆垫高度的偏移                                   | true          |
| 2    | x_velocity               | 水平线速度                                                            | true          |
| 3    | y_velocity               | 垂直线速度                                                            | true          |
| 4    | body_angle               | 身体朝向角（弧度）                                                    | true          |
| 5    | angular_velocity         | 角速度                                                                | true          |
| 6    | left_support_contact     | 左支撑脚接触标志（1.0 接触，0.0 未接触）                             | true          |
| 7    | right_support_contact    | 右支撑脚接触标志（1.0 接触，0.0 未接触）                             | true          |

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作表：

| 动作 id | 名称                        | 含义                                                         |
|--------|-----------------------------|------------------------------------------------------------|
| 0      | no_engine                   | 不启动任何引擎，自由漂移                                    |
| 1      | left_orientation_engine     | 点燃左侧姿态引擎（产生逆时针？力矩，用于调整姿态）           |
| 2      | main_engine                 | 点燃主引擎（产生垂直向上的推力？或向下的推力？根据相对坐标系，可能提供垂直方向推力抵消重力/加速） |
| 3      | right_orientation_engine    | 点燃右侧姿态引擎（产生与左侧相反的力矩）                     |

注：虽然动作空间为离散，但动力学为连续（位置、速度、角度）。主引擎和姿态引擎的具体推力方向由底层物理决定，奖励函数只需知道动作 ID 即可识别是否使用了推力（id ≠ 0 时为有燃料消耗的动作）。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  - `body_not_awake_or_settled` 中的 *settled* 状态：当身体处于静止、双支撑脚接触、且位置/姿态满足一定标准时，被判为已停稳，终止回合。此即任务成功的信号。  
  - 从观测推断：若终止发生时，左右接触标志均为 1，位置 (0,0) 附近，角度≈0，速度≈0，则极大概率为成功。
- **failure-like termination**:  
  - `crash_or_body_contact`：身体与不可碰撞部位（如地面或非着陆垫物体）发生接触，或除支撑脚外的部位触地。  
  - `horizontal_position_outside_viewport`：水平位置超出视野边界。  
  - `body_not_awake_or_settled` 中的 *body_not_awake*：身体失去“意识”（可能因高速撞击、翻滚导致），但并非稳定停泊，属于失败。
- **ambiguous termination**:  
  - 仅有终止信号，没有显式 success/failure 标志时，需根据最终观测状态判断成败。  
  - `body_not_awake_or_settled` 内部可能包含成功（settled）和失败（not awake），完全依赖于观测解读。
- **truncation**: 本描述中未见最大步长截断（MASKED_STEP_SOURCE 中 `truncated=False` 始终返回），因此所有终止均为 terminated=True。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **false**  
- allowed_info_fields: 根据 step 源码，info 字典为空，**无任何可用字段**。  
- forbidden_or_uncertain_info_fields: 任何 info 字段均不可用；不得假设存在 `success`、`failure`、`termination_reason` 等。

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
| 1 | extreme_tilt_penalty + fuel_penalty + potential_delta + stable_contact_bonus | -119.08 | -119.08 | 0.00 | 68.30 | extreme_tilt_penalty=-0.001 fuel_penalty=-0.003 potential_delta=0.029 stable_contact_bonus=0.003 | new_best |

```
