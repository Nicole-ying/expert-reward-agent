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

## 0. 信号覆盖审计（先于诊断，逐项过）

a) **终止 → 前兆**：#5 §5 声明了哪些终止条件？#2 代码里每个终止条件都有前兆软信号吗？
b) **目标 → 进度**：#5 §1 声明的任务目标是什么？#2 代码有没有组件直接给它梯度？
c) **效率信号**：#5 §4 动作维度 ≥ 6 且代码无 action penalty → 备选方向。
d) **僵尸组件**：#4 组件表中 active_rate < 2% → 应删除或改造。
e) **一句话结论**：当前 reward 漏了什么信号？

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
- target_score: 300.000000
- current_score: 294.818019
- gap_to_target: 5.181981
- target_achievement_ratio: 98.273%

# 2. 上一轮奖励函数代码（该轮得分: 294.818019）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号（动作结果）
    hull_angle_next = next_obs[0]          # 躯干倾角，越小越稳
    hull_ang_vel_next = next_obs[1]        # 躯干角速度
    horizontal_velocity_next = next_obs[2] # 前向速度，正值向前

    # ---------- 健康门控：当躯干倾斜过大时自动衰减前向奖励 ----------
    danger_angle = 0.8   # 接近摔倒的阈值（~45°）
    max_angle = 1.2      # 完全关闭主奖励的阈值（~69°）
    # 线性衰减门：在 [0, danger_angle] 恒为 1，在 [danger_angle, max_angle] 从 1 线性降到 0
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号：被门控的前向速度 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 稳定性约束：轻量角速度惩罚，抑制剧烈晃动 ----------
    w_ang_vel = 0.05
    stability_penalty = -w_ang_vel * (hull_ang_vel_next ** 2)

    # ---------- 总奖励与组件字典 ----------
    total_reward = velocity_gated + stability_penalty
    components = {
        "velocity_gated": velocity_gated,
        "stability_penalty": stability_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=294.818019, len=911.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[98.710590, 307.810929]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_gated | 495.511425 | 100.0% | 100.0% | 100.0% |
| stability_penalty | -0.089571 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 双足行走任务。智能体控制一个拥有两条腿（各含髋、膝两个关节）的身体，在不平坦地形上向前行走。核心目标是 **走得尽可能远、尽可能快**，同时 **最小化能量消耗**。要获得好成绩，必须通过协调四肢关节力矩生成稳定、高效的双足步态；一旦身体倾斜过度、摔倒，回合即告失败。任务**不存在明确的到达点**，过程终止于身体摔倒（失败）或到达地形末端（可视为成功完成路程）。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float（具体为 float32/64，下同）
- 各分量含义及 reward_usable 标记：

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | hull_angle | 躯干相对于竖直方向的倾角（弧度） | true |
| 1 | hull_angular_velocity | 躯干的角速度 | true |
| 2 | horizontal_velocity | 前方的线速度（正值表示向前） | true |
| 3 | vertical_velocity | 上下方向线速度 | true |
| 4 | hip1_angle | 腿1髋关节角度 | true |
| 5 | hip1_speed | 腿1髋关节角速度 | true |
| 6 | knee1_angle | 腿1膝关节角度 | true |
| 7 | knee1_speed | 腿1膝关节角速度 | true |
| 8 | leg1_contact | 腿1触地标志（1.0 触地，0.0 离地） | true |
| 9 | hip2_angle | 腿2髋关节角度 | true |
|10 | hip2_speed | 腿2髋关节角速度 | true |
|11 | knee2_angle | 腿2膝关节角度 | true |
|12 | knee2_speed | 腿2膝关节角速度 | true |
|13 | leg2_contact | 腿2触地标志（1.0 触地，0.0 离地） | true |
|14 | lidar_0 | LIDAR 距离传感器 0（前方地形距离） | true（但需谨慎使用） |
|15 | lidar_1 | LIDAR 距离传感器 1 | true |
|16 | lidar_2 | LIDAR 距离传感器 2 | true |
|17 | lidar_3 | LIDAR 距离传感器 3 | true |
|18 | lidar_4 | LIDAR 距离传感器 4 | true |
|19 | lidar_5 | LIDAR 距离传感器 5 | true |
|20 | lidar_6 | LIDAR 距离传感器 6 | true |
|21 | lidar_7 | LIDAR 距离传感器 7 | true |
|22 | lidar_8 | LIDAR 距离传感器 8 | true |
|23 | lidar_9 | LIDAR 距离传感器 9 | true |

> 注：所有字段在理论上都可参与奖励计算。LIDAR 读数可用于感知前方地形起伏，但作为连续控制任务，直接奖励地形平整度可能引入噪声；除非明确需要地形自适应，否则奖励函数不一定要使用它们。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- dtype: float  
- 各维度含义：

| index | name | meaning |
|-------|------|---------|
| 0 | hip_torque_leg1 | 施加到腿1髋关节的力矩，范围 [-1, 1] |
| 1 | knee_torque_leg1 | 施加到腿1膝关节的力矩，范围 [-1, 1] |
| 2 | hip_torque_leg2 | 施加到腿2髋关节的力矩，范围 [-1, 1] |
| 3 | knee_torque_leg2 | 施加到腿2膝关节的力矩，范围 [-1, 1] |

动作是连续的关节力矩，智能体需生成协调的关节动作以形成稳定步态。

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**：`reached_end_of_terrain` — 身体抵达地形终点，说明已走完全程。由于无明确成功标志，该终止仅暗示路径被完整覆盖，必须谨慎使用，不能直接作为奖励。
- **failure-like termination**：`body_fallen_over` — 躯干倾斜过度导致摔倒，这是明确的失败。
- **ambiguous termination**：无其它模式。
- **truncation**：源码中仅返回 `terminated`，无 `truncated` 标记，因此默认所有终止都视为 episode 结束（可能由最大步数截断，但未指明，需假设环境可能包含基于步数的截断）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
  （无 `info` 输出，无法直接得知 `reached_end_of_terrain` 的具体值）
- explicit_failure_flag_available: **false**  
  （同样，`body_fallen_over` 没有作为独立标志提供给 reward 函数）
- allowed_info_fields: 无（`info` 为 `{}`）
- forbidden_or_uncertain_info_fields: 任何需要从 `info` 读取的字段均禁止

> 奖励函数**必须**基于观测序列隐式推断失败风险（例如通过 `hull_angle` 绝对值过大），**绝对不能**依赖 `done` 或任何终止标志。

## 7. 可用于奖励函数的信号

- **姿态/平衡**：
  - `hull_angle`（obs[0]）：躯干偏离竖直的程度，越小越平衡。
  - `hull_angular_velocity`（obs[1]）：倾斜变化快慢，可用于惩罚剧烈晃动。
- **推进速度**：
  - `horizontal_velocity`（obs[2]）：正向速度，越大前进越快。
  - `vertical_velocity`（obs[3]）：可辅助检测颠簸或跳跃，但不一定是主要奖励源。
- **关节状态**：
  - 髋/膝角度与角速度（obs[4..7], obs[9..12]）：可用于限制关节极限、惩罚过大动作或鼓励平滑运动。
- **接触状态**：
  - `leg1_contact`, `leg2_contact`（obs[8], [13]）：反映脚是否着地，可用于检测步态交替或防止双腿同时离地。
- **动作/能量**：
  - `action` 四维力矩：直接反映控制能量，平方和或绝对值和可用来惩罚低效动作。
- **地形感知**：
  - LIDAR 传感器（obs[14..23]）：10个距离读数，描述前方地形轮廓，可用于奖励对地形的适应性（如避免过陡坡），但使用需谨慎，会大幅增加奖励设计复杂度。

# 6. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 7. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | stability_penalty + velocity_gated | 294.82 | 294.82 | 0.00 | 911.20 | stability_penalty=-0.000 velocity_gated=0.331 | new_best |

```
