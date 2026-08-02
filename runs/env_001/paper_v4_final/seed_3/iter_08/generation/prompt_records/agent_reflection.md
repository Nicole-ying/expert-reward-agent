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
- current_score: 123.308295
- gap_to_target: 76.691705
- target_achievement_ratio: 61.654%

# 2. 上一轮奖励函数代码（该轮得分: 123.308295）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observation indices: 0:x, 1:y, 2:vx, 3:vy, 4:angle, 5:ang_vel, 6:left_contact, 7:right_contact
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_lc = next_obs[6]  # left contact
    n_rc = next_obs[7]  # right contact

    # ---------- 1. progress towards origin (distance decrease) ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 30.0

    # ---------- 2. attitude gate: suppress progress when angle is dangerous ----------
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (12.0 * (abs_angle - 0.30)) / 
                              (1.0 + 2.718281828 ** (12.0 * (abs_angle - 0.30))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.08          # ← 唯一修改：从 0.6 降为 0.08
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty (smooth rotation) ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. contact-based landing proxy ----------
    mean_contact = (n_lc + n_rc) / 2.0

    k_y = 10.0
    k_vy = 8.0
    k_ang = 15.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs(n_angle))

    contact_landing_factor = (mean_contact * f_y * f_vy * f_ang) ** 0.25
    w_contact_land = 5.0
    contact_landing_reward = w_contact_land * contact_landing_factor

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + contact_landing_reward)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_landing_reward": contact_landing_reward
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 101.60 | -111.37 | ✅ |
| 2 | 将二次稳定性惩罚改为角度门控（gate进步奖励），在危险角度区域强力抑制progress，迫使agent学习保持小... | 将二次稳定性惩罚改为角度门控（gate进步奖励），在危险角度区域强力抑制progress，迫使agent学习保持小... | 68.70 | -103.62 | ✅ |
| 3 | 将接触条件从 `min` 改为均值，使单腿触地也能获得部分奖励，提供连续梯度；agent 将开始尝试保持触地并逐步... | 将接触条件从 `min` 改为均值，使单腿触地也能获得部分奖励，提供连续梯度；agent 将开始尝试保持触地并逐步... | 70.55 | -71.40 | ✅ |
| 4 | 全局连续的 soft landing proxy 会让 agent 在每个 step 都能感知“接近地面且稳定”的... | 全局连续的 soft landing proxy 会让 agent 在每个 step 都能感知“接近地面且稳定”的... | 915.45 | -178.95 | ❌ |
| 5 | 引入接触传感器作为着陆奖励的触发条件，将全局 soft proxy 替换为接触门控的着陆奖励，迫使 agent 必... | 引入接触传感器作为着陆奖励的触发条件，将全局 soft proxy 替换为接触门控的着陆奖励，迫使 agent 必... | 668.35 | 30.06 | ✅ |
| 6 | 提升 progress 和 lateral 惩罚权重能使 agent 在保持成功着陆的前提下优化飞行效率与中心对准... | 提升 progress 和 lateral 惩罚权重能使 agent 在保持成功着陆的前提下优化飞行效率与中心对准... | 936.95 | -13.38 | ❌ |
| 7 | 将 `w_lat_pos` 回调至接近第 5 轮水平，惩罚强度降至主信号的 0.3x 以内，总奖励将恢复正向驱动，... | 将 `w_lat_pos` 回调至接近第 5 轮水平，惩罚强度降至主信号的 0.3x 以内，总奖励将恢复正向驱动，... | 1000.00 | 123.31 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=123.308295, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[81.681663, 156.942525]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 2969.395453 | 98.6% | 98.6% | 61.0% |
| progress_gated | 37.419699 | 1.2% | 1.4% | 100.0% |
| lateral_pos_penalty | -1.446742 | -0.0% | 0.0% | 100.0% |
| angvel_penalty | -0.042195 | -0.0% | 0.0% | 47.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。飞行器从视口顶部中央附近启动，受到随机初始外力。主要目标是**尽快、平稳地降落在中央的目标平台上**（即到达并稳定停留在着陆点），同时

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
| 1 | landing_bonus + lateral_drift_penalty + progress + stability_penalty | -111.37 | -111.37 | 0.00 | 101.60 | landing_bonus=0.057 lateral_drift_penalty=-0.033 progress=0.147 stability_penalty=-0.028 | new_best |
| 2 | angvel_penalty + landing_bonus + lateral_drift_penalty + progress_gated | -103.62 | -103.62 | 0.00 | 68.70 | angvel_penalty=-0.020 landing_bonus=0.016 lateral_drift_penalty=-0.039 progress_gated=0.144 | new_best |
| 3 | angvel_penalty + landing_bonus + lateral_drift_penalty + progress_gated | -71.40 | -71.40 | 0.00 | 70.55 | angvel_penalty=-0.019 landing_bonus=0.017 lateral_drift_penalty=-0.039 progress_gated=0.146 | new_best |
| 4 | angvel_penalty + landing_reward + lateral_pos_penalty + progress_gated | -178.95 | -71.40 | -107.55 | 915.45 | angvel_penalty=-0.001 landing_reward=1.111 lateral_pos_penalty=-0.025 progress_gated=0.003 | no_meaningful_improvement |
| 5 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | 30.06 | 30.06 | 0.00 | 668.35 | angvel_penalty=-0.002 contact_landing_reward=2.151 lateral_pos_penalty=-0.017 progress_gated=0.026 | new_best |
| 6 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | -13.38 | 30.06 | -43.44 | 936.95 | angvel_penalty=-0.002 contact_landing_reward=2.420 lateral_pos_penalty=-0.046 progress_gated=0.108 | no_meaningful_improvement |
| 7 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | 123.31 | 123.31 | 0.00 | 1000.00 | angvel_penalty=-0.002 contact_landing_reward=2.166 lateral_pos_penalty=-0.008 progress_gated=0.120 | new_best |

```
