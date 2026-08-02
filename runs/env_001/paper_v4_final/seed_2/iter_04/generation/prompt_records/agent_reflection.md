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
- current_score: 222.690009
- gap_to_target: -22.690009
- target_achievement_ratio: 111.345%

# 2. 上一轮奖励函数代码（该轮得分: 222.690009）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs dimensions per environment card
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    # obs[6], obs[7] not used in this version

    # 1. Primary progress signal: dense quadratic penalty on position and velocity errors.
    pos_sq_error = x_pos**2 + y_pos**2
    vel_sq_error = x_vel**2 + y_vel**2
    progress = -0.05 * pos_sq_error - 0.1 * vel_sq_error

    # 2. Stability constraint: quadratic penalty on body angle and angular velocity.
    pose_penalty = -5.0 * (body_angle**2) - 0.5 * (angular_vel**2)

    # 3. Approach & soft landing bonus: now based on proximity to origin, not contact.
    #    proximity = 1 at (0,0), decays with squared distance.
    proximity = 1.0 / (1.0 + 10.0 * (x_pos**2 + y_pos**2))
    speed_magnitude = abs(x_vel) + abs(y_vel)
    speed_factor = 1.0 / (1.0 + 5.0 * speed_magnitude)
    angle_factor = 1.0 / (1.0 + 20.0 * abs(body_angle))
    landing_bonus = 2.0 * proximity * speed_factor * angle_factor

    total_reward = progress + pose_penalty + landing_bonus

    components = {
        'progress': progress,
        'pose_penalty': pose_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 530.15 | -102.51 | ✅ |
| 2 | 连续化接触因子将使 agent 在单脚触地时即获得奖励，大幅提升 reward 密度，从而引导它学习降落到平台并最... | 连续化接触因子将使 agent 在单脚触地时即获得奖励，大幅提升 reward 密度，从而引导它学习降落到平台并最... | 1000.00 | -6.91 | ✅ |
| 3 | 位置接近奖励将创造指向原点的正向梯度，引导 agent 下降并尝试着陆；速度/角度因子保证着陆质量。 | 位置接近奖励将创造指向原点的正向梯度，引导 agent 下降并尝试着陆；速度/角度因子保证着陆质量。 | 547.70 | 222.69 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=222.690009, len=547.700000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[112.761295, 275.006661]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 426.356525 | 94.4% | 94.4% | 100.0% |
| pose_penalty | -15.447610 | -3.4% | 3.4% | 100.0% |
| progress | -10.022957 | -2.2% | 2.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 飞行器（着陆器）轨迹优化任务。主体从视口顶部中心附近出发，初始受到随机扰动。核心目标是 **尽快、平稳地降落到中央目标平台上，并保持机身竖直稳定**。次要目标是 **尽量节省主引擎燃料**，即少用主推力。  
Agent 需要学会：向目标平台逼近、适时减速、保持小角度、最终实现低冲击的安全着陆。  
不要把“平稳着陆”与单纯的“位置到达”混淆，着陆质量（速度、姿态、接触）与燃料效率不可忽略，但到达目标是第一优先级。

## 3. 观察空间 observation_space
- **type**: `Box`
- **shape**: `[8]`
- **dtype**: `float32` (推断)
- **obs[0]**: `x_position` —— 相对于目标平台中心的水平坐标（正右方向），reward_usable: true
- **obs[1]**: `y_position` —— 相对于目标平台着陆面高度的垂直坐标（疑似上为正，平台面为 0），reward_usable: true
- **obs[2]**: `x_velocity` —— 水平线速度，reward_usable: true
- **obs[3]**: `y_velocity` —— 垂直线速度，reward_usable: true
- **obs[4]**: `body_angle` —— 机身倾角（很可能以弧度表示，0 为竖直），reward_usable: true
- **obs[5]**: `angular_velocity` —— 角速度，reward_usable: true
- **obs[6]**: `left_support_contact` —— 左支撑脚接地标志（1.0 表示接触），reward_usable: true
- **obs[7]**: `right_support_contact` —— 右支撑脚接地标志，reward_usable: true

## 4. 动作空间 action_space
- **type**: `Discrete`
- **n**: 4
- **动作清单**：
  - **action 0**: `no_engine` —— 所有引擎关闭
  - **action 1**: `left_orientation_engine` —— 启动左姿态引擎（产生角力矩，主要用于调整机头方向）
  - **action 2**: `main_engine` —— 启动主引擎（向下喷气，产生向上的推力，同时可能带来微小角力矩）
  - **action 3**: `right_orientation_engine` —— 启动右姿态引擎（与左姿态引擎相反方向）

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：机体与地面或平台发生强烈接触（可能包含坠毁或非常粗糙的着陆），触发终止。
- **horizontal_position_outside_viewport**：水平位置超出有效视野/世界边界，任务失败终止。
- **body_not_awake_or_settled**：机体进入稳定/不活跃状态（如着陆后静止），终止发生。此极可能是成功着陆后的正常终止。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 无（环境 step 返回的 `info` 为空字典 `{}`）  
- **forbidden_or_uncertain_info_fields**: 所有可能的终止原因、成败标记、elapsed steps 等均不可直接使用  
- 尽管如此，成功着陆的迹象可通过 **next_obs** 间接推断：
  - 位置接近目标原点 `(0,0)`，速度接近 0，角度接近 0，且左右支撑接地标志同时为 1。  
  - 该推断路径记为 **derived_possible**，可在奖励设计中使用，但不可作为绝对成功判决。

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
| 1 | landing_bonus + pose_penalty + progress | -102.51 | -102.51 | 0.00 | 530.15 | landing_bonus=2.237 pose_penalty=-0.109 progress=-0.066 | new_best |
| 2 | landing_bonus + pose_penalty + progress | -6.91 | -6.91 | 0.00 | 1000.00 | landing_bonus=2.317 pose_penalty=-0.142 progress=-0.066 | new_best |
| 3 | landing_bonus + pose_penalty + progress | 222.69 | 222.69 | 0.00 | 547.70 | landing_bonus=0.738 pose_penalty=-0.104 progress=-0.048 | target_solved_new_best |

```
