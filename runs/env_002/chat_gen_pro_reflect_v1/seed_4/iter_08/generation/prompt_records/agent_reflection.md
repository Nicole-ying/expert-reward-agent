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
- current_score: 281.820049
- gap_to_target: 18.179951
- target_achievement_ratio: 93.940%

# 2. 上一轮奖励函数代码（该轮得分: 281.820049）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (linear main signal)
    #   Simple proportional reward for forward speed, bounded by
    #   the environment's natural velocity limits.
    # ============================================================
    forward_reward = 1.0 * max(0.0, horizontal_vel)

    # ============================================================
    # Component 2: soft_health_gate
    #   Attenuates forward reward when tilt approaches fall threshold.
    #   gate = 1.0 for |angle| ≤ 0.25, decays to 0.0 at |angle| ≥ 0.5.
    # ============================================================
    gate_lower = 0.25
    gate_upper = 0.5
    gate_raw = max(0.0, 1.0 - (abs(hull_angle) - gate_lower) / (gate_upper - gate_lower))
    gate_factor = gate_raw
    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (light action smoothness)
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Total reward (angle_hinge_penalty removed – never triggered)
    # ============================================================
    total_reward = gated_forward + energy_penalty

    # ============================================================
    # Components dict
    # ============================================================
    components = {
        'forward_reward': forward_reward,
        'gate_factor': gate_factor,
        'gated_forward': gated_forward,
        'energy_penalty': energy_penalty,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 820.35 | 166.54 | ✅ |
| 2 | 将 `angle_penalty` 改为 hinge 并在 `gate_factor` 中引入更宽的免衰减区，可以... | 将 `angle_penalty` 改为 hinge 并在 `gate_factor` 中引入更宽的免衰减区，可以... | 901.50 | 287.89 | ✅ |
| 3 | 加入动作幅度惩罚会迫使 agent 探索更节能的步态，从而拉动外部得分中能量项的提升，逼近 300 分 | 加入动作幅度惩罚会迫使 agent 探索更节能的步态，从而拉动外部得分中能量项的提升，逼近 300 分 | 86.10 | -82.07 | ❌ |
| 4 | 将能量惩罚系数降至 -0.02 可使前进速度信号重新主导梯度，agent 恢复有效步态探索，len 和 score... | 将能量惩罚系数降至 -0.02 可使前进速度信号重新主导梯度，agent 恢复有效步态探索，len 和 score... | 1032.25 | 297.68 | ✅ |
| 5 | 在危险角度区间加入微弱但持续的平方惩罚，能提供姿态修正梯度，使步态更稳定、能耗更低，从而补齐 300 分的最后 2... | 在危险角度区间加入微弱但持续的平方惩罚，能提供姿态修正梯度，使步态更稳定、能耗更低，从而补齐 300 分的最后 2... | 905.75 | 281.45 | ❌ |
| 6 | 平方速度奖励提供更强的“更快更好”梯度，配合 gate 约束，可推动步态加速并守住平衡，从而将外部得分从 ~297... | 平方速度奖励提供更强的“更快更好”梯度，配合 gate 约束，可推动步态加速并守住平衡，从而将外部得分从 ~297... | 136.45 | -59.50 | ❌ |
| 7 | 回归线性速度奖励后，agent 将重新学习在 gate 约束下稳定行走，恢复 ~297 的 baseline 分数。 | 回归线性速度奖励后，agent 将重新学习在 gate 约束下稳定行走，恢复 ~297 的 baseline 分数。 | 1036.40 | 281.82 | ➖ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=281.820049, len=1036.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[16.907162, 304.671329]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 1927.051562 | 57.7% | 57.7% | 100.0% |
| gated_forward | 900.076313 | 26.9% | 26.9% | 99.8% |
| forward_reward | 484.644203 | 14.5% | 14.5% | 99.8% |
| energy_penalty | -28.607421 | -0.9% | 0.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：虽然能量消耗是优化项，但核心驱动力是前进速度和距离，能量是附属约束；任务不是单纯保持平衡（生存），也不是精确导航到某个点，而是持续前进的 locomotion 任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推测)
- obs[0]: hull_angle，身体相对于竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，身体角速度，reward_usable: true
- obs[2]: horizontal_velocity，前后线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1地面接触标志（1.0=接触，0.0=无接触），reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2地面接触标志（1.0=接触，0.0=无接触），reward_usable: true
- obs[14..23]: lidar_0..9，10个LIDAR测距仪沿前方地形的距离测量值，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- dtype: float32
- bounds: [-1.0, 1.0] per dimension
- action[0]: hip_torque_leg1，腿1髋关节施加的力矩
- action[1]: knee_torque_leg1，腿1膝关节施加的力矩
- action[2]: hip_torque_leg2，腿2髋关节施加的力矩
- action[3]: knee_torque_leg2，腿2膝关节施加的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain — 成功到达地形末端，属于成功终止。
- failure-like termination: body_fallen_over — 身体摔倒，属于失败终止。
- ambiguous termination: 无。
- truncation: 无显式截断（step 返回 False 作为 truncation 标志）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false — info 字典为空 {}，没有显式 success 标志。
- explicit_failure_flag_available: false — info 字典为空 {}，没有显式 failure 标志。
- allowed_info_fields: 无（info 为空）。
- forbidden_or_uncertain_info_fields: 无（info 为空）。

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，但可通过 horizontal_velocity 积分或 LIDAR 间接推断
- velocity: horizontal_velocity (obs[2])，vertical_velocity (obs[3])
- orientation: hull_angle (obs[0])，hull_angular_velocity (obs[1])
- contact: leg1_contact (obs[8])，leg2_contact (obs[13])
- action/engine: action 本身（4维力矩），可计算动作幅度、变化率
- other: LIDAR 距离 (obs[14..23])，关节角度和速度 (obs[4..7], obs[9..12])

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
| 1 | angle_penalty + angular_vel_penalty + balance_penalty + forward_reward + gate_factor + gated_forward | 166.54 | 166.54 | 0.00 | 820.35 | angle_penalty=-0.034 angular_vel_penalty=-0.000 balance_penalty=-0.034 forward_reward=0.173 gate_factor=0.889 | new_best |
| 2 | angle_deviation + angle_penalty + angular_vel_penalty + balance_penalty + forward_reward + gate_factor | 287.89 | 287.89 | 0.00 | 901.50 | angle_deviation=0.005 angle_penalty=-0.020 angular_vel_penalty=-0.000 balance_penalty=-0.021 forward_reward=0.236 | new_best |
| 3 | energy_penalty + forward_reward + gate_factor + gated_forward | -82.07 | 287.89 | -369.95 | 86.10 | energy_penalty=-0.178 forward_reward=0.224 gate_factor=1.396 gated_forward=0.352 | no_meaningful_improvement |
| 4 | energy_penalty + forward_reward + gate_factor + gated_forward | 297.68 | 297.68 | 0.00 | 1032.25 | energy_penalty=-0.036 forward_reward=0.304 gate_factor=1.610 gated_forward=0.528 | new_best |
| 5 | angle_hinge_penalty + energy_penalty + forward_reward + gate_factor + gated_forward | 281.45 | 297.68 | -16.23 | 905.75 | angle_hinge_penalty=-0.001 energy_penalty=-0.038 forward_reward=0.275 gate_factor=1.658 gated_forward=0.485 | no_meaningful_improvement |
| 6 | energy_penalty + forward_reward + gate_factor + gated_forward | -59.50 | 297.68 | -357.18 | 136.45 | energy_penalty=-0.030 forward_reward=0.262 gate_factor=1.377 gated_forward=0.410 | no_meaningful_improvement |
| 7 | energy_penalty + forward_reward + gate_factor + gated_forward | 281.82 | 297.68 | -15.86 | 1036.40 | energy_penalty=-0.036 forward_reward=0.297 gate_factor=1.632 gated_forward=0.513 | unsolved_high_achievement_continue_from_best |

```
