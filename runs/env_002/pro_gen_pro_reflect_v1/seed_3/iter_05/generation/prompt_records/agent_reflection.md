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
- current_score: 319.098622
- gap_to_target: -19.098622
- target_achievement_ratio: 106.366%

# 2. 上一轮奖励函数代码（该轮得分: 319.098622）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Main learning signal: forward velocity, gated by posture health
    forward_vel = next_obs[2]                 # positive = moving forward
    hull_angle = next_obs[0]                  # radians, 0 = vertical
    health = 1.0 / (1.0 + 10.0 * (hull_angle ** 2))   # posture gate
    forward_progress = 1.0 * forward_vel * health

    # Stability: keep torso upright (relaxed safety net)
    tilt_penalty = -2.0 * (hull_angle ** 2)

    # Efficiency: light action cost
    avg_action_power = (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2) / 4.0
    action_cost = -0.05 * avg_action_power

    total_reward = forward_progress + tilt_penalty + action_cost

    components = {
        'forward_progress': forward_progress,
        'tilt_penalty': tilt_penalty,
        'action_cost': action_cost
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 1010.20 | 217.05 | ✅ |
| 2 | 将前进奖励与姿态健康直接乘积耦合，迫使 agent 为获得高速奖励而主动维持直立，从而在速度与稳定性之间自动平衡，... | 将前进奖励与姿态健康直接乘积耦合，迫使 agent 为获得高速奖励而主动维持直立，从而在速度与稳定性之间自动平衡，... | 885.10 | 284.14 | ✅ |
| 3 | 降低 tilt_penalty 系数将减少在加速/倾斜阶段的惩罚负担，允许 agent 在保持健康门控的前提下追求... | 降低 tilt_penalty 系数将减少在加速/倾斜阶段的惩罚负担，允许 agent 在保持健康门控的前提下追求... | 793.45 | 311.39 | ✅ |
| 4 | 移除无用组件可简化奖励结构，避免无关信号干扰后续策略微调，预期性能持平或轻微改善。 | 移除无用组件可简化奖励结构，避免无关信号干扰后续策略微调，预期性能持平或轻微改善。 | 958.70 | 319.10 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=319.098622, len=958.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[317.629809, 320.426292]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 497.898325 | 96.9% | 96.9% | 100.0% |
| action_cost | -13.347650 | -2.6% | 2.6% | 100.0% |
| tilt_penalty | -2.407466 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一项 2D 双足移动任务。智能体需要控制两条腿的髋、膝共四个关节的连续力矩，驱动身体在不同地形上尽可能远、尽可能快地向前行走，同时尽量减少能量消耗。机器人必须保持直立，避免摔倒，因为摔倒将直接终止当前回合。次级目标包括：保持平稳步态、不出现过多不必要的跳跃或抖动、使用合理的关节力矩。该任务不能混淆为定点到达或生存平衡：核心驱动信号是前进速度，成功条件是到达地形尽头，而非存活时长。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float32（连续值）  
- obs[0]: hull_angle – 本体与垂线的夹角（弧度），可用于惩罚倾斜，保持直立。reward_usable: true  
- obs[1]: hull_angular_velocity – 本体角速度，用于抑制旋转抖动。reward_usable: true  
- obs[2]: horizontal_velocity – 水平线速度（正向为前进），是主进度信号。reward_usable: true  
- obs[3]: vertical_velocity – 垂直线速度，可用于限制跳跃/过重落地。reward_usable: true  
- obs[4]: hip1_angle – 腿1髋关节角度。reward_usable: potentially  
- obs[5]: hip1_speed – 腿1髋关节角速度。reward_usable: potentially  
- obs[6]: knee1_angle – 腿1膝关节角度。reward_usable: potentially  
- obs[7]: knee1_speed – 腿1膝关节角速度。reward_usable: potentially  
- obs[8]: leg1_contact – 腿1是否接触地面（1.0/0.0），可用于步态协调。reward_usable: true  
- obs[9]: hip2_angle – 腿2髋关节角度。reward_usable: potentially  
- obs[10]: hip2_speed – 腿2髋关节角速度。reward_usable: potentially  
- obs[11]: knee2_angle – 腿2膝关节角度。reward_usable: potentially  
- obs[12]: knee2_speed – 腿2膝关节角速度。reward_usable: potentially  
- obs[13]: leg2_contact – 腿2是否接触地面。reward_usable: true  
- obs[14]~obs[23]: lidar_0~lidar_9 – 10个前方激光雷达测距值，反映地形轮廓，早期可不依赖。reward_usable: true，但不强制使用。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- continuous: true  
- bounds: 每个关节力矩限制在 [-1.0, 1.0]  
- action[0]: hip_torque_leg1 – 腿1髋关节力矩  
- action[1]: knee_torque_leg1 – 腿1膝关节力矩  
- action[2]: hip_torque_leg2 – 腿2髋关节力矩  
- action[3]: knee_torque_leg2 – 腿2膝关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain – 到达地形尽头，视为成功完成。  
- failure-like termination: body_fallen_over – 机器人摔倒，视为失败。  
- ambiguous termination: 无。  
- truncation: 环境中无内部步数截断（外部训练循环可能限定最大步数）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （step 返回 info 为空字典）  
- explicit_failure_flag_available: false （同上）  
- allowed_info_fields: [] （空字典，无任何允许字段）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均为非法，因为环境中不提供任何信息字段。  
注意：奖励函数无法访问 `terminated` 或 `done`，必须完全基于 `obs`、`next_obs`、`action` 推断状态好坏。

## 7. 可用于奖励函数的信号
- position: 无直接位移坐标，可通过对 horizontal_velocity 在时间上的积分近似，但奖励中通常直接使用速度信号。  
- velocity: next_obs[2]（或 obs[2]）horizontal_velocity 作为前进速度；obs[3]/next_obs[3] vertical_velocity 可用于惩罚跳跃。  
- orientation: next_obs[0] hull_angle 衡量姿态倾斜；next_obs[1] hull_angular_velocity 抑制旋转。  
- contact: next_obs[8] leg1_contact, next_obs[13] leg2_contact，用于步态协调。  
- action/engine: 本次动作 action[0:4] 可用于力矩惩罚（能耗）；也可使用动作变化量（需结合上一步动作，但接口未提供，可放弃）。  
- other: lidar 信号（next_obs[14:24] 或 obs[14:24]），可能用于感知前方地形，不强制使用。

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
| 1 | action_cost + angular_velocity_penalty + forward_progress + tilt_penalty | 217.05 | 217.05 | 0.00 | 1010.20 | action_cost=-0.025 angular_velocity_penalty=-0.001 forward_progress=0.160 tilt_penalty=-0.088 | new_best |
| 2 | action_cost + angular_velocity_penalty + forward_progress + tilt_penalty | 284.14 | 284.14 | 0.00 | 885.10 | action_cost=-0.024 angular_velocity_penalty=-0.001 forward_progress=0.169 tilt_penalty=-0.078 | new_best |
| 3 | action_cost + angular_velocity_penalty + forward_progress + tilt_penalty | 311.39 | 311.39 | 0.00 | 793.45 | action_cost=-0.024 angular_velocity_penalty=-0.001 forward_progress=0.224 tilt_penalty=-0.034 | target_solved_new_best |
| 4 | action_cost + forward_progress + tilt_penalty | 319.10 | 319.10 | 0.00 | 958.70 | action_cost=-0.024 forward_progress=0.248 tilt_penalty=-0.032 | target_solved_new_best |

```
