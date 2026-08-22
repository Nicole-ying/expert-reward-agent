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
- target_score: 300.000000
- current_score: -63.340537
- gap_to_target: 363.340537
- target_achievement_ratio: -21.114%

# 2. 上一轮奖励函数代码（该轮得分: -63.340537）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract useful signals ----------
    horizontal_speed = obs[2]          # forward direction
    hull_angle = obs[0]                # tilt (rad)
    hull_angular_velocity = obs[1]     # tilt speed (rad/s)

    # ---------- health gate: close to 1 when upright, decays when tilting ----------
    #   gate = 1 / (1 + 10 * angle^2 + 0.1 * angvel^2)
    #   avoids over-punishing early exploration, but heavily cuts progress reward
    #   when tilt becomes dangerous.
    denom = 1.0 + 10.0 * hull_angle * hull_angle + 0.1 * hull_angular_velocity * hull_angular_velocity
    health_gate = 1.0 / denom

    # ---------- forward progress (only positive direction) ----------
    #   only reward moving forward; ignore backward motion (max to avoid penalizing it)
    fwd_speed = max(0.0, horizontal_speed)
    progress_component = 1.0 * fwd_speed * health_gate   # w_speed = 1.0

    # ---------- moderate action regularisation ----------
    #   small penalty on large joint torques – just enough to avoid extreme signals
    action_sum_sq = action[0]*action[0] + action[1]*action[1] + action[2]*action[2] + action[3]*action[3]
    action_penalty = -0.01 * action_sum_sq

    # ---------- total reward ----------
    total_reward = progress_component + action_penalty

    components = {
        "progress": progress_component,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 246.25 | -59.97 | ✅ |
| 2 | 加入双脚离地与垂直速度惩罚后，agent 将学会抑制不安全的腾空行为，保持至少单脚接地，从而减少摔倒、延长存活并最... | 加入双脚离地与垂直速度惩罚后，agent 将学会抑制不安全的腾空行为，保持至少单脚接地，从而减少摔倒、延长存活并最... | 105.95 | -86.30 | ❌ |
| 3 | 骨架变化: air_stability_penalty + balance_penalty + forward_ | — | 74.80 | -95.84 | ❌ |
| 4 | 移除双脚离地惩罚后，agent 可恢复自然摆动相，存活步数回升至 150+，整体 score 接近 -10 ~ 0... | 移除双脚离地惩罚后，agent 可恢复自然摆动相，存活步数回升至 150+，整体 score 接近 -10 ~ 0... | 376.95 | -74.85 | ❓ |
| 5 | 骨架变化: energy_penalty + forward_reward + hinge_penalty | — | 243.15 | -52.46 | ✅ |
| 6 | 让前进奖励的 gate 在身体摇晃时快速衰减， agent 将学会抑制危险振荡，摔倒率下降，有效存活步数和综合得分... | 让前进奖励的 gate 在身体摇晃时快速衰减， agent 将学会抑制危险振荡，摔倒率下降，有效存活步数和综合得分... | 380.95 | -59.50 | ❌ |
| 7 | 加回 hinge_penalty 提供明确的“保持小倾角”梯度，与双因子门控配合，能进一步降低摔倒率，提升平均得分... | 加回 hinge_penalty 提供明确的“保持小倾角”梯度，与双因子门控配合，能进一步降低摔倒率，提升平均得分... | 401.70 | -52.19 | ❓ |
| 8 | 骨架变化: action_penalty + progress | — | 217.05 | -63.34 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-63.340537, len=217.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-89.731580, -23.407339]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 59.710870 | 96.1% | 96.1% | 95.4% |
| action_penalty | -2.418903 | -3.9% | 3.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
一个双足机器人在崎岖不平、布满障碍（阶梯、树桩、坑洼）的地形上尽可能远且高效地向前行走。机器人配备前向激光雷达，可感知前方地形高度。核心目标是学会利用激光雷达信息调整步态，在不摔倒的前提下持续前进；附属目标是减少不必要的关节扭矩（能量效率）并争取到达地形终点。不应将“到达终点”误解为唯一成功信号——能够稳定行走不摔倒才是关键，终点到达是终止条件之一但无独立奖励标注。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: 连续浮点 + 部分二值
- 详细字段（按索引）：
  - obs[0]: hull_angle (name: hull_angle) — 身体俯仰/倾斜角，reward_usable: true，用于检测摔倒和姿态稳定
  - obs[1]: hull_angular_velocity — 身体角速度，reward_usable: true，辅助姿态惩罚
  - obs[2]: horizontal_speed — 质心水平速度，reward_usable: true，核心前进信号
  - obs[3]: vertical_speed — 质心垂直速度，reward_usable: true，可能帮助判断弹跳或摔倒
  - obs[4]: hip_1_angle — 第1髋关节角度，reward_usable: true（关节状态跟踪）
  - obs[5]: hip_1_speed — 第1髋关节角速度
  - obs[6]: knee_1_angle — 第1膝关节角度
  - obs[7]: knee_1_speed — 第1膝关节角速度
  - obs[8]: hip_2_angle — 第2髋关节角度
  - obs[9]: hip_2_speed — 第2髋关节角速度
  - obs[10]: knee_2_angle — 第2膝关节角度
  - obs[11]: knee_2_speed — 第2膝关节角速度
  - obs[12]: leg_1_ground_contact — 第1腿接地指示（0/1），reward_usable: true，可作为步态接触约束
  - obs[13]: leg_2_ground_contact — 第2腿接地指示，同上
  - obs[14]~[23]: lidar_1~lidar_10 — 10个激光测距仪读数，表示前方地形高度。reward_usable: 谨慎使用，不可直接作为奖励项，但可间接推导预见性调整；初始训练阶段不建议直接奖励，但可帮助分析失败模式。

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- bounds: [-1.0, 1.0] 连续值
- 动作含义：
  - action[0]: hip_1_torque — 施加到第1髋关节的扭矩
  - action[1]: knee_1_torque — 施加到第1膝关节的扭矩
  - action[2]: hip_2_torque — 施加到第2髋关节的扭矩
  - action[3]: knee_2_torque — 施加到第2膝关节的扭矩

所有动作均为连续扭矩控制，无离散动作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形尽头），但无显式成功标志。可视为**成功的行走存活**导致的终止。
- failure-like termination: body_fallen_over（身体摔倒），常见于 hull_angle 过大或质心跳跃、触地异常。
- ambiguous termination: 无。
- truncation: 未定义明确截断（step source 中仅 terminated，无 truncated 分支）。因此所有 episode 结束均由终止条件触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 标志）
- explicit_failure_flag_available: false（同上）
- allowed_info_fields: []（interface 规定 info_is_empty，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为环境实际不提供任何 info。

尽管如此，可**从观测推导**终止类型：
- 摔倒 (derived_possible): 身体倾斜角 |hull_angle| 超出临界阈值（如 >0.5 rad），或 hull_angular_velocity 突变，同时 leg contact 可能消失。
- 到达终点 (derived_possible): 水平速度仍较高、姿态稳定时 episode 突然终止；也可结合上一步位置推断（但观测无位置），只能依赖速度与姿态平滑终止时的表现进行事后推测。但其可靠性不足以成为奖励条件，可偶尔用于事后分析。

## 7. 可用于奖励函数的信号
- position: 观察中无绝对位置，仅可通过速度累积间接推断位移；无直接可用位置坐标。
- velocity: horizontal_speed（obs[2]）、vertical_speed（obs[3]）、各关节速度（obs[5,7,9,11]）
- orientation: hull_angle（obs[0]）、hull_angular_velocity（obs[1]）
- contact: leg_1_ground_contact（obs[12]）、leg_2_ground_contact（obs[13]）
- action/engine: 动作本身（4维扭矩）
- other:
  - laser scan（obs[14:23]）——可用于推断地形粗糙度，但需谨慎映射为奖励时容易引入噪声；暂时建议不作为常规奖励信号。
  - derived_possible: 通过 hull_angle 阈值或角速度突变推断摔倒；通过 episode 终止时水平速度 & 姿态推断“疑似成功到达”。

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
| 1 | balance_penalty + forward_progress | -59.97 | -59.97 | 0.00 | 246.25 | balance_penalty=-0.011 forward_progress=0.185 | new_best |
| 2 | air_stability_penalty + balance_penalty + forward_progress | -86.30 | -59.97 | -26.33 | 105.95 | air_stability_penalty=-0.142 balance_penalty=-0.011 forward_progress=0.215 | no_meaningful_improvement |
| 3 | air_stability_penalty + balance_penalty + forward_reward + terrain_gate + terrain_roughness | -95.84 | -59.97 | -35.87 | 74.80 | air_stability_penalty=-0.075 balance_penalty=-0.005 forward_reward=0.072 terrain_gate=0.501 terrain_roughness=0.214 | no_meaningful_improvement |
| 4 | balance_penalty + forward_reward + terrain_gate + terrain_roughness | -74.85 | -59.97 | -14.88 | 376.95 | balance_penalty=-0.008 forward_reward=0.106 terrain_gate=0.497 terrain_roughness=0.216 | unsolved_stagnation_fresh_restart |
| 5 | energy_penalty + forward_reward + hinge_penalty | -52.46 | -52.46 | 0.00 | 243.15 | energy_penalty=-0.018 forward_reward=0.183 hinge_penalty=-0.001 | new_best |
| 6 | energy_penalty + forward_reward | -59.50 | -52.46 | -7.04 | 380.95 | energy_penalty=-0.018 forward_reward=0.175 | no_meaningful_improvement |
| 7 | energy_penalty + forward_reward + hinge_penalty | -52.19 | -52.19 | 0.00 | 401.70 | energy_penalty=-0.018 forward_reward=0.170 hinge_penalty=-0.004 | unsolved_stagnation_fresh_restart |
| 8 | action_penalty + progress | -63.34 | -52.19 | -11.15 | 217.05 | action_penalty=-0.017 progress=0.187 | no_meaningful_improvement |

```
