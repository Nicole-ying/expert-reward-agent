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
# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: 出现未允许的 obs/next_obs 切片 (record: runs\env_006\paper_bipedal_hardcore_v1\seed_2\iter_03\generation\validations\reward_v3.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由
本轮干预：**Level 2 结构变换 — 新增组件填补信号缺口**。  
核心问题：
- 第 0 步审计发现 **激光雷达（obs[14:23]）完全未被使用**，而环境事实明确指出机器人需利用激光雷达感知前方地形调整步态。
- 第二轮加入 air_stability_penalty 后存活步数从 246 暴跌至 105，且几乎所有 episode 均为早期失败终止，说明 agent 在没有地形感知的情况下无法有效规避障碍，惩罚反而干扰了学习。
- 当前 agent 行为：快速摔倒（短 episode），累积前进奖励很少。
- 信号缺口：缺前方地形粗糙度信息 → agent 在未知障碍上高速冲撞 → 摔倒。

修改方案：
- **修改 forward_progress 组件**，将其由恒等水平速度改为 **地形感知门控速度**：`forward_reward = horizontal_speed * terrain_gate`。
- 用 lidar 读数计算粗糙度（标准差），并映射到 `[0.3, 1.0]` 的 gate 上：平坦地形 gate→1.0（全速奖励），崎岖地形 gate→0.3（大幅减速，降低摔倒风险）。
- 保留 balance_penalty 和 air_stability_penalty 不变，维持姿态约束。
- 该变换为 **Level 2 add 新组件（但形式上通过修改 forward_progress 实现）**，填补信号缺口，且符合“乘法门控优于累加惩罚”的原则。
- 设计校准：gate 最小值 0.3 防止完全抑制前进；总惩罚负担不变（约 0.11/step），主信号平均约 0.15～0.28/step，惩罚/主信号比依然可控。

预期：agent 学会在前方地形崎岖时主动减速，从而减少摔倒，延长存活，累积更高前进回报。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    leg1_contact = obs[12]
    leg2_contact = obs[13]

    # ------------------------------------------------------------
    # 1. Terrain awareness gate (NEW - replaces raw forward_progress)
    #    Use lidar readings (obs[14:24]) to measure terrain roughness.
    # ------------------------------------------------------------
    lidar_vals = [obs[i] for i in range(14, 24)]
    n_lidar = len(lidar_vals)
    mean_lidar = sum(lidar_vals) / n_lidar
    variance = sum((v - mean_lidar) ** 2 for v in lidar_vals) / n_lidar
    roughness = variance ** 0.5

    # Map roughness to a gate: 1.0 (smooth) -> 0.3 (rough)
    roughness_threshold = 0.3
    # Clip roughness to threshold so gate saturates at 1.0
    roughness_clipped = min(roughness, roughness_threshold)
    gate = 1.0 - 0.7 * (roughness_clipped / roughness_threshold)  # [0.3, 1.0]

    forward_reward = horizontal_speed * gate

    # ------------------------------------------------------------
    # 2. Balance penalty (unchanged)
    # ------------------------------------------------------------
    angle_threshold = 0.4
    angvel_threshold = 1.0
    angle_excess = max(0.0, abs(hull_angle) - angle_threshold)
    angvel_excess = max(0.0, abs(hull_angvel) - angvel_threshold)
    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    # ------------------------------------------------------------
    # 3. Air-stability penalty (unchanged)
    # ------------------------------------------------------------
    both_feet_off = max(0.0, 1.0 - (leg1_contact + leg2_contact))
    air_penalty = -0.3 * both_feet_off
    vertical_fall_penalty = -1.0 * both_feet_off * max(0.0, -vertical_speed)
    air_stability_penalty = air_penalty + vertical_fall_penalty

    total_reward = forward_reward + balance_penalty + air_stability_penalty

    components = {
        'forward_reward': forward_reward,          # now gated
        'balance_penalty': balance_penalty,
        'air_stability_penalty': air_stability_penalty,
        'terrain_roughness': roughness,
        'terrain_gate': gate
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺失 — 激光雷达（10维地形高度）完全未被使用，agent 失去前方地形信息，无法解释为何在平坦地形外频繁摔倒。
- **behavior**: agent 在 105 步内快速摔倒（20/20 terminated），存活时间比上一轮暴跌 57%，前进奖励极低。
- **signal**: 缺少地形感知 → 高速撞上未预见障碍物导致摔倒；现有惩罚无法弥补前瞻缺口。
- **level**: Level 2（填补信号缺口，新增 terrain‑aware gate）
- **hypothesis**: 将前进奖励与前方地形粗糙度门控，迫使 agent 在崎岖地形减速，从而减少摔倒、延长存活、最终提高累积前进奖励。
- **risk**: 门控阈值若与地形特性不匹配（ roughness_threshold=0.3 偏小）可能导致 agent 过度减速而在平坦段停滞；需后续依据步态长度进行调整。

# 1. Search objective
- target_score: 300.000000
- current_score: -86.300185
- gap_to_target: 386.300185
- target_achievement_ratio: -28.767%

# 2. 上一轮奖励函数代码（该轮得分: -86.300185）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    leg1_contact = obs[12]
    leg2_contact = obs[13]

    # 1. Forward progress (unchanged)
    forward_progress = horizontal_speed

    # 2. Balance penalty (unchanged)
    angle_threshold = 0.4
    angvel_threshold = 1.0
    angle_excess = max(0.0, abs(hull_angle) - angle_threshold)
    angvel_excess = max(0.0, abs(hull_angvel) - angvel_threshold)
    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    # 3. Air-stability penalty (NEW)
    #    Punish having both feet off the ground, especially when falling downward.
    #    leg contact is 0/1, so sum 0 => both off, 1 => one on, 2 => both on.
    both_feet_off = max(0.0, 1.0 - (leg1_contact + leg2_contact))
    # Base penalty for any airborne frame (small, to allow natural brief flight)
    air_penalty = -0.3 * both_feet_off
    # Extra penalty when airborne and descending (hard landing / falling)
    vertical_fall_penalty = -1.0 * both_feet_off * max(0.0, -vertical_speed)
    air_stability_penalty = air_penalty + vertical_fall_penalty

    total_reward = forward_progress + balance_penalty + air_stability_penalty

    components = {
        'forward_progress': forward_progress,
        'balance_penalty': balance_penalty,
        'air_stability_penalty': air_stability_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-86.300185, len=105.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-89.688014, -84.316982]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 29.558881 | 70.5% | 70.6% | 100.0% |
| air_stability_penalty | -10.686779 | -25.5% | 25.5% | 68.0% |
| balance_penalty | -1.615357 | -3.9% | 3.9% | 2.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
```
