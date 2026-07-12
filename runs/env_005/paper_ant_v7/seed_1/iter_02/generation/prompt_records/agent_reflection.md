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
- target_score: 2000.000000
- current_score: 2.911636
- gap_to_target: 1997.088364
- target_achievement_ratio: 0.146%

# 2. 上一轮奖励函数代码（该轮得分: 2.911636）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation signals
    body_z = obs[0]                     # body height
    vx = obs[13]                        # forward velocity (world x)
    quat_x = obs[2]
    quat_y = obs[3]

    # ---- 1. Survival gate: height factor ----
    # height health: allow full reward when body_z is in [0.4, 0.8]
    # linear ramp from 0 at boundary (0.2 or 1.0) to 1 at safe zone
    low_safe  = max(0.0, min(1.0, (body_z - 0.2) / 0.2))   # distance above 0.2
    high_safe = max(0.0, min(1.0, (1.0 - body_z) / 0.2))   # distance below 1.0
    height_factor = min(low_safe, high_safe)                 # clamp by the more critical side

    # ---- 2. Survival gate: uprightness factor ----
    # compute body_up_z from quaternion: how close the body's z-axis is to world z
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, min(1.0, body_up_z))          # clip to [0,1], 1 = fully upright

    survival_gate = height_factor * upright_factor

    # ---- 3. Main progress component (gated) ----
    w_progress = 1.0
    forward_gated = w_progress * vx * survival_gate

    # ---- 4. Light energy efficiency term ----
    w_energy = 0.0005
    action_penalty = -w_energy * sum(a**2 for a in action)

    # Total reward
    total_reward = forward_gated + action_penalty
    components = {
        'forward_gated': forward_gated,
        'action_penalty': action_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=2.911636, len=694.350000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-638.628762, 321.872725]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_gated | 728.613833 | 97.8% | 99.8% | 79.9% |
| action_penalty | -1.445550 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一项连续控制运动任务：控制一个具有 8 个扭矩关节的四足机器人在 3D 环境中向前行进（走路或奔跑），同时必须保持身体高度处于健康范围内（0.2 m ~ 1.0 m）。主要目标是实现稳定的前进速度，而非仅仅保持站立；次要目标可包括运动平滑性、能量效率及减少侧向漂移等。任务不应与简单悬停或固定位置平衡混淆。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32 （推断，未明确指定但通常如此）
- 各维度含义与奖励可用性：

| index | 名称               | 含义                                      | reward_usable |
|-------|-------------------|-------------------------------------------|---------------|
| 0     | body_z            | 身体质心高度 (m)                           | true          |
| 1     | quat_w            | 方向四元数实部 w                           | true (用于计算直立度) |
| 2     | quat_x            | 四元数 x 分量                              | true (用于计算直立度) |
| 3     | quat_y            | 四元数 y 分量                              | true (用于计算直立度) |
| 4     | quat_z            | 四元数 z 分量                              | true (用于计算直立度) |
| 5     | joint_1_angle     | 前左髋关节角度                              | true          |
| 6     | joint_2_angle     | 前左踝关节角度                              | true          |
| 7     | joint_3_angle     | 前右髋关节角度                              | true          |
| 8     | joint_4_angle     | 前右踝关节角度                              | true          |
| 9     | joint_5_angle     | 后左髋关节角度                              | true          |
| 10    | joint_6_angle     | 后左踝关节角度                              | true          |
| 11    | joint_7_angle     | 后右髋关节角度                              | true          |
| 12    | joint_8_angle     | 后右踝关节角度                              | true          |
| 13    | body_x_velocity   | 世界坐标系下身体前进速度 (m/s)               | true (主指标) |
| 14    | body_y_velocity   | 世界坐标系下侧向速度 (m/s)                   | true          |
| 15    | body_z_velocity   | 垂直速度 (m/s)                              | true          |
| 16    | body_roll_velocity| 滚转角速度 (rad/s)                          | true          |
| 17    | body_pitch_velocity| 俯仰角速度 (rad/s)                         | true          |
| 18    | body_yaw_velocity | 偏航角速度 (rad/s)                          | true          |
| 19    | joint_1_velocity  | 关节1角速度                                | true          |
| 20    | joint_2_velocity  | 关节2角速度                                | true          |
| 21    | joint_3_velocity  | 关节3角速度                                | true          |
| 22    | joint_4_velocity  | 关节4角速度                                | true          |
| 23    | joint_5_velocity  | 关节5角速度                                | true          |
| 24    | joint_6_velocity  | 关节6角速度                                | true          |
| 25    | joint_7_velocity  | 关节7角速度                                | true          |
| 26    | joint_8_velocity  | 关节8角速度                                | true          |

注：观测中不包含全局 x/y 位置、接触力或任何地面反力信息。

## 4. 动作空间 action_space
- type: Box (continuous)
- shape: (8,)
- 范围：每个关节 [-1.0, 1.0]（扭矩归一化或扭矩值本身）
- 各维度含义：

| action dim | 名称             | 含义                         |
|-----------|------------------|------------------------------|
| 0         | hip_1_torque     | 前左髋关节扭矩               |
| 1         | ankle_1_torque   | 前左踝关节扭矩               |
| 2         | hip_2_torque     | 前右髋关节扭矩               |
| 3         | ankle_2_torque   | 前右踝关节扭矩               |
| 4         | hip_3_torque     | 后左髋关节扭矩               |
| 5         | ankle_3_torque   | 后左踝关节扭矩               |
| 6         | hip_4_torque     | 后右髋关节扭矩               |
| 7         | ankle_4_torque   | 后右踝关节扭矩               |

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确成功终止；任务没有设定“到达目标点”等成功条件。
- failure-like termination: 
  1. body_height_outside_healthy_range：身体高度超出 (0.2, 1.0)（过低视为跌倒/倒塌，过高视为异常弹跳或失控）。
  2. state_value_outside_finite_range：任何状态值为 NaN 或无穷。
- ambiguous termination: 无（上述均为失败类终止）
- truncation: 时间限制到达（time_limit_reached），通常为 max_episode_steps 截断，不应视为成功或失败，只能作为“存活但未完成特定目标”。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false （终止标志 `terminated` 可间接视为失败触发，但未在 info 中提供明确 label）
- allowed_info_fields: [] （空，不能从 info 获取任何信息）
- forbidden_or_uncertain_info_fields:
  - reward_forward
  - reward_ctrl
  - reward_contact
  - reward_survive
  - x_position
  - y_position
  - distance_from_origin
  这些字段明确被禁止用于奖励函数设计。

## 7. 可用于奖励函数的信号
这些信号均可从 obs/next_obs/action 获得：

- position:
  - 身体高度 `body_z` (obs[0])，可评估与理想高度的偏差。
  - 所有关节角度 obs[5:13]，可鼓励或惩罚特定姿态。
- velocity:
  - 前进速度 `body_x_velocity` (obs[13])，直接反映前进表现。
  - 侧向速度 `body_y_velocity` (obs[14])，可用于惩罚侧向漂移。
  - 垂直速度 `body_z_velocity` (obs[15])，可惩罚过度跳跃或不稳定。
  - 身体角速度 (roll/pitch/yaw) obs[16:19]，可惩罚剧烈摇晃。
  - 关节角速度 obs[19:27]，可惩罚关节振动或高能耗。
- orientation:
  - 由四元数 obs[1:5] 可计算 body_up_z = 1 - 2*(quat_x² + quat_y²)（取值范围约 -1 到 1，1 表示完全直立），可用于奖励直立姿态。
- action/engine:
  - 动作扭矩 `action` 的 L2 范数或绝对值和，用于鼓励小力矩（节能）。
  - 动作变化率（需要前后步动作，但 `compute_reward` 只有单步信息，无法直接获前一动作），因此动作平滑（变化惩罚）只能通过两帧动作差来近似，若函数不提供 `prev_action`，该信号不可用。通常环境要求无状态，故可能禁用动作变化惩罚。

- other:
  - 存活信号可用 `body_z` 是否在健康范围内（但范围外的已终止不会调用 reward 函数，所以密集奖励可用到 “接近边界” 的惩罚）。

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
| 1 | action_penalty + forward_gated | 2.91 | 2.91 | 0.00 | 694.35 | action_penalty=-0.002 forward_gated=0.425 | new_best |

```
