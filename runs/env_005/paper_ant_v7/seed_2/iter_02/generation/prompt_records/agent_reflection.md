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
- current_score: -102.079657
- gap_to_target: 2102.079657
- target_achievement_ratio: -5.104%

# 2. 上一轮奖励函数代码（该轮得分: -102.079657）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：前进速度（正向） ----
    forward_velocity = next_obs[13]  # body_x_velocity
    forward_reward = 2.0 * forward_velocity

    # ---- 稳定/健康约束：身体高度安全区间 ----
    body_height = next_obs[0]
    lower_safe = 0.3   # 终止边界 0.2 的 150%，留有缓冲
    upper_safe = 0.9   # 终止边界 1.0 的 90%
    height_penalty = (
        -5.0 * max(0.0, lower_safe - body_height) +
        -5.0 * max(0.0, body_height - upper_safe)
    )

    # ---- 稳定/健康约束：直立姿态 ----
    quat_x, quat_y = next_obs[2], next_obs[3]
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)  # 1 为完全直立
    upright_penalty = -1.0 * (1.0 - body_up_z)**2

    # ---- 辅助约束：侧向漂移抑制 ----
    lateral_velocity = next_obs[14]
    lateral_penalty = -0.5 * (lateral_velocity)**2

    # ---- 效率约束（极小权重）：动作能量代价 ----
    action_energy = sum(a**2 for a in action)
    energy_penalty = -0.01 * action_energy

    total_reward = forward_reward + height_penalty + upright_penalty + lateral_penalty + energy_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "height_health_penalty": height_penalty,
        "upright_orientation_penalty": upright_penalty,
        "lateral_drift_penalty": lateral_penalty,
        "action_energy_penalty": energy_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=-102.079657, len=218.100000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-883.698747, 18.669044]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 159.081901 | 36.6% | 47.4% | 59.4% |
| upright_orientation_penalty | -177.884621 | -40.9% | 40.9% | 58.7% |
| lateral_drift_penalty | -36.084192 | -8.3% | 8.3% | 58.9% |
| action_energy_penalty | -7.974121 | -1.8% | 1.8% | 100.0% |
| height_health_penalty | -6.991385 | -1.6% | 1.6% | 20.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个连续控制运动任务。控制一个 3D 四足机器人（8 个扭矩关节）在保证身体高度处于健康区间、保持直立姿态的前提下，以尽可能快的速度稳定前进。核心目标是稳定的前进运动（**前进速度最大化**），维持身体高度和直立姿态是避免提前终止的必要条件，但不是任务本身的核心优化目标。不允许依赖任何官方奖励项（info 被清空，official reward masked）。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: 推断为 float32（来自 continuous locomotion）
- obs[0] (body_z)：身体重心垂直高度。reward_usable: true（可构建高度健康奖励/惩罚）
- obs[1] (quat_w)：身体姿态四元数实部 w。reward_usable: true（用于计算直立程度）
- obs[2] (quat_x)：姿态四元数虚部 x。reward_usable: true
- obs[3] (quat_y)：姿态四元数虚部 y。reward_usable: true
- obs[4] (quat_z)：姿态四元数虚部 z。reward_usable: true
- obs[5] (joint_1_angle)：第一 hip 关节角度。reward_usable: true（可用来约束关节范围、平滑动作）
- obs[6] (joint_2_angle)：第一 ankle 关节角度。reward_usable: true
- obs[7] (joint_3_angle)：第二 hip 关节角度。reward_usable: true
- obs[8] (joint_4_angle)：第二 ankle 关节角度。reward_usable: true
- obs[9] (joint_5_angle)：第三 hip 关节角度。reward_usable: true
- obs[10] (joint_6_angle)：第三 ankle 关节角度。reward_usable: true
- obs[11] (joint_7_angle)：第四 hip 关节角度。reward_usable: true
- obs[12] (joint_8_angle)：第四 ankle 关节角度。reward_usable: true
- obs[13] (body_x_velocity)：身体在世界系 x 方向的前进速度。reward_usable: **true（核心前进信号）**
- obs[14] (body_y_velocity)：身体横向速度（世界 y）。reward_usable: true（可用于惩罚侧向漂移）
- obs[15] (body_z_velocity)：身体垂直速度。reward_usable: true（可用于惩罚剧烈上下颠簸）
- obs[16] (body_roll_velocity)：滚转角速度。reward_usable: true
- obs[17] (body_pitch_velocity)：俯仰角速度。reward_usable: true
- obs[18] (body_yaw_velocity)：偏航角速度。reward_usable: true
- obs[19] (joint_1_velocity)：第一 hip 关节角速度。reward_usable: true（用于平滑或能耗惩罚）
- obs[20] (joint_2_velocity)：第一 ankle 关节角速度。reward_usable: true
- obs[21] (joint_3_velocity)：第二 hip 关节角速度。reward_usable: true
- obs[22] (joint_4_velocity)：第二 ankle 关节角速度。reward_usable: true
- obs[23] (joint_5_velocity)：第三 hip 关节角速度。reward_usable: true
- obs[24] (joint_6_velocity)：第三 ankle 关节角速度。reward_usable: true
- obs[25] (joint_7_velocity)：第四 hip 关节角速度。reward_usable: true
- obs[26] (joint_8_velocity)：第四 ankle 关节角速度。reward_usable: true

## 4. 动作空间 action_space
- type: Box（连续）
- shape: (8,)
- 范围：[-1.0, 1.0] per joint（扭矩归一化值）
- action_dim 0 (hip_1_torque)：第一 hip 关节扭矩
- action_dim 1 (ankle_1_torque)：第一 ankle 关节扭矩
- action_dim 2 (hip_2_torque)：第二 hip 关节扭矩
- action_dim 3 (ankle_2_torque)：第二 ankle 关节扭矩
- action_dim 4 (hip_3_torque)：第三 hip 关节扭矩
- action_dim 5 (ankle_3_torque)：第三 ankle 关节扭矩
- action_dim 6 (hip_4_torque)：第四 hip 关节扭矩
- action_dim 7 (ankle_4_torque)：第四 ankle 关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **无明显成功终止**。episode 如果一直保持安全状态直到被截断（truncation）则可能被视为成功完成一次稳定的前进回合。
- failure-like termination: 身体高度低于 0.2（跌倒）或高于 1.0（过度起跳）；任何状态值变为 NaN 或 inf（数值崩溃）。
- ambiguous termination: 无。
- truncation: 达到环境预设的最大仿真步数（时间限制），此时 episode 直接结束，无特殊终止标志。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false（terminated 信号在 step 外部返回，但 reward 函数接口无法直接获取 terminated 标志）
- allowed_info_fields: **无**（info 字典被清空，接口声明禁止使用任何 info 字段）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等全部官方奖励或定位数据（明确禁止）

## 7. 可用于奖励函数的信号
- position: body_z（高度），关节角度（可通过与目标姿态的偏差设计奖励）
- velocity: body_x_velocity（前进速度，核心），body_y_velocity（侧向），body_z_velocity（垂直速度），各关节角速度
- orientation: body_up_z = 1 - 2*(quat_x² + quat_y²) 量化直立程度（0~1，1 为完全竖直）
- contact: 无直接接触力，本环境版本无接触信息
- action/engine: action 本身（扭矩可构成能量/平滑惩罚），action 变化量（需自行维护上次动作，但奖励函数无状态，故无法直接计算 delta；可以惩罚 action 的绝对大小）
- other: 关节角度偏离正常范围（如设定目标关节位置）可用作风格约束

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
| 1 | action_energy_penalty + forward_velocity_reward + height_health_penalty + lateral_drift_penalty + upright_orientation_penalty | -102.08 | -102.08 | 0.00 | 218.10 | action_energy_penalty=-0.042 forward_velocity_reward=0.745 height_health_penalty=-0.063 lateral_drift_penalty=-0.291 upright_orientation_penalty=-1.511 | new_best |

```
