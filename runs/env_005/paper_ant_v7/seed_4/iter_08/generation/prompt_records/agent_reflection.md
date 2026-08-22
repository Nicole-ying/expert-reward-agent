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
# ⚠️ REBUILD MODE
系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。
参考 #6 完整公式算子库选新的主信号框架，基于 #3 累积记录避开已失败的路径。
不要受上一轮代码结构约束。


# 1. Search objective
- target_score: 2000.000000
- current_score: -282.035627
- gap_to_target: 2282.035627
- target_achievement_ratio: -14.102%

# 2. 上一轮奖励函数代码（该轮得分: -282.035627）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract state from next_obs ----------
    body_z = next_obs[0]                     # body height
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]                         # quaternion (not used directly)
    body_x_vel = next_obs[13]                # forward velocity
    body_y_vel = next_obs[14]                # lateral velocity
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]                   # angular velocities

    # ---------- 1. soft health gate: upright factor ----------
    body_up_z = 1.0 - 2.0 * (qx*qx + qy*qy)
    tilt_error = 1.0 - body_up_z
    lambda_up = 3.0
    upright_gate = 2.718281828 ** (-lambda_up * tilt_error)   # ~1 when upright, ~0 when tilted

    # main progress signal: forward velocity, gated by upright
    gated_forward = body_x_vel * upright_gate

    # ---------- 2. height hinge penalty ----------
    lower_safe = 0.30
    upper_safe = 0.90
    height_penalty = max(0.0, lower_safe - body_z) + max(0.0, body_z - upper_safe)
    height_penalty_weighted = -10.0 * height_penalty

    # ---------- 3. lateral velocity penalty ----------
    lateral_penalty = -0.5 * (body_y_vel ** 2)

    # ---------- 4. angular velocity penalty (bounded) ----------
    sum_sq = roll_vel**2 + pitch_vel**2 + yaw_vel**2
    ang_penalty = -0.03 * sum_sq / (1.0 + sum_sq / 100.0)

    # ---------- combine ----------
    w_forward = 2.0
    total_reward = (w_forward * gated_forward
                    + height_penalty_weighted
                    + lateral_penalty
                    + ang_penalty)

    components = {
        "gated_forward": w_forward * gated_forward,
        "height_hinge_penalty": height_penalty_weighted,
        "lateral_penalty": lateral_penalty,
        "angular_velocity_penalty": ang_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 647.80 | -544.72 | ✅ |
| 2 | 添加动作惩罚将抑制高扭矩操作，引导学习更稳定、节能的步态，预期可降低终止率并提高前进效率 | 添加动作惩罚将抑制高扭矩操作，引导学习更稳定、节能的步态，预期可降低终止率并提高前进效率 | 349.35 | -49.90 | ✅ |
| 3 | 将高度约束改为 gate 将迫使 agent 在高度不良时接受近乎零的前进奖励，打破「加速→失衡→死亡」的循环。添... | 将高度约束改为 gate 将迫使 agent 在高度不良时接受近乎零的前进奖励，打破「加速→失衡→死亡」的循环。添... | 16.35 | -9.56 | ✅ |
| 4 | 减小旋转和侧向惩罚系数使每步净奖励为正，agent 存活时间变长后能获取更多前进奖励，从而自发降低角速度。 | 减小旋转和侧向惩罚系数使每步净奖励为正，agent 存活时间变长后能获取更多前进奖励，从而自发降低角速度。 | 797.25 | -779.13 | ❌ |
| 5 | 将前进奖励系数放大 10 倍后，高速行走的回报将显著超过苟活策略，agent 会被迫学习更快、更稳定的步态，从而大... | 将前进奖励系数放大 10 倍后，高速行走的回报将显著超过苟活策略，agent 会被迫学习更快、更稳定的步态，从而大... | 493.45 | -337.63 | ❓ |
| 6 | 骨架变化: angular_velocity_penalty + gated_forward + height_ | — | 577.30 | -184.03 | ❌ |
| 7 | 骨架变化: angular_velocity_penalty + gated_forward + height_ | — | 663.90 | -282.04 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=-282.035627, len=663.900000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-1034.263527, -10.131371]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 1421.015355 | 86.3% | 88.0% | 96.6% |
| lateral_penalty | -99.651550 | -6.1% | 6.1% | 93.9% |
| angular_velocity_penalty | -65.178205 | -4.0% | 4.0% | 94.0% |
| height_hinge_penalty | -32.046613 | -1.9% | 1.9% | 12.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个连续控制运动任务，控制一台具有四条腿、八个扭矩驱动关节的 3D 四足机器人。核心任务是让它持续、稳定地向前行走或奔跑，同时保持身体竖直。  
次要目标包括：抑制侧向漂移、降低身体角速度、减小关节动作能耗。机器人身体高度必须维持在健康范围内（0.2～1.0），否则立刻终止；同时任何状态量若变为 `NaN` 或 `inf` 也会终止。  
不应被混淆的目标：这里没有预定义的目标位置，不能把任务简单地当成“站立平衡”任务，因为必须产生稳定的前向位移才能持续获得奖励。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: 连续浮点（推测 float32）
- 各维度含义（索引 0～26）：
  - obs[0]: `body_z`，身体质心在世界坐标系中的高度，可用于检查健康范围，reward_usable: true
  - obs[1]: `quat_w`，身体姿态四元数的实部，reward_usable: true（用于计算直立程度）
  - obs[2]: `quat_x`，四元数第一虚部，reward_usable: true
  - obs[3]: `quat_y`，四元数第二虚部，reward_usable: true
  - obs[4]: `quat_z`，四元数第三虚部，reward_usable: true
  - obs[5]: `joint_1_angle`（第一个髋关节角度），reward_usable: true（可选用于姿态约束）
  - obs[6]: `joint_2_angle`（第一个踝关节角度），reward_usable: true
  - obs[7]: `joint_3_angle`（第二个髋关节角度），reward_usable: true
  - obs[8]: `joint_4_angle`（第二个踝关节角度），reward_usable: true
  - obs[9]: `joint_5_angle`（第三个髋关节角度），reward_usable: true
  - obs[10]: `joint_6_angle`（第三个踝关节角度），reward_usable: true
  - obs[11]: `joint_7_angle`（第四个髋关节角度），reward_usable: true
  - obs[12]: `joint_8_angle`（第四个踝关节角度），reward_usable: true
  - obs[13]: `body_x_velocity`，身体在世界 x 方向的线速度（前进速度），核心信号，reward_usable: true
  - obs[14]: `body_y_velocity`，身体在世界 y 方向（横向）的线速度，reward_usable: true（可惩罚侧向漂移）
  - obs[15]: `body_z_velocity`，身体在竖直方向的线速度，reward_usable: true（可选）
  - obs[16]: `body_roll_velocity`，滚转角速度，reward_usable: true
  - obs[17]: `body_pitch_velocity`，俯仰角速度，reward_usable: true
  - obs[18]: `body_yaw_velocity`，偏航角速度，reward_usable: true
  - obs[19]: `joint_1_velocity`，第一个髋关节角速度，reward_usable: true（可选用于平滑或能耗）
  - obs[20]: `joint_2_velocity`，第一个踝关节角速度，reward_usable: true
  - obs[21]: `joint_3_velocity`，第二个髋关节角速度，reward_usable: true
  - obs[22]: `joint_4_velocity`，第二个踝关节角速度，reward_usable: true
  - obs[23]: `joint_5_velocity`，第三个髋关节角速度，reward_usable: true
  - obs[24]: `joint_6_velocity`，第三个踝关节角速度，reward_usable: true
  - obs[25]: `joint_7_velocity`，第四个髋关节角速度，reward_usable: true
  - obs[26]: `joint_8_velocity`，第四个踝关节角速度，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- 连续动作，每个维度范围 [-1.0, 1.0]（归一化扭矩指令）
- 各维度含义：
  - action[0]: `hip_1_torque`（第 1 条腿髋关节扭矩）
  - action[1]: `ankle_1_torque`（第 1 条腿踝关节扭矩）
  - action[2]: `hip_2_torque`
  - action[3]: `ankle_2_torque`
  - action[4]: `hip_3_torque`
  - action[5]: `ankle_3_torque`
  - action[6]: `hip_4_torque`
  - action[7]: `ankle_4_torque`

## 5. step 与终止条件分析
### 5.1 终止模式
- **失败型终止（failure-like termination）**：
  - 身体高度超出健康范围：`body_z ≤ 0.2` 或 `body_z ≥ 1.0`（collapse 或 jump/launch）。
  - 任意状态量变为 `NaN` 或 `inf`（在模拟中通常由极端碰撞或数值发散导致）。
- **截断型终止（truncation）**：
  - 达到时间步上限 `time_limit_reached`（超出最大步数），属于正常截断，不应直接视为成功或失败。

- **成功标志**：不存在显式的 `info["success"]` 字段。
- **模糊终止**：无。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均被禁止，包括 `reward_forward`、`reward_ctrl`、`reward_contact`、`reward_survive`、`x_position`、`y_position`、`distance_from_origin` 等官方内部项。

注意：任务没有提供任何显式的成功/失败标志，奖励函数设计必须完全基于观察和动作信号，不能依赖 `info` 中的任何信息。

## 7. 可用于奖励函数的信号
- **位置与姿态**：
  - `body_z`（obs[0]）：用于高度存活惩罚。
  - 身体姿态四元数（obs[1:5]）：通过 `body_up_z = 1 - 2*(quat_x^2 + quat_y^2)` 计算身体 z 轴在世界坐标系中的投影，1.0 表示完全直立，0 表示水平。
  - 关节角度（obs[5:13]）：可用于姿态正则（但默认理想姿态未知，慎用）。

- **速度**：
  - 前向速度 `body_x_velocity`（obs[13]）：核心前进奖励信号。
  - 侧向速度 `body_y_velocity`（obs[14]）：横向漂移惩罚信号。
  - 垂直速度 `body_z_velocity`（obs[15]）：可选用于稳定性（例如惩罚快速起落）。
  - 角速度 `body_roll/pitch/yaw_velocity`（obs[16:19]）：旋转惩罚信号。
  - 关节角速度（obs[19:27]）：可用于动作平滑或能耗建模。

- **接触/力**：无。环境未提供接触力或足端接触布尔量。

- **动作/引擎**：
  - 动作向量 `action`：可直接惩罚扭矩大小（`torque_penalty = sum(action^2)`）或动作变化（若允许使用 `prev_action`，但本接口为单步无状态，无法直接获取上一步动作，因此只能使用当前动作幅度惩罚）。

- **其他**：`next_obs` 可对上述信号计算差分或在下个状态施加条件（例如若使用 next_obs 的身体高度，但可能导致奖励归属偏移，需谨慎）。

# 6. Formula Operator Library（完整版，用于 Level 3 重建）
# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |



# 7. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_upright_product + height_survival_penalty | -544.72 | -544.72 | 0.00 | 647.80 | forward_upright_product=0.253 height_survival_penalty=-0.023 | new_best |
| 2 | forward_upright_product + height_survival_penalty + torque_penalty | -49.90 | -49.90 | 0.00 | 349.35 | forward_upright_product=0.447 height_survival_penalty=-0.017 torque_penalty=-0.016 | new_best |
| 3 | forward_upright_gated + height_gate_value + lateral_penalty + rotation_penalty + torque_penalty | -9.56 | -9.56 | 0.00 | 16.35 | forward_upright_gated=0.125 height_gate_value=0.450 lateral_penalty=-0.105 rotation_penalty=-0.260 torque_penalty=-0.018 | new_best |
| 4 | forward_upright_gated + height_gate_value + lateral_penalty + rotation_penalty + torque_penalty | -779.13 | -9.56 | -769.57 | 797.25 | forward_upright_gated=0.154 height_gate_value=0.498 lateral_penalty=-0.023 rotation_penalty=-0.068 torque_penalty=-0.018 | no_meaningful_improvement |
| 5 | forward_upright_gated + height_gate_value + lateral_penalty + rotation_penalty + torque_penalty | -337.63 | -9.56 | -328.07 | 493.45 | forward_upright_gated=1.178 height_gate_value=0.395 lateral_penalty=-0.027 rotation_penalty=-0.096 torque_penalty=-0.017 | unsolved_stagnation_fresh_restart |
| 6 | angular_velocity_penalty + gated_forward + height_hinge_penalty + lateral_penalty | -184.03 | -9.56 | -174.48 | 577.30 | angular_velocity_penalty=-0.128 gated_forward=0.751 height_hinge_penalty=-0.151 lateral_penalty=-0.177 | no_meaningful_improvement |
| 7 | angular_velocity_penalty + gated_forward + height_hinge_penalty + lateral_penalty | -282.04 | -9.56 | -272.48 | 663.90 | angular_velocity_penalty=-0.114 gated_forward=0.600 height_hinge_penalty=-0.185 lateral_penalty=-0.143 | no_meaningful_improvement |

```
