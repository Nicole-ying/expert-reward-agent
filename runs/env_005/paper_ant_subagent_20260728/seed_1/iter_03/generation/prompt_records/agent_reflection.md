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
# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: 缺少准确函数签名; 函数体中使用了 original_reward; 代码无法解析 AST: unexpected EOF while parsing (<unknown>, line 1) (record: runs/env_005/paper_ant_subagent_20260728/seed_1/iter_03/generation/validations/reward_v3.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由

当前 agent 的得分从 -11.32 暴跌至 -240.56，尽管存活长度显著提升至 244 步，这强烈暗示：
1. 乘积门控（`health_gate` × `forward_progress`）导致大量时间步 forward 信号为 0，agent 缺乏有效的引导；
2. 同时环境中很可能存在隐含的 per‑step 动作成本（动作维度多且无动作惩罚），存活越久累积的负代价越大，净收益崩盘。

本轮在**同一骨架内**实施一个 Level 2 结构变换：**将乘积门控改造成插值形式的“基础奖励 + 健康加成”**，使 forward 信号在所有状态下都保持非零梯度，从根本上解决乘积塌缩问题。具体变化：

- 移除 `forward_progress = max(0, body_x_velocity)` 的非负裁剪，改为直接使用原始前进速度 `body_x_velocity`（允许负值惩罚倒退）。
- 将 reward_forward 由 `forward_progress × health_gate` 改为 `forward_velocity × (0.3 + 0.7 × health_gate)`。  
  当健康状态完美时，增益为 1.0；当健康门控完全坍塌（gate=0）时，仍保留 0.3 的基础信号，保证每步都能感知“应该向前”的梯度。
- 系数保持 1.0，保证预期步均主信号量级与前轮大致可比，不引入额外尺度风险。

其他组件（upright_penalty, z_velocity_penalty）暂时不动，留待后续轮次根据反馈微调。

```python
def compute_reward(obs, action, next_obs, original_reward,

# 1. Search objective
- target_score: 2000.000000
- current_score: -240.556906
- gap_to_target: 2240.556906
- target_achievement_ratio: -12.028%

# 2. 上一轮奖励函数代码（该轮得分: -240.556906）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_w = obs[1]
    quat_x = obs[2]
    quat_y = obs[3]
    quat_z = obs[4]
    body_x_velocity = obs[13]
    body_z_velocity = obs[15]

    # ---------- Derived signals ----------
    # Uprightness: body_up_z = 1.0 - 2*(qx^2 + qy^2), range ~[-1, 1]
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Health gate factors ----------
    # Height factor: healthy in (0.25, 0.95), smooth decay outside
    # Below 0.25: linear from 1.0 at z=0.25 down to 0.0 at z=0.15
    height_low_danger = max(0.0, (0.25 - body_z) / 0.10)  # 0.0 when z>=0.25, 1.0 when z<=0.15
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    # Above 0.95: linear from 1.0 at z=0.95 down to 0.0 at z=1.05
    height_high_danger = max(0.0, (body_z - 0.95) / 0.10)
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # Upright factor: healthy when body_up_z > 0.6, linear decay below
    upright_factor = min(1.0, max(0.0, (body_up_z - 0.2) / 0.4))
    # At body_up_z=0.6: (0.6-0.2)/0.4 = 1.0  → gate=1.0
    # At body_up_z=0.2: (0.2-0.2)/0.4 = 0.0  → gate=0.0
    # Smooth gradient throughout

    # Combined health gate: must satisfy both to earn full forward reward
    health_gate = min(height_factor, upright_factor)

    # ---------- Reward components ----------
    # 1) Forward progress gated by health
    forward_progress = max(0.0, float(body_x_velocity))
    reward_forward = 1.0 * forward_progress * health_gate

    # 2) Uprightness hinge penalty (only active when severely tilted)
    # body_up_z < 0.6: penalty grows linearly as tilt increases
    upright_error = max(0.0, 0.6 - body_up_z)
    reward_upright_penalty = -0.5 * upright_error

    # 3) Vertical velocity penalty (discourage bouncing/launching)
    # Soft hinge: penalize |vz| > 1.5
    vz_magnitude = abs(float(body_z_velocity))
    vz_excess = max(0.0, vz_magnitude - 1.5)
    reward_z_velocity_penalty = -0.2 * vz_excess

    # ---------- Total reward ----------
    total_reward = reward_forward + reward_upright_penalty + reward_z_velocity_penalty

    components = {
        "forward_gated": reward_forward,
        "health_gate": health_gate,
        "upright_penalty": reward_upright_penalty,
        "z_velocity_penalty": reward_z_velocity_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-240.556906, len=244.400000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-1133.041330, -2.945546]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| health_gate | 214.880584 | 65.8% | 65.8% | 88.1% |
| forward_gated | 88.621857 | 27.1% | 27.1% | 52.5% |
| upright_penalty | -22.195416 | -6.8% | 6.8% | 12.0% |
| z_velocity_penalty | -0.731142 | -0.2% | 0.2% | 3.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个连续控制四足机器人运动的环境。主要目标是使四足机器人持续、稳定地向前行走或奔跑，同时保持身体在健康高度范围内（0.2 ~ 1.0），并维持身体基本直立。机器人不应仅仅满足于不倒，而是需要产生有效的向前位移。次要目标包括抑制横向滑移、垂直弹跳、过大的关节力矩或速度，以提升运动的平稳性和能效。没有预设的目标位置，成功由存活至时间截断并持续向前移动来体现，不存在显式的“到达终点”成功标志。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32（推断）
- each index 的含义与 reward 可用性：

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | body_z | 身体质心垂直位置（高度） | true |
| 1 | quat_w | 身体姿态四元数实部 | true（需组合） |
| 2 | quat_x | 身体姿态四元数 x | true |
| 3 | quat_y | 身体姿态四元数 y | true |
| 4 | quat_z | 身体姿态四元数 z | true |
| 5 | joint_1_angle | 第1髋关节角度 | true |
| 6 | joint_2_angle | 第1踝关节角度 | true |
| 7 | joint_3_angle | 第2髋关节角度 | true |
| 8 | joint_4_angle | 第2踝关节角度 | true |
| 9 | joint_5_angle | 第3髋关节角度 | true |
| 10 | joint_6_angle | 第3踝关节角度 | true |
| 11 | joint_7_angle | 第4髋关节角度 | true |
| 12 | joint_8_angle | 第4踝关节角度 | true |
| 13 | body_x_velocity | 身体在世界系 x 轴的线速度（前进速度） | true |
| 14 | body_y_velocity | 身体在世界系 y 轴的线速度（横向速度） | true |
| 15 | body_z_velocity | 身体在 z 轴的速度（垂直速度） | true |
| 16 | body_roll_velocity | 身体滚转角速度 | true |
| 17 | body_pitch_velocity | 身体俯仰角速度 | true |
| 18 | body_yaw_velocity | 身体偏航角速度 | true |
| 19 | joint_1_velocity | 第1髋关节角速度 | true |
| 20 | joint_2_velocity | 第1踝关节角速度 | true |
| 21 | joint_3_velocity | 第2髋关节角速度 | true |
| 22 | joint_4_velocity | 第2踝关节角速度 | true |
| 23 | joint_5_velocity | 第3髋关节角速度 | true |
| 24 | joint_6_velocity | 第3踝关节角速度 | true |
| 25 | joint_7_velocity | 第4髋关节角速度 | true |
| 26 | joint_8_velocity | 第4踝关节角速度 | true |

注：所有观测均可用于奖励，没有禁止字段。四元数可转换为直立度量：`body_up_z = 1 - 2*(quat_x² + quat_y²)`，取值范围大约 `[-1,1]`，`1.0` 表示完美直立。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- continuous: true
- bounds: [-1.0, 1.0] 每维
- 动作维度含义：

| dim | name | meaning |
|-----|------|---------|
| 0 | hip_1_torque | 第1髋关节力矩 |
| 1 | ankle_1_torque | 第1踝关节力矩 |
| 2 | hip_2_torque | 第2髋关节力矩 |
| 3 | ankle_2_torque | 第2踝关节力矩 |
| 4 | hip_3_torque | 第3髋关节力矩 |
| 5 | ankle_3_torque | 第3踝关节力矩 |
| 6 | hip_4_torque | 第4髋关节力矩 |
| 7 | ankle_4_torque | 第4踝关节力矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无显式成功条件；时间截断（`truncated = time_limit_reached`）且未提前失败时可视为一种存活成功，但从终止信号角度不具有明确成功标志。
- **failure-like termination**: 身体高度超出 (0.2, 1.0) 范围（即跌落或弹起过高）；任意状态值变为 NaN 或 infinity。
- **ambiguous termination**: 无。
- **truncation**: 达到最大时间步，truncated 为 True，此时 episode 正常结束，不应视为失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields:  
  `reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive`, `x_position`, `y_position`, `distance_from_origin` 等均被明确禁止。  
  说明：虽然原始环境可能提供这些字段，但当前实验环境下已屏蔽，奖励函数中不得使用。

## 7. 可用于奖励函数的信号
- position: 
  - 身体高度 `body_z` (obs[0])
  - 关节角度 (obs[5:13])
- velocity:
  - 前进速度 `body_x_velocity` (obs[13])
  - 横向速度 `body_y_velocity` (obs[14])
  - 垂直速度 `body_z_velocity` (obs[15])
  - 关节速度 (obs[19:27])
- orientation:
  - 四元数 (obs[1:5])，可计算出直立程度 `body_up_z`
  - 角速度 roll/pitch/yaw (obs[16:18])，可额外作用于稳定性
- action/engine:
  - 动作向量 `action` (8,)，可用于惩罚大扭矩、动作变化等
- other:
  - 无。
```
