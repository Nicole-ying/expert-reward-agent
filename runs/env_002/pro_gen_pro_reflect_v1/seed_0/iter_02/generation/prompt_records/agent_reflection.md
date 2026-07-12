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
# Duplicate reward retry
The previous generation duplicated iter 1 (runs/env_002/pro_gen_pro_reflect_v1/seed_0/iter_01/generation/reward_v1.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]

    # ----- 1. Forward progress reward -----
    # Encourage positive horizontal velocity only (discourage backward motion implicitly via other penalties)
    forward_speed_reward = 2.0 * max(0.0, horizontal_velocity)

    # ----- 2. Upright stability penalty -----
    # Penalize deviation from upright orientation and rapid angular motion
    angle_penalty = 1.0 * (hull_angle ** 2)
    angvel_penalty = 0.1 * (hull_angular_velocity ** 2)
    upright_penalty = -(angle_penalty + angvel_penalty)

    # ----- 3. Energy efficiency cost -----
    # Light penalty on joint torque magnitudes to discourage wasteful actions
    torque_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    action_cost = -0.01 * torque_sq_sum

    total_reward = forward_speed_reward + upright_penalty + action_cost
    components = {
        "forward_speed_reward": forward_speed_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 1. Search objective
- target_score: 300.000000
- current_score: 313.860015
- gap_to_target: -13.860015
- target_achievement_ratio: 104.620%

# 2. 上一轮奖励函数代码（该轮得分: 313.860015）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]

    # ----- 1. Forward progress reward -----
    # Encourage positive horizontal velocity only (discourage backward motion implicitly via other penalties)
    forward_speed_reward = 2.0 * max(0.0, horizontal_velocity)

    # ----- 2. Upright stability penalty -----
    # Penalize deviation from upright orientation and rapid angular motion
    angle_penalty = 1.0 * (hull_angle ** 2)
    angvel_penalty = 0.1 * (hull_angular_velocity ** 2)
    upright_penalty = -(angle_penalty + angvel_penalty)

    # ----- 3. Energy efficiency cost -----
    # Light penalty on joint torque magnitudes to discourage wasteful actions
    torque_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    action_cost = -0.01 * torque_sq_sum

    total_reward = forward_speed_reward + upright_penalty + action_cost
    components = {
        "forward_speed_reward": forward_speed_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=313.860015, len=811.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[312.709792, 314.658906]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_speed_reward | 1007.176566 | 98.5% | 98.5% | 99.2% |
| action_cost | -13.487510 | -1.3% | 1.3% | 100.0% |
| upright_penalty | -2.225808 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个平面双足行走控制问题。双足身体必须在不平坦地形上尽可能远、尽可能快地向前移动，同时希望尽量减少关节能耗。核心目标是生成稳定前进的步态并安全走完全程；摔倒会立即终止回合，视为失败。到达地形尽头也终止回合，可理解为成功完成。能耗最小化是附属目标，不应干扰前进和平衡主任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 （推断）
- obs[0]: hull_angle，主躯干相对直立方向的倾角，越过大可能导致摔倒，可用于奖励保持直立。
  reward_usable: true
- obs[1]: hull_angular_velocity，躯干角速度，可用于抑制剧烈摇晃。
  reward_usable: true
- obs[2]: horizontal_velocity，前向/后向线速度（前进为正），核心前进进度信号。
  reward_usable: true
- obs[3]: vertical_velocity，垂直速度，可能反映弹跳或不平整接触，可用来惩罚不必要弹跳。
  reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度。
  reward_usable: true（可结合周期性步态约束，但非必需）
- obs[5]: hip1_speed，腿1髋关节角速度。
  reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度。
  reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度。
  reward_usable: true
- obs[8]: leg1_contact，腿1是否触地（1.0=接触，0.0=未接触），可用于步态模式分析与奖励。
  reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度。
  reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度。
  reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度。
  reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度。
  reward_usable: true
- obs[13]: leg2_contact，腿2触地标志。
  reward_usable: true
- obs[14]~obs[23]: lidar_0..lidar_9，10个激光雷达距离测量值，描述前方地形高度剖面。
  reward_usable: true（可用于提前准备地形适应，但对前进奖励不易直接使用）

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- dtype: float32
- 范围: 每个动作分量均在 [-1.0, 1.0] 之间
- action[0]: hip_torque_leg1，施加于腿1髋关节的扭矩
- action[1]: knee_torque_leg1，施加于腿1膝关节的扭矩
- action[2]: hip_torque_leg2，施加于腿2髋关节的扭矩
- action[3]: knee_torque_leg2，施加于腿2膝关节的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain，走到地形末端，视为成功完成全程。
- failure-like termination: body_fallen_over，身体摔倒，任务失败。
- ambiguous termination: 无。
- truncation: 未提及超时，但实际环境中可能发生，环境规范中为 False 截断，prompt 未给出具体限制。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空字典，没有显式成功标记）
- explicit_failure_flag_available: false（info 为空，没有显式失败标记；但可以从终止条件推断，终止标志本身并未传入奖励函数）
- allowed_info_fields: 空字典 {}，无可用的额外信息。
- forbidden_or_uncertain_info_fields: 任何假设的 info 字段如 "success", "failure", "termination_reason" 均未出现，禁止使用。

## 7. 可用于奖励函数的信号
- position: 无绝对位置测量，只有速度。
- velocity: horizontal_velocity (obs[2])、vertical_velocity (obs[3])、hull_angular_velocity (obs[1]) 以及各个关节角速度（obs[5,7,10,12]）。
- orientation: hull_angle (obs[0]) 可作为倾角惩罚。
- contact: leg1_contact (obs[8])、leg2_contact (obs[13])，可用于步态切换或支撑相惩罚。
- action/engine: 动作扭矩 action（4 维向量），可用于能量或动作平滑惩罚。
- other: LIDAR 数组 (obs[14:23]) 提供地形距离，可间接用于不确定性惩罚或鼓励适应。

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
| 1 | action_cost + forward_speed_reward + upright_penalty | 313.86 | 313.86 | 0.00 | 811.90 | action_cost=-0.019 forward_speed_reward=0.612 upright_penalty=-0.025 | target_solved_new_best |

```
