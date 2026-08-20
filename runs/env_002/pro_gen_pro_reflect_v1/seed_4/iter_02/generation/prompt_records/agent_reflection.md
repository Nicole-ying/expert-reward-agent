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
- current_score: 212.370151
- gap_to_target: 87.629849
- target_achievement_ratio: 70.790%

# 2. 上一轮奖励函数代码（该轮得分: 212.370151）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- 核心观测提取 --------------------
    hull_angle     = float(obs[0])   # 身体倾角 (rad)
    hull_ang_vel   = float(obs[1])   # 身体角速度 (rad/s)
    horiz_vel      = float(obs[2])   # 前向速度 (m/s)
    vert_vel       = float(obs[3])   # 竖直速度 (m/s)

    # -------------------- 主进展奖励 (核心驱动力) --------------------
    # 只奖励正的前进速度，负值不惩罚也不奖励
    forward_reward = 1.0 * max(0.0, horiz_vel)

    # -------------------- 姿态与稳定性约束 --------------------
    # 倾角 hinge 惩罚：只在 |angle| > 0.5 rad 时生效，避免安全范围内全时惩罚
    angle_threshold = 0.5
    angle_penalty = -0.5 * max(0.0, abs(hull_angle) - angle_threshold)

    # 角速度二次惩罚：始终有轻量梯度，抑制剧烈摇摆
    angvel_penalty = -0.001 * (hull_ang_vel ** 2)

    # -------------------- 竖直速度抑制（防跳跃） --------------------
    # 步行中正常竖直速度波动 ≤0.5 m/s，超过部分轻罚，避免 agent 通过跳跃刷分
    vert_threshold = 0.5
    vert_penalty = -0.1 * max(0.0, abs(vert_vel) - vert_threshold)

    # -------------------- 总奖励 --------------------
    total_reward = forward_reward + angle_penalty + angvel_penalty + vert_penalty

    components = {
        "forward_reward": forward_reward,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "vert_penalty": vert_penalty,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=212.370151, len=632.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-30.337260, 318.220843]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 404.485881 | 99.8% | 99.8% | 99.5% |
| angle_penalty | -0.841930 | -0.2% | 0.2% | 1.3% |
| angvel_penalty | -0.000814 | -0.0% | 0.0% | 99.8% |
| vert_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
环境要求一个双腿身体在不平坦地形上尽可能快、尽可能远地向前行走，同时最小化能量消耗。主体必须协调两侧髋、膝关节力矩，产生稳定的双足步态。**摔倒将导致回合提前终止**，因此保持身体平衡是隐含的刚性前提。次要是节省能量、获得平滑的关节动作，但它们绝不能以牺牲前进和平衡为代价。不应混淆“生存”与“到达终点”：虽然走过整个地形也算终止，但核心奖励必须源于持续的向前位移，而不是一次性的终点信号。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（假定）
- obs[0]: hull_angle — 身体相对竖直方向的倾斜角度，reward_usable: true（防摔倒）
- obs[1]: hull_angular_velocity — 身体角速度，reward_usable: true（姿态稳定性）
- obs[2]: horizontal_velocity — 前进/后退线速度，reward_usable: true（**主前进信号**）
- obs[3]: vertical_velocity — 上下线速度，reward_usable: true（可辅助检测跳跃或不稳定）
- obs[4]: hip1_angle — 腿1髋关节角度，reward_usable: true（姿态约束）
- obs[5]: hip1_speed — 腿1髋关节角速度，reward_usable: true（关节平滑性）
- obs[6]: knee1_angle — 腿1膝关节角度，reward_usable: true（姿态约束）
- obs[7]: knee1_speed — 腿1膝关节角速度，reward_usable: true（关节平滑性）
- obs[8]: leg1_contact — 腿1着地标志，1.0=接触，0.0=无接触，reward_usable: true（步态节律/接触）
- obs[9]: hip2_angle — 腿2髋关节角度，reward_usable: true（对称/姿态）
- obs[10]: hip2_speed — 腿2髋关节角速度，reward_usable: true（平滑）
- obs[11]: knee2_angle — 腿2膝关节角度，reward_usable: true（姿态）
- obs[12]: knee2_speed — 腿2膝关节角速度，reward_usable: true（平滑）
- obs[13]: leg2_contact — 腿2着地标志，reward_usable: true（步态节律/接触）
- obs[14..23]: lidar_0..9 — 10 路激光测距值（前方地形高度轮廓），reward_usable: 不确定，通常不直接用于简单前进奖励，可预留为高级地形感知，但基础奖励设计中不建议使用，防止引入不必要复杂度。

所有观测均可用，但是否用于奖励需依据职责。

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 范围: 每维 [-1.0, 1.0]
- action_dim 0: hip_torque_leg1 — 腿1髋关节力矩
- action_dim 1: knee_torque_leg1 — 腿1膝关节力矩
- action_dim 2: hip_torque_leg2 — 腿2髋关节力矩
- action_dim 3: knee_torque_leg2 — 腿2膝关节力矩

说明：动作是连续力矩控制，无离散选项。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（走完全程）
- failure-like termination: body_fallen_over（身体摔倒）
- ambiguous termination: 无
- truncation: 未在描述中提及，假定有最大步数限制（通用环境），但不作为奖励信号依据。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无法直接获得 reached_end_of_terrain 标志）
- explicit_failure_flag_available: false（同样无法直接获得摔倒标志）
- allowed_info_fields: 无（当前 step 返回 info={}）
- forbidden_or_uncertain_info_fields: 所有的 info 字段均不存在，因此不能依赖任何外部成功/失败标记。奖励函数只能基于观测和动作构建生存和前进信号。

## 7. 可用于奖励函数的信号
- position: 无直接位置观测，但可通过积分水平速度推算（但由于基础，不宜复杂，建议仅用瞬时速度）
- velocity: obs[2] horizontal_velocity（主信号），obs[3] vertical_velocity（辅助），obs[1] hull_angular_velocity（姿态稳定）
- orientation: obs[0] hull_angle（倾斜惩罚）
- contact: obs[8] leg1_contact, obs[13] leg2_contact（步态节律/双脚支撑）
- action/engine: action 四个维度的幅值（能耗惩罚），action 的前后变化量（平滑惩罚）
- other: lidar 测距阵列（obs[14:23]），不建议用于基础奖励；关节角度/角速度可用于姿态正则化。

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
| 1 | angle_penalty + angvel_penalty + forward_reward + vert_penalty | 212.37 | 212.37 | 0.00 | 632.50 | angle_penalty=-0.004 angvel_penalty=-0.000 forward_reward=0.358 vert_penalty=0.000 | new_best |

```
