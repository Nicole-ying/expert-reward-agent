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
- current_score: -57.471340
- gap_to_target: 357.471340
- target_achievement_ratio: -19.157%

# 2. 上一轮奖励函数代码（该轮得分: -57.471340）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward for rough terrain:
    - Primary: forward velocity reward with soft health gate based on hull stability
    - Constraint: postural hinge penalty for extreme tilt

    role-based component budget: 2 components (forward_progress + postural_stability)
    """
    # ==================== Extract signals ====================
    # Current hull state
    hull_angle = obs[0]
    hull_angvel = abs(obs[1])

    # Next hull state
    next_hull_angle = next_obs[0]
    next_hull_angvel = abs(next_obs[1])

    # Forward velocity (next step)
    horizontal_speed = next_obs[2]

    # ==================== Constants ====================
    # Hull tilt safety thresholds (radians, empirically near falling boundary)
    TILT_CRITICAL = 0.6      # near-falling severe tilt
    TILT_WARNING_START = 0.25  # begin gentle attenuation
    TILT_WARNING_MARGIN = 0.35  # attenuation window width

    # Hinge penalty thresholds
    HINGE_THRESHOLD = 0.35   # start penalizing tilt above this
    HINGE_SCALE = 1.0

    # Weights (balanced for per-step magnitude comparable)
    FORWARD_WEIGHT = 2.0
    POSTURE_HINGE_WEIGHT = 0.5

    # ==================== Component A: Forward progress with soft health gate ====================
    # Gate factor: linear attenuation from 1.0 (safe) to 0.0 (critical)
    # Uses next_hull_angle (immediate future stability) to gate reward
    abs_tilt = abs(next_hull_angle)
    if abs_tilt <= TILT_WARNING_START:
        gate = 1.0
    elif abs_tilt >= TILT_CRITICAL:
        gate = 0.0
    else:
        gate = (TILT_CRITICAL - abs_tilt) / TILT_WARNING_MARGIN

    # Forward reward: convex to encourage speed, not just minimal forward motion
    forward_reward = FORWARD_WEIGHT * horizontal_speed ** 2
    gated_forward = gate * forward_reward

    # ==================== Component B: Postural hinge penalty ====================
    # Only penalize when tilt exceeds safe threshold
    # Penalizes both current and next tilt, and angular velocity
    current_excess = max(0.0, abs(hull_angle) - HINGE_THRESHOLD)
    next_excess = max(0.0, abs_tilt - HINGE_THRESHOLD)

    # Average excess over the step (current + next) with velocity penalty
    tilt_penalty = HINGE_SCALE * (current_excess + next_excess) * 0.5
    angvel_penalty = 0.3 * next_hull_angvel  # angular velocity contributes to instability

    posture_penalty = -POSTURE_HINGE_WEIGHT * (tilt_penalty + angvel_penalty)

    # ==================== Total reward ====================
    total_reward = gated_forward + posture_penalty

    # ==================== Components dict ====================
    components = {
        'gated_forward_speed': gated_forward,
        'posture_hinge_penalty': posture_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-57.471340, len=166.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-88.840113, -40.010828]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward_speed | 80.118276 | 97.7% | 97.7% | 97.7% |
| posture_hinge_penalty | -1.879948 | -2.3% | 2.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 5/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境中的双足机器人需要在不规则的粗糙地形上尽可能向前行进，同时保持能量效率。地形包含阶梯状、树桩、坑洞等变化，因此机体必须利用前方 10 个 LIDAR 测距信号来预判地形，动态调整步态。主要目标是：稳定行走、远离摔倒、尽量走远；次要目标是：最小化不必要的关节扭矩（能耗）。该任务的核心是崎岖地形上的持续运动控制，而非单纯到达指定坐标点。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float32（默认推断）  
- obs 各维含义：

| index | 名称                     | 含义                                   | reward_usable |
|-------|--------------------------|----------------------------------------|---------------|
| 0     | hull_angle               | 身体基座倾角                           | true          |
| 1     | hull_angular_velocity    | 身体基座角速度                         | true          |
| 2     | horizontal_speed         | 质心水平速度                           | true          |
| 3     | vertical_speed           | 质心垂直速度                           | true          |
| 4     | joint_0_angle (hip_1)    | 髋关节 1 角度                          | true          |
| 5     | joint_0_speed (hip_1)    | 髋关节 1 角速度                        | true          |
| 6     | joint_1_angle (knee_1)   | 膝关节 1 角度                          | true          |
| 7     | joint_1_speed (knee_1)   | 膝关节 1 角速度                        | true          |
| 8     | joint_2_angle (hip_2)    | 髋关节 2 角度                          | true          |
| 9     | joint_2_speed (hip_2)    | 髋关节 2 角速度                        | true          |
| 10    | joint_3_angle (knee_2)   | 膝关节 2 角度                          | true          |
| 11    | joint_3_speed (knee_2)   | 膝关节 2 角速度                        | true          |
| 12    | leg_1_ground_contact     | 腿 1 是否接地（0 或 1）                | true          |
| 13    | leg_2_ground_contact     | 腿 2 是否接地（0 或 1）                | true          |
| 14    | lidar_1                  | 第一根 LIDAR 测距值（前方地形高度）    | true          |
| 15    | lidar_2                  | 第二根 LIDAR 测距值                    | true          |
| 16    | lidar_3                  | 第三根 LIDAR 测距值                    | true          |
| 17    | lidar_4                  | 第四根 LIDAR 测距值                    | true          |
| 18    | lidar_5                  | 第五根 LIDAR 测距值                    | true          |
| 19    | lidar_6                  | 第六根 LIDAR 测距值                    | true          |
| 20    | lidar_7                  | 第七根 LIDAR 测距值                    | true          |
| 21    | lidar_8                  | 第八根 LIDAR 测距值                    | true          |
| 22    | lidar_9                  | 第九根 LIDAR 测距值                    | true          |
| 23    | lidar_10                 | 第十根 LIDAR 测距值                    | true          |

注：接地信号为 0/1 标量，间接反映了支撑相，可用于步态激励或摔倒检测。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- bounds: [-1.0, 1.0]  
- 各维含义：
  - action_dim 0: hip_1_torque，第一髋关节力矩
  - action_dim 1: knee_1_torque，第一膝关节力矩
  - action_dim 2: hip_2_torque，第二髋关节力矩
  - action_dim 3: knee_2_torque，第二膝关节力矩

所有动作均为连续值，力矩限幅在 [-1, 1] 内。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形末端，视为成功）
- failure-like termination: body_fallen_over（机体摔倒）
- ambiguous termination: 无
- truncation: 无时间上限截断（隐含 episode 可能在短步数内因摔倒而终止，但未明确提供 truncation 信号）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，无法直接读取 success 标志）
- explicit_failure_flag_available: false （同上，无直接 failure 字段）
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；original_reward 被强制屏蔽

终止条件的判断只能通过观测间接进行：
- 摔倒：可依据 hull_angle 绝对值超过某经验阈值（如 >0.5 rad）且可能伴随 vertical_speed 突变或 leg 接触异常；标记为 derived_possible。
- 到达终点：在 episode 结束时若 terminated=True 且未检测到摔倒，可推测为成功。但 compute_reward 中无法直接获取 terminated 标志，只能通过最后一步的 next_obs 状态推测，存在误判风险。

因此，成功/失败的信号是弱可用的，理想情况下应避免依赖终点信号，而是专注于持续前进和生存的激励。

## 7. 可用于奖励函数的信号
- position: 无直接位置（但 horizontal_speed 可积分得到水平位移增量）；垂直位移可从 vertical_speed 累积或间接通过高度变化推断（但无绝对高度观测）。
- velocity: horizontal_speed (obs[2])，vertical_speed (obs[3])，各关节角速度 (obs[5,7,9,11])
- orientation: hull_angle (obs[0])，hull_angular_velocity (obs[1])
- contact: leg_1_ground_contact (obs[12])，leg_2_ground_contact (obs[13])，二值信号，用于检测支撑相或摔倒（例如连续若干步双脚未接地即可能摔倒）。
- action/engine: action 四维力矩（hip_1, knee_1, hip_2, knee_2），可直接用于扭矩惩罚。
- other: LIDAR 读数 (obs[14:24])，提供地形预览，可用于鼓励预判性步态调整，但不易直接转化为标量奖赏，通常用于辅助特征而非独立 reward 项；也可用于检测极端地形。

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
| 1 | gated_forward_speed + posture_hinge_penalty | -57.47 | -57.47 | 0.00 | 166.90 | gated_forward_speed=0.238 posture_hinge_penalty=-0.009 | new_best |

```
