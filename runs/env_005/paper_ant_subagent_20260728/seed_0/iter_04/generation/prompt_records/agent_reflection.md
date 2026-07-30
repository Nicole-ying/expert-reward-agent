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
- current_score: 1839.706014
- gap_to_target: 160.293986
- target_achievement_ratio: 91.985%

# 2. 上一轮奖励函数代码（该轮得分: 1839.706014）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]  # forward velocity
    v_y      = obs[14]  # lateral velocity

    # ---- upright measure (continuous, 1.0 when perfectly upright, -1.0 inverted) ----
    up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---- upright gate (soft multiplication to forward progress) ----
    # Gate starts to decay when up_z < 1.0, reaches 0.0 when up_z ≤ 0.3
    gate_lower = 0.3
    gate = (up_z - gate_lower) / (1.0 - gate_lower)
    gate = max(0.0, min(1.0, gate))

    # ---- gated forward progress ----
    w_fwd = 1.0
    gated_forward = w_fwd * v_x * gate

    # ---- upright bonus (continuous, provides direct posture gradient) ----
    w_up_bonus = 0.2
    upright_bonus = w_up_bonus * up_z

    # ---- body height safety (soft hinge quadratic, only near boundaries) ----
    z_low_safe  = 0.3
    z_high_safe = 0.8
    w_h = 0.5
    low_hinge  = max(0.0, z_low_safe - body_z)
    high_hinge = max(0.0, body_z - z_high_safe)
    height_penalty = -w_h * (low_hinge**2 + high_hinge**2)

    # ---- lateral stability (quadratic penalty) ----
    w_lat = 0.2
    lateral_penalty = -w_lat * (v_y**2)

    # ---- action smoothing (light energy regularizer) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a**2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = (gated_forward + upright_bonus +
                    height_penalty + lateral_penalty + action_penalty)

    components = {
        "gated_forward":      gated_forward,
        "upright_bonus":      upright_bonus,
        "height_penalty":     height_penalty,
        "lateral_penalty":    lateral_penalty,
        "action_penalty":     action_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 11.80 | 0.72 | ✅ |
| 2 | 移除 health_gate 释放 forward 信号 + 将 upright 改为连续温和二次惩罚 → age... | 移除 health_gate 释放 forward 信号 + 将 upright 改为连续温和二次惩罚 → age... | 503.55 | -383.09 | ❌ |
| 3 | 将前进收益门控在直立姿态上，迫使 agent 必须先保持直立才能获利，能打破两大项对冲的局面，使 forward ... | 将前进收益门控在直立姿态上，迫使 agent 必须先保持直立才能获利，能打破两大项对冲的局面，使 forward ... | 981.50 | 1839.71 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=1839.706014, len=981.500000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[896.580849, 2166.796600]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 3421.016521 | 90.3% | 90.3% | 98.2% |
| upright_bonus | 186.142573 | 4.9% | 5.1% | 100.0% |
| lateral_penalty | -170.575763 | -4.5% | 4.5% | 98.7% |
| action_penalty | -3.307078 | -0.1% | 0.1% | 100.0% |
| height_penalty | -0.030798 | -0.0% | 0.0% | 2.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Score=1839.7 across ~981-step episodes; only 2/20 terminated (10%), 18/20 truncated. Per-step generated_reward=1.14 vs original_env_reward=-0.11 — the native env considers behavior suboptimal despite high generated scores. The agent survives long but rarely reaches terminal state.

**Component Anomalies**: gated_forward dominates at 90.3% signed share (episode_sum=3421). upright_bonus (+4.9%) and lateral_penalty (-4.5%) nearly cancel. height_penalty is dead: 2.7% active rate, 0.0% share — safe zone [0.3, 0.8] is never breached. action_penalty negligible at -0.1%.

**Training Dynamics**: No temporal snapshots available (component_dynamics returned empty). Cannot assess scaffold→final drift or checkpoint-level trends. All evidence is from final-policy fixed-seed evaluation.

**Signal Quality**: height_penalty dead (safe zone too wide — thresholds never triggered). original_env_reward is negative (-0.11/step) despite high generated reward — possible reward hacking where policy exploits gated_forward without truly solving the locomotion task. Low termination rate (10%) suggests the agent isn't reaching episode goals.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个3D四足机器人向前稳定行走/奔跑。核心目标是产生持续的前向速度，同时保持身体高度在安全范围（0.2 ~ 1.0）内不摔倒。次要目标包括维持直立姿态、减少侧向漂移、控制能耗和动作平滑。任务 **不要求** 到达某个指定位置，仅要求长期存活并向前移动。不能混淆为“仅站立不动”或“最小化能量消耗”，前进是刚性主目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: 连续浮点数（具体精度由环境决定）
- obs[0] (body_z): 身体高度，reward_usable: true，可用作安全高度监控
- obs[1] (quat_w): 身体姿态四元数实部，reward_usable: true，参与直立度计算
- obs[2] (quat_x): 四元数虚部 x，reward_usable: true
- obs[3] (quat_y): 四元数虚部 y，reward_usable: true
- obs[4] (quat_z): 四元数虚部 z，reward_usable: true
- obs[5] (joint_1_angle): 髋关节1角度，reward_usable: true（可做动作平滑或参考姿态）
- obs[6] (joint_2_angle): 踝关节1角度，reward_usable: true
- obs[7] (joint_3_angle): 髋关节2角度，reward_usable: true
- obs[8] (joint_4_angle): 踝关节2角度，reward_usable: true
- obs[9] (joint_5_angle): 髋关节3角度，reward_usable: true
- obs[10] (joint_6_angle): 踝关节3角度，reward_usable: true
- obs[11] (joint_7_angle): 髋关节4角度，reward_usable: true
- obs[12] (joint_8_angle): 踝关节4角度，reward_usable: true
- obs[13] (body_x_velocity): 世界x轴（前向）速度，reward_usable: true，**主前向奖励信号**
- obs[14] (body_y_velocity): 世界y轴（侧向）速度，reward_usable: true，可惩罚侧向
- obs[15] (body_z_velocity): 垂直速度，reward_usable: true，可惩罚剧烈上下起伏
- obs[16] (body_roll_velocity): 滚转角速度，reward_usable: true，用于稳定性惩罚
- obs[17] (body_pitch_velocity): 俯仰角速度，reward_usable: true
- obs[18] (body_yaw_velocity): 偏航角速度，reward_usable: true，转弯惩罚
- obs[19] (joint_1_velocity): 关节1角速度，reward_usable: true（动作平滑/能耗）
- obs[20] (joint_2_velocity): 关节2角速度，reward_usable: true
- obs[21] (joint_3_velocity): 关节3角速度，reward_usable: true
- obs[22] (joint_4_velocity): 关节4角速度，reward_usable: true
- obs[23] (joint_5_velocity): 关节5角速度，reward_usable: true
- obs[24] (joint_6_velocity): 关节6角速度，reward_usable: true
- obs[25] (joint_7_velocity): 关节7角速度，reward_usable: true
- obs[26] (joint_8_velocity): 关节8角速度，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- 连续动作，每个维度范围 [[-1.0, 1.0]]
- action_dim 0: hip_1_torque — 第一髋关节扭矩
- action_dim 1: ankle_1_torque — 第一踝关节扭矩
- action_dim 2: hip_2_torque — 第二髋关节扭矩
- action_dim 3: ankle_2_torque — 第二踝关节扭矩
- action_dim 4: hip_3_torque — 第三髋关节扭矩
- action_dim 5: ankle_3_torque — 第三踝关节扭矩
- action_dim 6: hip_4_torque — 第四髋关节扭矩
- action_dim 7: ankle_4_torque — 第四踝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确的成功终止标志；可默认为“在时间限制（truncation）内始终保持健康姿态”视为一次成功完整运行。
- **failure-like termination**:  
  - body_height_outside_healthy_range：身体高度 z ≤ 0.2（摔倒）或 z ≥ 1.0（过度跃起）。  
  - state_value_outside_finite_range：任何状态值变为 NaN 或 inf，通常代表物理崩溃。  
  两类均直接终止回合，属于硬失败。
- **ambiguous termination**: 无。
- **truncation**: time_limit_reached（达到最大步数），表示存活完全程。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** 
- explicit_failure_flag_available: **false** （`info` 字段为空，不能直接获得终止原因，仅能从环境返回的 `terminated` 或 `truncated` 在 RL 循环中判断，但奖励函数接口不提供这些标志）
- allowed_info_fields: 无（info 为空字典）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin

## 7. 可用于奖励函数的信号
- **位置相关**：身体高度 `body_z`（obs[0]）；身体姿态四元数 `quat_w,x,y,z`（obs[1:5]），可计算 body_up_z。关节角度（obs[5:13]）可构造姿态正则化或对称性惩罚。
- **速度相关**：前向速度 `body_x_velocity`（obs[13]）——直接前进奖励；侧向速度 `body_y_velocity`（obs[14]）——侧向漂移惩罚；垂直速度 `body_z_velocity`（obs[15]）——起伏惩罚；角速度 `body_roll/pitch/yaw_vel`（obs[16:19]）——稳定性和转向惩罚；关节角速度（obs[19:27]）——动作平滑/能耗。
- **动作/执行器**：`action`（8维扭矩）可用于计算力矩大小、变化量。
- **其他**：训练进度（若环境描述明确需要，但此处未强调，谨慎使用）。

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + forward + height_penalty + lateral_penalty + upright_penalty | 0.72 | 0.72 | 0.00 | 11.80 | action_penalty=-0.003 forward=0.222 height_penalty=-0.032 lateral_penalty=-0.097 upright_penalty=-5.673 | new_best |
| 2 | action_penalty + forward + height_penalty + lateral_penalty + upright_penalty | -383.09 | 0.72 | -383.81 | 503.55 | action_penalty=-0.003 forward=0.372 height_penalty=-0.003 lateral_penalty=-0.117 upright_penalty=-0.758 | no_meaningful_improvement |
| 3 | action_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | 1839.71 | 1839.71 | 0.00 | 981.50 | action_penalty=-0.003 gated_forward=1.150 height_penalty=-0.000 lateral_penalty=-0.103 upright_bonus=0.095 | new_best |

```
