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
- current_score: -36.964653
- gap_to_target: 336.964653
- target_achievement_ratio: -12.322%

# 2. 上一轮奖励函数代码（该轮得分: -36.964653）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测提取
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    horizontal_speed = obs[2]

    # 基础前进奖励
    progress_base = 1.0 * horizontal_speed

    # 稳定性门控：abs(hull_angle) > 0.15 时开始削弱前进奖励
    angle_deviation = abs(hull_angle) - 0.15
    gate = 1.0 - 2.0 * max(0.0, angle_deviation)
    gate = max(0.0, gate)
    progress_reward = progress_base * gate

    # 躯干角速度二次惩罚（保留，轻微抑制剧烈旋转）
    angular_velocity_penalty = -0.1 * (hull_angular_velocity ** 2)

    # 动作效率惩罚
    action_efficiency_penalty = -0.01 * sum(a * a for a in action)

    total_reward = progress_reward + angular_velocity_penalty + action_efficiency_penalty
    components = {
        'progress_reward': progress_reward,
        'angular_velocity_penalty': angular_velocity_penalty,
        'action_efficiency_penalty': action_efficiency_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 411.40 | -18.01 | ✅ |
| 2 | 惩罚快速角速度将使策略学习避免剧烈旋转，减少摔倒概率，从而提升真实环境得分。 | 惩罚快速角速度将使策略学习避免剧烈旋转，减少摔倒概率，从而提升真实环境得分。 | 222.20 | -50.60 | ❌ |
| 3 | 把稳定性信号从前置松散惩罚改为强耦合的进度缩放门，会迫使 agent 在学到高速行走前先学会保持躯干稳定，从而延长... | 把稳定性信号从前置松散惩罚改为强耦合的进度缩放门，会迫使 agent 在学到高速行走前先学会保持躯干稳定，从而延长... | 255.70 | -36.96 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-36.964653, len=255.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-81.428413, 80.263518]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 105.218582 | 96.7% | 97.1% | 99.6% |
| action_efficiency_penalty | -3.168688 | -2.9% | 2.9% | 100.0% |
| angular_velocity_penalty | -0.036026 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
双足机器人需要在布满障碍（阶梯、树桩、坑洼等）的粗糙地形上持续向前行走，尽可能走得远且高效。  
主要目标是**稳定前进**并避免摔倒；次要目标包括**最小化关节力矩消耗**和**最终抵达地形末端**。  
机器人可利用前方的激光雷达（LiDAR）感知地形，提前调整步态。  
到达地形末端会正常结束，摔倒则提前失败。

## 3. 观察空间 observation_space
- type: Box  
- shape: [24]  
- dtype: 根据 float32 推断  
- 各维度含义及 reward 可用性：

| 索引 | 名称                      | 含义                           | reward_usable |
|------|---------------------------|--------------------------------|---------------|
| 0    | hull_angle                | 躯干倾斜角                     | true          |
| 1    | hull_angular_velocity     | 躯干角速度                     | true          |
| 2    | horizontal_speed           | 质心水平速度                   | true          |
| 3    | vertical_speed             | 质心垂直速度                   | true          |
| 4    | joint_0_angle （hip_1）    | 髋关节1角度                    | true          |
| 5    | joint_0_speed             | 髋关节1角速度                  | true          |
| 6    | joint_1_angle （knee_1）   | 膝关节1角度                    | true          |
| 7    | joint_1_speed             | 膝关节1角速度                  | true          |
| 8    | joint_2_angle （hip_2）    | 髋关节2角度                    | true          |
| 9    | joint_2_speed             | 髋关节2角速度                  | true          |
| 10   | joint_3_angle （knee_2）   | 膝关节2角度                    | true          |
| 11   | joint_3_speed             | 膝关节2角速度                  | true          |
| 12   | leg_1_ground_contact      | 左腿触地指示（二值）           | true          |
| 13   | leg_2_ground_contact      | 右腿触地指示（二值）           | true          |
| 14–23| lidar_1…lidar_10          | 前方10个LiDAR测距（地形高度）   | 谨慎使用      |

- 注意：LiDAR原始数值是距离测量值，可用于隐式学习地形应对，但不建议直接作为稠密奖励信号，因为其语义与前进或平衡无直接线性关系。

## 4. 动作空间 action_space
- type: Box  
- shape: [4]  
- 范围: [-1.0, 1.0]  
- 每个动作维度含义：
  - action[0]: hip_1_torque – 第一个髋关节力矩  
  - action[1]: knee_1_torque – 第一个膝关节力矩  
  - action[2]: hip_2_torque – 第二个髋关节力矩  
  - action[3]: knee_2_torque – 第二个膝关节力矩  
- 连续力矩控制，无离散动作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain – 抵达地形末端，环境正常结束。
- failure-like termination: body_fallen_over – 身体倾倒（典型失败）。
- ambiguous termination: 无。
- truncation: 无明显时间截断（原文未提及 max steps，但可能存在于环境中，视为 ambiguous）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （info 字段为空，不允许使用）
- explicit_failure_flag_available: **false**
- allowed_info_fields: []  （info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（因为不允许使用任何 info 内容）
- 终止原因只能通过以下方式**推断**（derived_possible）：
  - **摔倒推断**：终止时 `next_obs[0]`（hull_angle）很可能超过阈值（如>0.8 rad），或者两腿触地指示同时为0且躯干姿态异常。可利用 `next_obs` 在 reward 中检测。
  - **到达终点推断**：终止时 `next_obs` 的 hull_angle 较小且无异常，但无法从观测直接区分；因为无位置信息，可通过 episode 忽然结束且未触发摔倒检测来判断。奖励函数设计中可仅通过前进速度奖励覆盖此目标，避免依赖显式到达奖励。

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
| 1 | action_efficiency_penalty + progress_reward + stability_penalty | -18.01 | -18.01 | 0.00 | 411.40 | action_efficiency_penalty=-0.018 progress_reward=0.218 stability_penalty=-0.010 | new_best |
| 2 | action_efficiency_penalty + angular_velocity_penalty + progress_reward + stability_penalty | -50.60 | -18.01 | -32.59 | 222.20 | action_efficiency_penalty=-0.018 angular_velocity_penalty=-0.000 progress_reward=0.202 stability_penalty=-0.008 | no_meaningful_improvement |
| 3 | action_efficiency_penalty + angular_velocity_penalty + progress_reward | -36.96 | -18.01 | -18.95 | 255.70 | action_efficiency_penalty=-0.017 angular_velocity_penalty=-0.000 progress_reward=0.267 | no_meaningful_improvement |

```
