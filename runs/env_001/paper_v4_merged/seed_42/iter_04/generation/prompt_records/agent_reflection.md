# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。先用训练证据解释失败，再选择最小且可验证的干预。正常模式每轮只改一个组件。重建模式（用户 prompt 标有 REBUILD MODE）下可更换主信号框架。

# 证据边界

- 只根据环境事实理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- `episode_sum_mean`=每回合有符号累计量，`magnitude_share`=绝对累计量份额，`signed_share`=净方向，`active_rate`=非零触发率。
- 组件统计是观察证据不是因果贡献，必须结合 score、episode_length、terminated/truncated、历史修改判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件不能套同一个比例阈值。
- 不得仅因任务描述出现语义关键词就断言缺失职责。新增职责必须有轨迹行为、终止分布或组件激活证据。

# 决策流程（按顺序，不可跳级）

## 0. 信号覆盖审计（清单式，逐项过）

- **0.1 终止模式**：大部分 episode 是 truncated(=超时存活) 还是 terminated 且短(=快速失败)？结合环境声明的终止条件推断哪种触发。
- **0.2 观测扫描**：哪些 obs 维度未被使用？未使用的维度能否解释当前终止模式？
- **0.3 信号缺口**：综合 0.1+0.2 → 信号齐全但校准问题？还是信号缺失需新组件？
- **0.4 僵尸组件**：`active_rate < 2%` 且分数未因它改善 → 该组件意图未实现，删除或替换。

## 1. 行为与历史诊断

1. **agent 发生了什么？** 快速失败(短ep) / 徘徊(长ep全truncated) / 刷分exploit？
2. **哪个组件最值得干预？** 结合数学形态、episode_sum_mean、signed_share、magnitude_share、active_rate、外部 score 和 episode_length 判断。一次只选一个目标。
3. **我之前改了什么？** 从累积记录检查上一轮动作和实际效果。如果上次改了A但得分没变，这次不要再次改A。
4. **这个方向还值得继续吗？** 累积记录中同骨架连续 ≥3 轮未刷新 best → 当前方向大概率错误，考虑 Level 3。

## 2. 选择干预层级

**Level 1 — 尺度修复**：职责完备、数学形态合理，只是系数/阈值异常。
- `|penalty/progress| > 0.5` 且 active_rate≈100% → 降系数至 0.1~0.3x。
- 一次尺度修复后尺度异常已消失但行为没改善 → 不继续调同一系数，转 Level 2。

**Level 2 — 结构变换**：缺职责、active_rate 接近 0、数学形态塌缩。每轮只改一个组件。

| 证据模式 | 结构变换 | 下一轮应验证 |
|---|---|---|
| active_rate < 5%，缺少局部反馈 | sparse→dense：二值→连续 bounded factor | active_rate 上升，不产生 proxy 徘徊 |
| 极端值支配 reward | unbounded→bounded | 极端轨迹支配下降 |
| 占据好状态即持续获奖 | state→improvement：状态值→改善量 | 停留不再积累收益，任务进展改善 |
| 约束在无关阶段妨碍探索 | global→local：全局惩罚→局部门控 | 早期探索与局部约束同时改善 |
| 独立目标可互相补偿 | independent→joint：加权和→联合满足 | 单项刷分减少 |
| 乘积经常塌缩为 0 | product→noncollapsing：乘积→几何平均/独立求和 | 非零反馈增多 |
| proxy 提高但外部分数不升 | proxy→completion_alignment | proxy 与外部分数重新同向 |
| 第 0 步发现信号缺口 | add 新组件（使用已声明但未用的 obs 维度） | 新组件 active_rate > 0，不破坏现有正信号 |

**Level 3 — 重建骨架**：满足任一即重建（从累积记录的客观数据判断）：
- 同一骨架连续 ≥3 轮未刷新 best（看累积记录中同骨架的 best 列是否停滞）
- 同一骨架族已迭代 ≥4 轮，且历史最佳仍未超过 target×0.5
- Level 2 改变数学形态后得分没有实质改善

## 3. 设计校准（写代码前检查）

1. 新惩罚 per-step ≤ 主信号 per-step 的 0.3x。主信号 per-step ≈ episode_sum_mean/len。
2. hinge 阈值设在终止边界的 60-80% 处。
3. gate 在"不理想但安全"区域 ≥ 0.3。
4. 总惩罚 per-step ≤ 主信号 per-step 的 0.5x。
5. 若累积记录中 len 自某轮常驻惩罚加入后暴跌且未恢复 → 优先削弱它而非加新东西。

# 输出格式

先用 8 个固定字段各写一句，不复述输入表格：

1. `evidence`：支持判断的外部结果、组件证据和上一轮结果
2. `behavior_diagnosis`：策略当前的失败行为
3. `signal_completeness`：必要职责是否完备、可达
4. `selected_level`：Level 1/2/3 及触发条件
5. `selected_intervention`：唯一目标组件及具体修改
6. `falsifiable_hypothesis`：为什么该修改应改善策略（必须能被下一轮反馈证伪）
7. `expected_next_round`：下一轮哪些指标应如何变化（定量预测）
8. `main_risk`：最可能引入的新漏洞

然后立即输出完整 Python 代码。预期必须在下一轮反馈中可以验证。

# 代码约束

- 只用环境事实声明的 obs/action 维度和索引。
- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止 import、class、try/except、eval/exec/open。
- 平方根 `** 0.5`；指数 `2.718281828 ** exponent`。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`

```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: 260.060104
- gap_to_target: -60.060104
- target_achievement_ratio: 130.030%

# 2. 上一轮奖励函数代码（该轮得分: 260.060104）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 航向进展：距离目标越近越好（improvement_delta）
    d_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_prev - d_next
    goal_progress = 1.0 * progress

    # 2. 稳定停靠奖励：靠近目标时鼓励低速、竖直、双腿接触
    proximity_thresh = 0.5
    proximity_gate = max(0.0, 1.0 - d_next / proximity_thresh)

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    vel_thresh = 0.2
    velocity_bonus = 0.5 * max(0.0, 1.0 - speed / vel_thresh)

    angle_thresh = 0.1
    angle_bonus = 0.2 * max(0.0, 1.0 - abs(next_obs[4]) / angle_thresh)

    contact_bonus = 1.0 * next_obs[6] * next_obs[7]

    stable_bonus = proximity_gate * (velocity_bonus + angle_bonus + contact_bonus)

    # 3. 燃料效率惩罚
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 4. 密集距离奖励：越接近目标奖励越大（连续有界）
    approach_reward = 0.1 / (1.0 + d_next)

    # 5. 角速度稳定奖励（新组件，利用未使用的 obs[5]）
    ang_vel = abs(next_obs[5])
    ang_vel_thresh = 0.2
    angular_stability = 0.1 * max(0.0, 1.0 - ang_vel / ang_vel_thresh)

    total_reward = goal_progress + stable_bonus + fuel_penalty + approach_reward + angular_stability
    components = {
        'goal_progress': float(goal_progress),
        'stable_bonus': float(stable_bonus),
        'fuel_penalty': float(fuel_penalty),
        'approach_reward': float(approach_reward),
        'angular_stability': float(angular_stability)
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.75 | -92.75 | ✅ |
| 2 | 骨架变化: approach_reward + fuel_penalty + goal_progress + s | — | 661.90 | 194.87 | ✅ |
| 3 | 骨架变化: angular_stability + approach_reward + fuel_penalty | — | 316.35 | 260.06 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=260.060104, len=316.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[233.959286, 296.266506]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_bonus | 89.018430 | 62.8% | 62.8% | 60.9% |
| angular_stability | 24.841976 | 17.5% | 17.5% | 97.8% |
| approach_reward | 23.630877 | 16.7% | 16.7% | 100.0% |
| fuel_penalty | -2.844000 | -2.0% | 2.0% | 89.9% |
| goal_progress | 1.375708 | 1.0% | 1.0% | 95.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: mean_eval_reward=260.06, term=20/20, ep_len=316.35; score range 233-296

**Component Anomalies**: stable_bonus dominates signed_share=62.8% (active_rate=60.9%); fuel_penalty negative signed_share=-2.0% (magnitude_share=2.0%)

**Training Dynamics**: no checkpoint snapshots; temporal trends unavailable

**Signal Quality**: all components have nonzero activity; no dead gates; no self-cancelling (high magnitude near-zero share); stable_bonus activates less frequently but dominates reward

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从视口顶部出发，以最短时间和最少推力消耗到达并稳定停靠在画面中心的目标平台上。  
要求同时满足：水平与垂直位置均收敛至平台原点、速度趋近于零、身体姿态保持稳定、左右支撑腿同时与平台接触，且过程中避免坠毁、翻倾或飞出边界。  
任务核心是精准导航‑停靠；附属优化是燃料经济与快速性，两者不应混淆为主要目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (隐含，所有分量均为连续值或 0/1 浮点数)

各维度含义：

- obs[0]: x_position — 飞行器相对于目标平台中心的水平坐标，reward_usable: true  
- obs[1]: y_position — 飞行器相对于平台高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity — 水平线速度，reward_usable: true  
- obs[3]: y_velocity — 垂直线速度，reward_usable: true  
- obs[4]: body_angle — 身体朝向角，reward_usable: true  
- obs[5]: angular_velocity — 角速度，reward_usable: true  
- obs[6]: left_support_contact — 左支撑腿是否与表面接触 (1.0 接触，0.0 未接触)，reward_usable: true  
- obs[7]: right_support_contact — 右支撑腿是否与表面接触 (1.0 接触，0.0 未接触)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4

各动作含义：

- action 0: no_engine — 不启动任何引擎，只靠惯性运动  
- action 1: left_orientation_engine — 点燃左侧姿态引擎，用于调整角度/旋转  
- action 2: main_engine — 点燃主引擎，通常产生沿身体某方向的推力（可能包含垂直方向的一次性推力）  
- action 3: right_orientation_engine — 点燃右侧姿态引擎，与左引擎反向旋转

动作选择直接影响燃料消耗和姿态变化，奖励设计中需要跟踪动作计数来估计燃料/推力使用。

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact — 飞行器主体发生不应有的碰撞或坠毁，通常视为失败  
- horizontal_position_outside_viewport — 水平位置超出画面边界，视为失败  
- body_not_awake_or_settled — 身体进入沉睡状态或判定为已稳定停靠，可能是成功，但源码中未区分是否为正常着陆成功

无任何显式成功/失败标志传入 info 字典，因此需要**通过观测信号间接推断**终止原因。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: (空字典，没有任何字段)  
- forbidden_or_uncertain_info_fields: 禁止使用 info 读取任何字段，因为环境不提供额外信息

终止判断：
- 若 episode 终止 (not truncated) 且最终观测满足 `距离目标近、双腿接触、速度低、姿态角小`，则推断为**成功停靠** (derived_possible)  
- 若最终状态中出现任意一条不满足（如位置严重偏离、未接触、速度极大、角度过大），则推断为**失败**（坠毁、出界等）  
- 由于 termination 函数已混合了成功与失败条件，无法直接从环境获取标签，所以成功奖励必须通过 derived 推断给出

## 7. 可用于奖励函数的信号
可直接使用的观测信号：
- position (相对于目标): `obs[0]` (x), `obs[1]` (y)  
- velocity: `obs[2]` (vx), `obs[3]` (vy)  
- orientation: `obs[4]` (angle), `obs[5]` (angular_vel)  
- contact: `obs[6]` (left contact), `obs[7]` (right contact)  
- action/engine: 当前动作 `action`，可用于检测引擎使用

间接可用信号（从观测推导）：
- distance_to_target: ‖(obs[0], obs[1])‖  
- is_crashed_or_oob: 由最终状态的位置、速度突变、接触缺失推断 (derived_possible)  
- is_successful_landing: 距离近、双腿均接触、速度低、角速度低 (derived_possible)  
- fuel_usage: 动作 1、2、3 视为消耗燃料（可加权计数）

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
| 1 | fuel_penalty + goal_progress + stable_bonus | -92.75 | -92.75 | 0.00 | 68.75 | fuel_penalty=-0.002 goal_progress=0.016 stable_bonus=0.022 | new_best |
| 2 | approach_reward + fuel_penalty + goal_progress + stable_bonus | 194.87 | 194.87 | 0.00 | 661.90 | approach_reward=0.078 fuel_penalty=-0.007 goal_progress=0.002 stable_bonus=0.714 | new_best |
| 3 | angular_stability + approach_reward + fuel_penalty + goal_progress + stable_bonus | 260.06 | 260.06 | 0.00 | 316.35 | angular_stability=0.084 approach_reward=0.080 fuel_penalty=-0.007 goal_progress=0.002 stable_bonus=0.766 | target_solved_new_best |

```
