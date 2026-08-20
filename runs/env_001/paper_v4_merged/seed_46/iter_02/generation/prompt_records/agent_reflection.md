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
- current_score: 236.610458
- gap_to_target: -36.610458
- target_achievement_ratio: 118.305%

# 2. 上一轮奖励函数代码（该轮得分: 236.610458）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    # obs: [x_pos, y_pos, x_vel, y_vel, body_angle, angular_vel, left_contact, right_contact]
    ox, oy, ovx, ovy, oangle, oav, olc, orc = obs
    nx, ny, nvx, nvy, nangle, nav, nlc, nrc = next_obs

    # Compute distances
    old_dist = (ox * ox + oy * oy) ** 0.5
    new_dist = (nx * nx + ny * ny) ** 0.5

    # --- Goal proximity progress (main learning signal) ---
    raw_progress = old_dist - new_dist   # positive when moving closer
    max_delta = 5.0                      # clip single-step changes
    progress = max(-max_delta, min(max_delta, raw_progress))
    progress_reward = 1.0 * progress     # weight = 1.0

    # --- Landing gentleness (constraint) ---
    CLOSE_DIST = 3.0
    SAFE_SPEED = 1.0
    LAND_WEIGHT = 0.5
    speed = (nvx * nvx + nvy * nvy) ** 0.5
    # Linear activation inside the close region (0 -> 1 as distance decreases)
    close_factor = max(0.0, 1.0 - new_dist / CLOSE_DIST)
    # Hinge penalty on excess speed, scaled by close_factor
    landing_penalty = -LAND_WEIGHT * max(0.0, speed - SAFE_SPEED) * close_factor

    # --- Orientation penalty (constraint) ---
    ANGLE_THRESHOLD = 0.3   # radians
    ORIENT_WEIGHT = 0.2
    orientation_penalty = -ORIENT_WEIGHT * max(0.0, abs(nangle) - ANGLE_THRESHOLD)

    # --- Terminal success bonus (task-completion proxy) ---
    SUCCESS_DIST = 0.2
    SUCCESS_SPEED = 0.5
    SUCCESS_ANGLE = 0.2
    SUCCESS_BONUS = 0.2
    success_bonus = 0.0
    if (new_dist < SUCCESS_DIST and speed < SUCCESS_SPEED
            and abs(nangle) < SUCCESS_ANGLE
            and (nlc > 0.5 or nrc > 0.5)):
        success_bonus = SUCCESS_BONUS

    total_reward = progress_reward + landing_penalty + orientation_penalty + success_bonus

    components = {
        "goal_proximity_progress": progress_reward,
        "landing_gentleness_penalty": landing_penalty,
        "orientation_penalty": orientation_penalty,
        "terminal_success_bonus": success_bonus
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=236.610458, len=419.850000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[64.846763, 310.187661]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success_bonus | 40.220000 | 96.3% | 96.3% | 47.9% |
| goal_proximity_progress | 1.326154 | 3.2% | 3.4% | 97.8% |
| orientation_penalty | -0.099911 | -0.2% | 0.2% | 2.3% |
| landing_gentleness_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Score 236.6, termination 85%, episode length ~420. Terminal_success_bonus dominates with 96.3% signed share.

**Component Anomalies**: landing_gentleness_penalty dead (active 0%). orientation_penalty near-dead (active 2.3%, -0.2% share). terminal_success_bonus >70% share.

**Training Dynamics**: No temporal snapshots provided; dynamics over checkpoints unavailable.

**Signal Quality**: Dead gate landing_gentleness. Progress reward active (97.8%) but contributes only 3.2% share. Success bonus fires in 47.9% steps, driving high accumulation.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从画面顶部附近出发，尽快飞到中央目标平台并稳定着陆，同时尽量减少引擎推力使用。  
次优要求：保持姿态稳定、安全轻触平台、避免侧向偏移过大。  
不应混淆的目标：单纯存活（有明确的到达位置要求），也不是无限制前进（目标是一个固定点位）。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `(8,)`
- dtype: `float32`（假定，实际由环境决定）
- obs[0]: `x_position` – 相对于目标平台的水平距离，`reward_usable: true`
- obs[1]: `y_position` – 相对于平台高度的垂直距离，`reward_usable: true`
- obs[2]: `x_velocity` – 水平线速度，`reward_usable: true`
- obs[3]: `y_velocity` – 垂直线速度，`reward_usable: true`
- obs[4]: `body_angle` – 机体倾角，`reward_usable: true`
- obs[5]: `angular_velocity` – 角速度，`reward_usable: true`
- obs[6]: `left_support_contact` – 左支撑触地标志（0/1 或连续），`reward_usable: true`
- obs[7]: `right_support_contact` – 右支撑触地标志，`reward_usable: true`

## 4. 动作空间 action_space
- type: `Discrete`
- n: `4`
- action 0: `no_engine` – 无推力，惯性飞行
- action 1: `left_orientation_engine` – 启动左姿态发动机（主要用于调整角速度）
- action 2: `main_engine` – 启动主发动机（提供反推力/升力）
- action 3: `right_orientation_engine` – 启动右姿态发动机（与左姿态对称）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` – 当机体静止（可能已着陆并稳定）时触发。此模式极可能表示成功着陆，尤其是配合近距离、低速、良好姿态和接触信号。
- **failure-like termination**: `crash_or_body_contact` – 机体与地面或其他物体非腿部接触（推测会导致姿态破坏、超出限制等）。  
  `horizontal_position_outside_viewport` – 水平位置超出视野，必然失败（飞离目标区）。
- **ambiguous termination**: 当同时满足多个条件时（如 crash 且出界），仍视为失败；但仅靠观测无法区分触发原因。`crash_or_body_contact` 和 `body_not_awake_or_settled` 可能同时触发，需以 fail 为准（因为 crash 优先级高）。
- **truncation**: 源码中未看到最大步数截断，但实际使用中可能存在。无额外截断标志。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: `false` (info 为空)
- `explicit_failure_flag_available`: `false`
- `allowed_info_fields`: 无（info 为 `{}`）
- `forbidden_or_uncertain_info_fields`: 任何假设的 `info["success"]`、`info["termination_reason"]` 等均禁止使用

即使没有明确标志，成功可通过终止后状态间接推断：位置接近零、速度极小、姿态平直且至少有一只脚接触平台。这属于 `derived_possible` 信号，在奖励设计时必须谨慎。

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]` (x 偏差), `next_obs[1]` (y 偏差) → 可计算到目标的距离
- **velocity**: `next_obs[2]`, `next_obs[3]` → 总速度或分速度
- **orientation**: `next_obs[4]` (机体倾角), `next_obs[5]` (角速度)
- **contact**: `next_obs[6]`, `next_obs[7]` – 左右支撑是否触地
- **action/engine**: 动作选择本身（离散 0‑3）可用于惩罚引擎使用
- **other**:
  - 距离变化量：`delta_distance = distance(obs) - distance(next_obs)`（进步信号）
  - 终端推断成功（derived_possible）：在 episode 结束时，结合位置、速度、倾角、接触判断是否为成功着陆，可给予稀疏终端奖励
  - 终端推断失败（derived_possible）：推断 crash/出界，可给予惩罚（谨慎使用）

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
| 1 | goal_proximity_progress + landing_gentleness_penalty + orientation_penalty + terminal_success_bonus | 236.61 | 236.61 | 0.00 | 419.85 | goal_proximity_progress=0.002 landing_gentleness_penalty=-0.001 orientation_penalty=-0.002 terminal_success_bonus=0.127 | target_solved_new_best |

```
