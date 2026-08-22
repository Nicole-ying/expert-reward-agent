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
- current_score: 243.728141
- gap_to_target: -43.728141
- target_achievement_ratio: 121.864%

# 2. 上一轮奖励函数代码（该轮得分: 243.728141）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs / next_obs : [x, y, vx, vy, angle, ang_vel, left_contact, right_contact]
    # goal is at origin (0,0); x,y relative to target pad
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    delta_distance = dist_old - dist_new  # positive when approaching

    # Soft landing progress: encourage closeness, low speed, low angle
    near_goal = 1.0 / (1.0 + 5.0 * dist_new)
    speed_sq = next_obs[2]**2 + next_obs[3]**2
    low_speed = 1.0 / (1.0 + 10.0 * speed_sq)
    abs_angle = abs(next_obs[4])
    low_angle = 1.0 / (1.0 + 20.0 * abs_angle)
    soft_progress = near_goal * low_speed * low_angle

    # Engine usage penalty: penalize any thrust action (discrete actions 1,2,3)
    engine_penalty = 1.0 if action != 0 else 0.0

    # Weights
    w_dist = 10.0
    w_soft = 2.0
    w_engine = 0.01

    total = (w_dist * delta_distance +
             w_soft * soft_progress -
             w_engine * engine_penalty)

    components = {
        'distance_delta': w_dist * delta_distance,
        'soft_landing_progress': w_soft * soft_progress,
        'engine_penalty': -w_engine * engine_penalty,
    }
    return float(total), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 554.20 | 216.19 | ✅ |
| 2 | 骨架变化: distance_delta + engine_penalty + soft_landing_pro | — | 356.30 | 243.73 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=243.728141, len=356.300000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[138.980073, 287.379143]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_progress | 145.720812 | 89.8% | 89.8% | 100.0% |
| distance_delta | 13.032984 | 8.0% | 8.5% | 96.3% |
| engine_penalty | -2.853500 | -1.8% | 1.8% | 80.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: soft_landing_progress dominates reward at 89.8% signed share; distance_delta only 8.0%. Episode sum mean soft=145.7, distance=13.0.

**Component Anomalies**: soft_landing_progress dominates (>70% share), distance_delta underweighted; engine_penalty small and negative but active 80%.

**Training Dynamics**: no temporal snapshots provided; cannot assess component growth/decay or scaffold drift.

**Signal Quality**: all components active (high active rates), no dead signals. soft_landing_progress overwhelms other signals, potentially masking distance improvements.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
任务目标是控制一个受随机初始力作用的二维飞行器，使其从视野顶部中央出发，尽可能快地飞抵并稳定停靠于中央目标平台上。核心目标是“到达并停靠”（reaching and settling），附属优化目标包括“尽可能少用引擎推力”和“保持姿态稳定、安全接触”。不应将其混淆为单纯的存活任务、无限制漫游任务或多目标博弈任务。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64 (推测连续量)
- obs[0]: x_position (水平方向相对于目标平台的坐标)，reward_usable: true
- obs[1]: y_position (垂直方向相对于平台高度的坐标)，reward_usable: true
- obs[2]: x_velocity (水平线速度)，reward_usable: true
- obs[3]: y_velocity (垂直线速度)，reward_usable: true
- obs[4]: body_angle (机体朝向角)，reward_usable: true
- obs[5]: angular_velocity (角速度)，reward_usable: true
- obs[6]: left_support_contact (左支撑腿接触标志, 0/1)，reward_usable: true
- obs[7]: right_support_contact (右支撑腿接触标志, 0/1)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine (无推力)，即不激活任何引擎
- action 1: left_orientation_engine (左姿态引擎)，产生顺时针或逆时针力矩中的一种；具体方向需在交互中推断，但用于调整朝向
- action 2: main_engine (主引擎)，沿机体纵轴提供推力，用于平移/减速/抵抗重力
- action 3: right_orientation_engine (右姿态引擎)，产生与左姿态引擎相反的力矩，用于反方向姿态修正

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止；任务期望通过“到达并稳定停靠”后 episode 结束，这很可能通过 **timeout/truncation** 或在目标区域达到低速度、双腿接触、小角度等条件后被环境内部判定为 settled 而终止。
- failure-like termination:
  - *crash_or_body_contact*: 机体部分（非支撑腿）碰撞地面/平台以外区域，或姿态严重偏离导致翻倒。
  - *horizontal_position_outside_viewport*: 机体飞出水平边界，视为严重失控。
  - *body_not_awake_or_settled*: 可能是检测到速度/加速度极小但未达成着陆条件，或进入睡眠状态的超时机制。
- ambiguous termination: 支撑腿接触目标平台但未满足所有稳定条件，被 terminated 可能属于部分成功/硬着陆，不能直接视为完美成功。
- truncation: 任务可能包含 episode 长度上限，届时会直接截断。该截断不携带成功/失败固有语义。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 根据 source，step 返回空字典 `{}` ，因此 `info` 无任何可用字段。
- forbidden_or_uncertain_info_fields: info字典为空，无字段可用；不得依赖任何隐式 info 键。

## 7. 可用于奖励函数的信号
- position: `next_obs[0:2]` 表示相对于目标平台的位置。可计算当前距离、距离变化量。
- velocity: `next_obs[2:4]` 线速度。可用于接近速度、稳定着陆时趋零、水平漂移控制。
- orientation: `next_obs[4]` 机体角度。可用于姿态维护、着陆时接近水平的奖励/惩罚。
- contact:
  - `next_obs[6]` 左支撑腿接触
  - `next_obs[7]` 右支撑腿接触
  - derived_possible: 双腿同时接触（legs_contact = left & right）是成功着陆的关键条件，可直接从观测构造。
- action/engine: `action` 可以用于对引擎使用施加惩罚。
- other:
  - angular_velocity `next_obs[5]` 可用于控制姿态抖动的阻尼惩罚。
  - derived_possible: settled 成功事件可间接推断：如果 episode 未因 crash/越界终止而截断，且最后几步保持双腿接触、低速度、小角度，则很可能为成功着陆。可在最终奖励中使用 sparse terminal success bonus，但必须标注为 derived_possible，且需在策略中小心处理以避免误判。

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
| 1 | angle_penalty + distance_delta + engine_penalty + soft_landing_progress | 216.19 | 216.19 | 0.00 | 554.20 | angle_penalty=-0.001 distance_delta=0.031 engine_penalty=-0.007 soft_landing_progress=0.728 | target_solved_new_best |
| 2 | distance_delta + engine_penalty + soft_landing_progress | 243.73 | 243.73 | 0.00 | 356.30 | distance_delta=0.029 engine_penalty=-0.007 soft_landing_progress=0.767 | target_solved_new_best |

```
