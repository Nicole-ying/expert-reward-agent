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
- current_score: 144.813937
- gap_to_target: 55.186063
- target_achievement_ratio: 72.407%

# 2. 上一轮奖励函数代码（该轮得分: 144.813937）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observations
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    angvel = obs[5]
    # left_contact, right_contact from obs (not used)
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Progress towards center (0,0)
    dist = (x**2 + y**2)**0.5 + 1e-6
    next_dist = (next_x**2 + next_y**2)**0.5 + 1e-6
    progress_delta = dist - next_dist

    # 2. Orientation stability penalty (hinge)
    angle_threshold = 0.3
    angvel_threshold = 0.5
    angle_violation = max(0.0, abs(next_angle) - angle_threshold)
    angvel_violation = max(0.0, abs(next_angvel) - angvel_threshold)
    orientation_penalty = -0.1 * angle_violation - 0.05 * angvel_violation

    # 3. Speed safety penalty (hinge)
    speed_threshold = 0.5
    vx_violation = max(0.0, abs(next_vx) - speed_threshold)
    vy_violation = max(0.0, abs(next_vy) - speed_threshold)
    speed_penalty = -0.05 * (vx_violation + vy_violation)

    # 4. NEW: contact encouragement (dense reward for feet on ground)
    contact_reward = 0.1 * (next_left + next_right)

    total_reward = progress_delta + orientation_penalty + speed_penalty + contact_reward

    components = {
        'progress_delta': progress_delta,
        'orientation_penalty': orientation_penalty,
        'speed_penalty': speed_penalty,
        'contact_reward': contact_reward
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 1000.00 | 128.47 | ✅ |
| 2 | 骨架变化: contact_reward + orientation_penalty + progress_de | — | 960.10 | 144.81 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=144.813937, len=960.100000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[73.665443, 183.494761]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 140.110002 | 98.7% | 98.7% | 77.2% |
| progress_delta | 1.372766 | 1.0% | 1.0% | 100.0% |
| orientation_penalty | -0.188116 | -0.1% | 0.1% | 1.9% |
| speed_penalty | -0.149573 | -0.1% | 0.1% | 3.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: score=144.8, ep_len=960.1, term_rate=5% (1/20). Contact_reward dominates with 98.7% signed share (episode sum mean=140.1). Other components negligible.

**Component Anomalies**: contact_reward >70% share (dominating). orientation_penalty and speed_penalty nearly dead (active rates 1.9%, 3.3%). progress_delta 100% active but only 1.0% share.

**Training Dynamics**: No checkpoint snapshots provided; temporal trends unknown.

**Signal Quality**: Dead gates: orientation/speed penalties rarely triggered. Contact_reward overshadows progress_delta, which provides tiny signal. Missing attractor: low termination suggests agent optimizes contact survival without successful landing.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个 2D 载体轨迹优化问题。主体从一个随机初始位置（上部中央附近）开始，受随机初始推力影响。主要目标是使主体到达并稳定停靠在中央目标平台上，同时尽可能少地使用发动机推力（省燃料）。主体需要学会：精确接近目标、降低线速度与角速度、保持姿态稳定、安全接触平台。次要目标是快速完成和省燃料，但不能与安全着陆冲突。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- 字段含义：
  - obs[0] `x_position`：相对目标垫中心的水平坐标，单位未指定，奖励可用 true
  - obs[1] `y_position`：相对目标垫高度的垂直坐标（正向可能代表高于垫），单位未指定，奖励可用 true
  - obs[2] `x_velocity`：水平线速度，奖励可用 true
  - obs[3] `y_velocity`：垂直线速度，奖励可用 true
  - obs[4] `body_angle`：主体朝向角（弧度，0为直立），奖励可用 true
  - obs[5] `angular_velocity`：角速度，奖励可用 true
  - obs[6] `left_support_contact`：左支撑脚接触标志（1.0=接触，0.0=未接触），奖励可用 true
  - obs[7] `right_support_contact`：右支撑脚接触标志（1.0=接触，0.0=未接触），奖励可用 true

所有维度均可直接或间接用于奖励函数。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - 0：`no_engine` — 不点火任何发动机，无推力
  - 1：`left_orientation_engine` — 点燃左侧姿态发动机，产生向左水平推力及/或旋转力矩（具体推力方向可影响水平速度与姿态角）
  - 2：`main_engine` — 点燃主发动机，产生垂直向上推力（对抗重力），同时可能产生微小力矩
  - 3：`right_orientation_engine` — 点燃右侧姿态发动机，产生向右水平推力及/或旋转力矩

注意：动作空间未描述精确力矩，但结合`body_angle`和`angular_velocity`，左右发动机可能同时影响水平加速度和角加速度。

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：主体非支撑部分撞击地面或与平台碰撞过猛导致坠毁（如角速度/速度过大）
- **horizontal_position_outside_viewport**：水平坐标超出视口范围（视为出界失败）
- **body_not_awake_or_settled**：主体进入“静止”或“稳定着陆”状态（可能包含成功着陆或长期静止）——这可能是成功着陆的主要终止触发器

没有显式的成功或失败标志。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: {}（终端返回空的info字典）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用。推断成功/失败只能通过观测信号组合与终止事件进行（derived_possible）：
  - 推断成功：终止时 `left_support_contact == 1 and right_support_contact == 1`，同时 `|x_position|` 和 `|y_position|` 接近0，`|x_velocity|`、`|y_velocity|`、`|body_angle|`、`|angular_velocity|` 均低于较小阈值。
  - 推断失败：终止时上述条件不满足，例如水平出界、或仅单脚接触、或角度/速度过大等。

## 7. 可用于奖励函数的信号
- **position**：x_position, y_position（相对目标垫中心坐标，可直接计算到目标(0,0)的距离）
- **velocity**：x_velocity, y_velocity
- **orientation**：body_angle, angular_velocity
- **contact**：left_support_contact, right_support_contact
- **action/engine**：当前 action（可用于燃料消耗惩罚，但无法知道推力大小，只能视为开关）
- **其他**：可从 next_obs 与 obs 构造差值（如 delta 位置、速度变化、角度变化），推断稳定性。

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
| 1 | orientation_penalty + progress_delta + speed_penalty | 128.47 | 128.47 | 0.00 | 1000.00 | orientation_penalty=-0.001 progress_delta=0.002 speed_penalty=-0.002 | new_best |
| 2 | contact_reward + orientation_penalty + progress_delta + speed_penalty | 144.81 | 144.81 | 0.00 | 960.10 | contact_reward=0.129 orientation_penalty=-0.001 progress_delta=0.002 speed_penalty=-0.001 | new_best |

```
