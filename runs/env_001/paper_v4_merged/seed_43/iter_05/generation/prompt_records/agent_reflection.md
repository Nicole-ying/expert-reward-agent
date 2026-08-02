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
- current_score: -87.194134
- gap_to_target: 287.194134
- target_achievement_ratio: -43.597%

# 2. 上一轮奖励函数代码（该轮得分: -87.194134）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    left_contact = float(next_obs[6])
    right_contact = float(next_obs[7])

    # ---------- 1. Main learning signal: potential-based shaping ----------
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    speed_obs = (vx * vx + vy * vy) ** 0.5
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5

    alpha = 0.5
    potential_obs = -(dist_obs + alpha * speed_obs)
    potential_next = -(dist_next + alpha * speed_next)
    progress_shaping = potential_next - potential_obs

    # ---------- 2. Landing speed gate (replaces dead angle_hinge) ----------
    # Gate: penalises high speed when close to target (dist < 0.5)
    proximity_factor = max(0.0, 1.0 - dist_next / 0.5)   # 0 at >=0.5, 1 at 0
    # speed_next multiplied by proximity: large number only when fast and close
    speed_cost_input = speed_next * proximity_factor
    landing_speed_gate = 1.0 / (1.0 + 5.0 * speed_cost_input)

    # Apply gate to progress shaping
    shaped_progress = progress_shaping * landing_speed_gate

    # ---------- 3. Efficiency: action penalty ----------
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    # ---------- 4. Landing contact bonus ----------
    contact_sum = left_contact + right_contact
    contact_factor = contact_sum / 2.0
    proximity = max(0.0, 1.0 - dist_next / 0.8)
    landing_contact_reward = 0.2 * contact_factor * proximity

    total_reward = shaped_progress + action_cost + landing_contact_reward

    components = {
        "progress_shaping": progress_shaping,
        "landing_speed_gate": landing_speed_gate,
        "shaped_progress": shaped_progress,
        "action_cost": action_cost,
        "landing_contact_reward": landing_contact_reward
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.30 | -117.88 | ✅ |
| 2 | 骨架变化: action_cost + angle_hinge + danger_penalty + progr | — | 68.35 | -117.48 | ✅ |
| 3 | 骨架变化: action_cost + angle_hinge + landing_contact_reward | — | 68.30 | -122.17 | ❌ |
| 4 | 骨架变化: action_cost + landing_contact_reward + landing_spe | — | 143.70 | -87.19 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-87.194134, len=143.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-225.198024, 27.283910]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_speed_gate | 127.298497 | 96.4% | 96.4% | 100.0% |
| progress_shaping | 1.081386 | 0.8% | 1.2% | 100.0% |
| shaped_progress | 0.872964 | 0.7% | 1.0% | 100.0% |
| landing_contact_reward | 1.260690 | 1.0% | 1.0% | 11.9% |
| action_cost | -0.659000 | -0.5% | 0.5% | 45.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 9/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: score=-87.2, 9/20 episodes terminate early (score<-50); landing_speed_gate dominates reward (96.4% signed share).

**Component Anomalies**: landing_speed_gate: 100% active, 96.4% share; action_cost active 46% but negligible; landing_contact_reward active 12% yet tiny share; other components effectively zero-share.

**Training Dynamics**: no temporal snapshots provided; cannot assess component growth/decay.

**Signal Quality**: landing_speed_gate mean 0.88 but high early-termination rate; shaping terms too weak to guide landing; missing attractor for slow, successful touchdown.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器/着陆器从视口上方出发，尽快降落到画面中央的水平目标垫上并稳定停靠。主目标是精确到达并停稳在目标垫中心（位置误差趋于零，速度接近零，两支撑脚着垫）。次要目标是尽量减少引擎使用（节能），快速完成任务。注意不要与此类任务可能混淆的纯飞行姿态控制、单纯前进速度优化或仅存活不要求停稳的任务混淆。

## 3. 观察空间 observation_space
- **type:** Box  
- **shape:** (8,)  
- **dtype:** 通常为 float64（环境默认），可视为连续浮点数。  

各索引含义：  
- `obs[0]`：`x_position`，飞行器相对目标垫中心的水平距离（向右为正），reward_usable: true  
- `obs[1]`：`y_position`，飞行器相对目标垫高度的垂直距离（向上为正，0 表示与垫面等高），reward_usable: true  
- `obs[2]`：`x_velocity`，水平线速度，reward_usable: true  
- `obs[3]`：`y_velocity`，垂直线速度，reward_usable: true  
- `obs[4]`：`body_angle`，机身倾角（弧度，0 为水平），reward_usable: true  
- `obs[5]`：`angular_velocity`，角速度，reward_usable: true  
- `obs[6]`：`left_support_contact`，左侧支撑脚触地标志（0.0 或 1.0），reward_usable: true  
- `obs[7]`：`right_support_contact`，右侧支撑脚触地标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- **type:** Discrete  
- **n:** 4  
- **动作说明：**  
  - `action 0`：“no_engine” — 所有引擎关闭，无推力。  
  - `action 1`：“left_orientation_engine” — 点燃左侧姿态引擎，产生偏航/旋转力矩。  
  - `action 2`：“main_engine” — 点燃主引擎，产生主体推力（通常向上或沿机身轴线）。  
  - `action 3`：“right_orientation_engine” — 点燃右侧姿态引擎，产生反方向旋转力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled`，三种触发情景：
- **crash_or_body_contact**：飞行器主体（非支撑脚）与地面或环境障碍碰撞，通常表示失败。  
- **horizontal_position_outside_viewport**：飞行器水平超出视口范围，失败。  
- **body_not_awake_or_settled**：物理体进入休眠状态或因稳定停靠而“settled”。根据任务目标，在目标垫上稳定停靠后应触发此条件，属于成功结果；但也可能因坠毁后体僵硬休眠触发，因此需要结合其他观测才能确定是成功还是失败。  

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** false  
- **explicit_failure_flag_available:** false  
- **allowed_info_fields:** `info` 当前为空字典 `{}`，无法直接获得任何结果标志。  
- **forbidden_or_uncertain_info_fields:** 任何未声明的字段（如 `success`、`failure`、`termination_reason` 等）均不可信。  

成功/失败只能通过 **derived_possible** 方式从观测序列中推断：  
- 成功终端（目标垫稳定停靠）：`episode` 结束时，`x_position`≈0, `y_position`≈0, `|x_velocity|` 和 `|y_velocity|` 很小，`left_support_contact`==1, `right_support_contact`==1，且未发生 `horizontal_out` 现象。  
- 坠毁终端：`episode` 结束时，倾角 `|body_angle|` 很大，或 `y_position` 异常低（地面以下），或只有一只脚接触物且位置远离目标垫。  
- 出界终端：`episode` 结束时，`x_position` 绝对值超出合理范围（范围需通过环境运行中观测到的边界估计，如 |x| > 1.5，或从 rollouts 中统计）。

## 7. 可用于奖励函数的信号
- **位置相关：**  
  - `x_position`, `y_position`（可直接计算到目标垫中心的欧氏距离 `dist = sqrt(x² + y²)`）  
  - 可衍生：`dist_to_target`，上一时刻距离与当前距离之差（delta progress）：`progress = dist(obs) - dist(next_obs)`，正值表示靠近。  
- **速度相关：**  
  - `x_velocity`, `y_velocity` 可用于惩罚接近时的剩余动能，或构建稳定条件。  
- **姿态相关：**  
  - `body_angle` 可用于 hinge penalty（防止倾斜过大）；`angular_velocity` 用于抑制快速旋转。  
- **接触信号：**  
  - `left_support_contact`, `right_support_contact` 可判断双脚是否着垫，是成功停靠的必要条件。  
- **动作相关：**  
  - `action` 值可用于计算动作成本（action ≠ 0 时轻微惩罚）。  
- **衍生信号（derived_possible，需与环境边界参数拟合）：**  
  - **终端成功事件：** 当 `terminated` 且 `dist_to_target` 小于阈值 (如 0.1)，速度幅值低于阈值，且 `left_support_contact` 和 `right_support_contact` 均为 1。  
  - **坠毁事件：** 当 `terminated` 且不满足成功条件，同时 `|body_angle|` 过大或 `y_position` 偏离过大。  
  - **出界事件：** 当 `terminated` 且 `x_position` 超出可靠运行范围。

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
| 1 | action_cost + angle_hinge + progress_shaping | -117.88 | -117.88 | 0.00 | 68.30 | action_cost=-0.001 angle_hinge=-0.001 progress_shaping=0.015 | new_best |
| 2 | action_cost + angle_hinge + danger_penalty + progress_shaping | -117.48 | -117.48 | 0.00 | 68.35 | action_cost=-0.002 angle_hinge=-0.001 danger_penalty=-0.002 progress_shaping=0.015 | new_best |
| 3 | action_cost + angle_hinge + landing_contact_reward + progress_shaping | -122.17 | -117.48 | -4.69 | 68.30 | action_cost=-0.001 angle_hinge=-0.001 landing_contact_reward=0.003 progress_shaping=0.015 | no_meaningful_improvement |
| 4 | action_cost + landing_contact_reward + landing_speed_gate + progress_shaping + shaped_progress | -87.19 | -87.19 | 0.00 | 143.70 | action_cost=-0.002 landing_contact_reward=0.007 landing_speed_gate=0.879 progress_shaping=0.014 shaped_progress=0.011 | new_best |

```
