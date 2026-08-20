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
- current_score: -24.964255
- gap_to_target: 224.964255
- target_achievement_ratio: -12.482%

# 2. 上一轮奖励函数代码（该轮得分: -24.964255）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, n_angle, n_ang_vel, n_left, n_right = next_obs

    # ---------- 1. Main progress: improvement in Euclidean distance to landing pad ----------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    delta_dist = dist - next_dist                # positive when getting closer
    progress_reward = 2.0 * delta_dist

    # ---------- 2. Attitude safety constraint ----------
    angle_err = abs(n_angle)
    ang_vel_abs = abs(n_ang_vel)
    attitude_penalty = -0.5 * (angle_err**2 + (0.5 * ang_vel_abs)**2)

    # ---------- 3. Landing approach reward (continuous multi-factor, replaces dead success_reward) ----------
    prox = max(0.0, 1.0 - next_dist / 5.0)
    upright = max(0.0, 1.0 - angle_err / 0.5)
    speed = (nvx**2 + nvy**2) ** 0.5
    stationary = max(0.0, 1.0 - speed / 1.0)
    contact = (n_left + n_right) / 2.0
    landing_reward = 1.0 * (prox + upright + stationary + contact) / 4.0

    # ---------- Aggregate ----------
    total_reward = progress_reward + attitude_penalty + landing_reward

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": attitude_penalty,
        "landing_reward": landing_reward
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 1000.00 | -36.03 | ✅ |
| 2 | 骨架变化: attitude_penalty + landing_reward + progress_rewar | — | 1000.00 | -24.96 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-24.964255, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-50.610096, 13.490928]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 686.511964 | 99.5% | 99.5% | 100.0% |
| progress_reward | 2.267035 | 0.3% | 0.4% | 100.0% |
| attitude_penalty | -1.082570 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Landing_reward dominates (99.5% signed share, ep_sum_mean=686.5) yet final score=-24.96, 0/20 terminated (all truncated at 1000 steps). Progress and attitude components negligible.

**Component Anomalies**: Landing_reward >99% share, not dead. Attitude_penalty mean=-0.0097, near-zero share. No component >70% magnitude share (landing_reward magnitude share 99.5% = dominating).

**Training Dynamics**: No temporal monitor snapshots provided; drift across checkpoints unknown.

**Signal Quality**: All components active 100%, no dead gates. Landing_reward signal fails to induce terminal landings; episodes never terminate early despite high reward sums.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个2D飞行器精确着陆任务。agent 从视野顶部中心附近出发，带有随机初始扰动。核心目标为安全、稳定地在中心目标平台上着陆——即到达指定相对水平位置 x≈0、高度 y≈0（平台高度），同时保持姿态接近竖直、双腿同时接触平台、速度几乎为零。次要目标为尽快完成着陆，并尽量少使用引擎推力（降低燃料消耗）。不应将存活时间或长时间悬停作为正面目标，也不应单纯最大化水平进度而忽略触地质量和姿态约束。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（或 float64）
- obs[0]: x_position，相对于目标平台水平坐标，reward_usable: true
- obs[1]: y_position，相对于平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，身体朝向角度，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不激活任何引擎（保持当前惯性）
- action 1: left_orientation_engine，点燃左朝向引擎（产生转向或侧向推力）
- action 2: main_engine，点燃主引擎（一般提供向上的推力，但也可能产生旋转分量）
- action 3: right_orientation_engine，点燃右朝向引擎（转向或侧向推力，方向与左相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact：身体（除双腿外的部分）与地面发生碰撞 → 很可能为失败终止（坠毁）。
- horizontal_position_outside_viewport：水平位置超出视野边界 → 失败终止（出界）。
- body_not_awake_or_settled：身体不再活跃（例如静止且未触发其他终止）或满足平台稳定着陆条件（settled） → 若为 settled 则属于成功终止，若仅为不活跃但未满足着陆要求则可能为中立或失败。从任务目标推断，成功着陆的唯一途径就是触发 settled 条件（双腿接触、速度极低、姿态竖直等），因此该条件可视为成功类终止，但需要谨慎对待可能的非成功不活跃情形。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中无 success 字段，原始观测亦无直接标志）
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 为空字典，禁止读取任何字段）
- forbidden_or_uncertain_info_fields: info 内的任何内容均不可用；禁止使用 original_reward

补充推断路径（derived_possible）：
- 成功着陆可通过“终止时的 next_obs 满足两条腿接触、速度接近零、角度接近零、且未发生 crash 或出界”间接判断。
- 坠毁可通过突然的高加速度、body_angle 突变、或 body 位置骤然下降并伴随 contact 信号异常间接推断。
- 出界可从 x_position 超出视野范围推测。

## 7. 可用于奖励函数的信号
位置相关：
- x_position, y_position（均可用于计算到目标点的欧氏距离、水平偏移、高度偏差；可构造距离进步量 delta_distance）
- 可通过 next_obs 与 obs 的 x/y 位置差获取位移方向

速度相关：
- x_velocity, y_velocity（可用于惩罚水平漂移、过大的垂直速度，特别是在接近目标时；可构造速度门控惩罚）
- 速度平方/模长可用于能量惩罚

姿态相关：
- body_angle（用于惩罚偏离竖直的姿态，着陆阶段应接近 0）
- angular_velocity（惩罚过大角速度，防止剧烈旋转）

接触相关：
- left_support_contact, right_support_contact（用于鼓励双腿同时接地，或惩罚单脚/belly着陆）

动作相关：
- action 的语义（no_engine、主引擎、偏转引擎）可用于燃料惩罚（如非零动作施加小惩罚）

间接推断成功的信号（derived_possible）：
- 当 next_obs 满足：双腿接触均为 1、x_velocity≈0、y_velocity≈0、|body_angle| ≈0、x_position≈0、y_position≈0，且当前步未检测到 crash 条件时，可以高置信度推断着陆成功，用于终端奖励。

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
| 1 | attitude_penalty + progress_reward + success_reward | -36.03 | -36.03 | 0.00 | 1000.00 | attitude_penalty=-0.022 progress_reward=0.010 success_reward=1.686 | new_best |
| 2 | attitude_penalty + landing_reward + progress_reward | -24.96 | -24.96 | 0.00 | 1000.00 | attitude_penalty=-0.010 landing_reward=0.643 progress_reward=0.002 | new_best |

```
