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
- current_score: -115.677399
- gap_to_target: 315.677399
- target_achievement_ratio: -57.839%

# 2. 上一轮奖励函数代码（该轮得分: -115.677399）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重和阈值
    w_progress = 1.0
    w_angle = 0.5
    w_angvel = 0.1
    w_soft_land = 2.0
    w_eff = 0.02

    angle_thresh = 0.3   # rad
    angvel_thresh = 1.0  # rad/s
    max_speed_land = 1.0 # 着陆容许最大合速度
    max_angle_land = 0.5 # 着陆容许最大倾角 rad
    max_safe_vy = 0.5    # 安全下降的垂直速度阈值（m/s）

    # 距离进展（步间距离减少量），加入安全下降门控
    old_dist = (obs[0]**2 + obs[1]**2)**0.5
    new_dist = (next_obs[0]**2 + next_obs[1]**2)**0.5
    delta_dist = old_dist - new_dist   # 正值表示向目标接近

    vy = next_obs[3]                   # 垂直速度
    downward_speed = -vy if vy < 0.0 else 0.0   # 向下速度
    if downward_speed > max_safe_vy:
        overshoot = downward_speed - max_safe_vy
        gate = max(0.0, 1.0 - overshoot / max_safe_vy)
    else:
        gate = 1.0
    progress = w_progress * delta_dist * gate

    # 姿态稳定性（hinge 惩罚）
    angle = next_obs[4]
    angvel = next_obs[5]
    angle_penalty = -w_angle * max(0.0, abs(angle) - angle_thresh)
    angvel_penalty = -w_angvel * max(0.0, abs(angvel) - angvel_thresh)

    # 软着陆奖励（仅在支撑腿接触时有效）
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    any_contact = 1.0 if (left_contact + right_contact) > 0.5 else 0.0

    speed = (next_obs[2]**2 + next_obs[3]**2)**0.5
    speed_factor = 1.0 - min(1.0, speed / max_speed_land)
    angle_factor = 1.0 - min(1.0, abs(angle) / max_angle_land)
    soft_landing_score = speed_factor * angle_factor
    soft_landing = w_soft_land * soft_landing_score * any_contact

    # 发动机使用惩罚（离散动作每次非零动作）
    eff_penalty = -w_eff * (0.0 if action == 0 else 1.0)

    total_reward = progress + angle_penalty + angvel_penalty + soft_landing + eff_penalty

    components = {
        'progress': progress,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'soft_landing': soft_landing,
        'efficiency': eff_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.45 | -113.71 | ✅ |
| 2 | 骨架变化: angle_penalty + angvel_penalty + efficiency + prog | — | 68.35 | -115.68 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-115.677399, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.820934, -92.536457]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 0.777300 | 56.6% | 56.6% | 1.2% |
| angvel_penalty | -0.298458 | -21.7% | 21.7% | 1.0% |
| progress | 0.199347 | 14.5% | 16.7% | 53.5% |
| efficiency | -0.069000 | -5.0% | 5.0% | 5.0% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Mean eval score -115.68, all episodes crash early at ~68 steps. Shaped reward per step +0.0074 but true env reward -1.6156, indicating severe proxy misalignment.

**Component Anomalies**: soft_landing dominates (56.6% signed share) yet active only 1.2% of steps. angle_penalty dead (0 mean, 0% active). angvel_penalty rare (1% active) but significant negative share (-21.7%).

**Training Dynamics**: No checkpoint snapshots; only final policy data. Temporal trends unavailable.

**Signal Quality**: Dead angle_penalty threshold never crossed. Sparse soft_landing reward (1.2% active) likely causes high variance. Positive shaped reward fails to reflect crashing outcome.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化任务。初始时，飞行器从视口顶部中心附近以随机初速度开始运动。核心目标是**尽快且平稳地降落在中央目标垫上**，同时**尽量减少发动机使用**。学习者必须学会接近目标、主动减速、保持机身姿态稳定，并在所有支撑腿（或接触点）安全触垫后停止动作。失败的着陆（碰撞、飞出水平边界）应彻底避免。注意：节能是次目标，不应凌驾于成功着陆之上。

## 3. 观察空间 observation_space
- type: Box（连续向量）
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x_position —— 飞行器中心相对目标垫中心的水平坐标。reward_usable: true
- obs[1]: y_position —— 飞行器底部（或重心）相对垫面的垂直高度。reward_usable: true
- obs[2]: x_velocity —— 水平线速度。reward_usable: true
- obs[3]: y_velocity —— 垂直线速度。reward_usable: true
- obs[4]: body_angle —— 机身倾斜角。reward_usable: true
- obs[5]: angular_velocity —— 角速度。reward_usable: true
- obs[6]: left_support_contact —— 左支撑/腿触地标志（0/1）。reward_usable: true
- obs[7]: right_support_contact —— 右支撑/腿触地标志（0/1）。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（无任何推力，仅靠惯性/重力）  
- action 1: left_orientation_engine（启动左侧姿态发动机，产生逆时针/向右的力矩）  
- action 2: main_engine（启动主发动机，向上产生推力）  
- action 3: right_orientation_engine（启动右侧姿态发动机，产生顺时针/向左的力矩）  

主发动机提供垂直方向推力，姿态发动机用于调整倾斜角度从而改变水平推力方向，典型的**主推力+姿态控制**模式。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  终止条件中的 `body_not_awake_or_settled` 极可能对应成功着陆后的稳定状态：飞行器停在目标垫上，速度降至零，物理引擎判定其静止休眠。
- failure-like termination:  
  - `crash_or_body_contact`：飞行器主体（非支撑腿）触地或与障碍物碰撞。  
  - `horizontal_position_outside_viewport`：水平坐标超出视口范围。
- ambiguous termination:  
  理论上 `body_not_awake_or_settled` 也可能因其它原因（如卡在边界外）触发，但实际环境中通常与成功着陆强关联。
- truncation: 本环境无时间截断（step 返回 truncated 固定为 False）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空字典）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 无任何 info 字段
- 成功/失败只能**间接推断**：  
  当 episode 终止时，通过最终观测（位置、速度、角度、接触标志）综合判断是否成功着陆：
  - 推断成功条件：`|x_position| < ε_x`，`|y_position| < ε_y`，`√(vx²+vy²) < ε_vel`，`|body_angle| < ε_ang`，且至少一个接触标志为 1。
  - 否则为失败。  
  该推断路径标记为 **derived_possible**。

## 7. 可用于奖励函数的信号
**位置与接近度**：
- `next_obs[0]`（x 偏移）：越靠近 0 越好。
- `next_obs[1]`（y 偏移）：越靠近 0 越好，且应在安全范围内。
- 可直接计算距离 `dist = sqrt(x² + y²)` 或步间距离减少量 `delta_dist`。

**速度**：
- `next_obs[2]`（vx）、`next_obs[3]`（vy）：着陆时需极低速度，飞行中可适度引导向下减速。

**姿态**：
- `next_obs[4]`（body_angle）：应保持在安全区间内（例如 ±0.3 rad），防止侧翻。
- `next_obs[5]`（angular_velocity）：应趋近 0。

**接触**：
- `next_obs[6]`（左触地）、`next_obs[7]`（右触地）：成功着陆通常需要双接触（或至少一腿触地且速度达标）。

**动作**：
- `action` 可用来惩罚发动机使用（尤其 punishment for engine actions），但禁止奖励“无动作”。

**终端事件（derived_possible）**：
- 由最终 `next_obs` 推断的成功奖励（大正分）或失败惩罚（负分）。

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
| 1 | angle_penalty + angvel_penalty + efficiency + progress + soft_landing | -113.71 | -113.71 | 0.00 | 68.45 | angle_penalty=-0.001 angvel_penalty=-0.003 efficiency=-0.003 progress=0.016 soft_landing=0.012 | new_best |
| 2 | angle_penalty + angvel_penalty + efficiency + progress + soft_landing | -115.68 | -113.71 | -1.96 | 68.35 | angle_penalty=-0.001 angvel_penalty=-0.003 efficiency=-0.003 progress=0.003 soft_landing=0.012 | no_meaningful_improvement |

```
