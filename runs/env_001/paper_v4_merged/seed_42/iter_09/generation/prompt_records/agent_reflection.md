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
# ⚠️ REBUILD MODE
系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。
参考 #6 完整公式算子库选新的主信号框架，基于 #3 累积记录避开已失败的路径。
不要受上一轮代码结构约束。


# 1. Search objective
- target_score: 200.000000
- current_score: -103.694628
- gap_to_target: 303.694628
- target_achievement_ratio: -51.847%

# 2. 上一轮奖励函数代码（该轮得分: -103.694628）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 距离目标中心
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    # 速度幅值
    current_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    next_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    
    # 角度
    current_angle_abs = abs(obs[4])
    next_angle_abs = abs(next_obs[4])
    
    # 主进展：距离缩短量（正值表示靠近目标）
    approach_delta = current_dist - next_dist
    
    # 速度改善：减速（正值表示减速）——仅在较接近目标时有效，避免早期抑制探索
    proximity_factor = max(0.0, 1.0 - next_dist / 2.0)
    speed_improvement = (current_speed - next_speed) * proximity_factor
    
    # 角度改善：姿态变正（正值表示角度变小）
    angle_improvement = current_angle_abs - next_angle_abs
    
    # 着陆软代理：距离近、速度低、双腿接触、姿态正（几何平均防塌缩）
    f_dist = max(0.0, 1.0 - next_dist / 0.3)
    f_speed = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.15)
    f_contact = (next_obs[6] + next_obs[7]) / 2.0
    f_angle = max(0.0, 1.0 - next_angle_abs / 0.1)
    landing_proxy = (f_dist * f_speed * f_contact * f_angle) ** 0.25
    
    # 水平边界约束（防止飞出视口）
    x_abs = abs(next_obs[0])
    hinge_x = max(0.0, x_abs - 0.85)
    x_penalty = -1.0 * hinge_x
    
    # 燃料效率（离散动作轻微惩罚）
    fuel_penalty = -0.02 if action != 0 else 0.0
    
    total = (8.0 * approach_delta 
             + 1.0 * speed_improvement 
             + 0.5 * angle_improvement 
             + 3.0 * landing_proxy 
             + x_penalty 
             + fuel_penalty)
    
    components = {
        'approach_delta': float(approach_delta),
        'speed_improvement': float(speed_improvement),
        'angle_improvement': float(angle_improvement),
        'landing_proxy': float(landing_proxy),
        'x_penalty': float(x_penalty),
        'fuel_penalty': float(fuel_penalty)
    }
    return (float(total), components)
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.75 | -92.75 | ✅ |
| 2 | 骨架变化: approach_reward + fuel_penalty + goal_progress + s | — | 661.90 | 194.87 | ✅ |
| 3 | 骨架变化: approach_reward + fuel_penalty + goal_progress + s | — | 896.45 | 163.45 | ❌ |
| 4 | 骨架变化: fuel_penalty + shaping | — | 68.40 | -119.85 | ❌ |
| 5 | 骨架变化: angular_penalty + fuel_penalty + shaping | — | 91.40 | -91.37 | ➖ |
| 6 | 骨架变化: angular_vel_penalty + approach_delta + fuel_penalt | — | 68.35 | -117.48 | ❌ |
| 7 | 骨架变化: approach_delta + fuel_penalty + hinge_angle + hing | — | 68.40 | -116.69 | ❌ |
| 8 | 骨架变化: angle_improvement + approach_delta + fuel_penalty  | — | 87.45 | -103.69 | ➖ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-103.694628, len=87.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-140.921511, -51.973360]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_improvement | 0.475419 | 9.1% | 41.6% | 100.0% |
| approach_delta | 1.242175 | 23.7% | 26.0% | 100.0% |
| angle_improvement | -0.044435 | -0.8% | 14.0% | 100.0% |
| fuel_penalty | -0.546000 | -10.4% | 10.4% | 31.2% |
| landing_proxy | 0.418318 | 8.0% | 8.0% | 0.5% |
| x_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Mean eval reward -103.7, all 20 episodes early-termination (mean len 87.4). Reward components: speed_improvement dominates magnitude (41.6%), approach_delta 26.0%, both positive; fuel_penalty -10.4% signed share but active only 31% of steps; landing_proxy near-zero active (0.5%).

**Component Anomalies**: landing_proxy and x_penalty dead/near-dead (0.5% and 0% active). angle_improvement self-cancelling (mean -0.044, 100% active). No component >70% magnitude share.

**Training Dynamics**: No temporal checkpoint data available.

**Signal Quality**: Desired attractor landing_proxy barely activated; dead x_penalty; fuel_penalty sparse; overall reward scale leads to early crashes.

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

# 7. Formula Operator Library（完整版，用于 Level 3 重建）
# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

---

## 1. Expert Schema 使用规则

- environment_card.md 中的任务画像和可用信号优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role（任务需要什么类型的奖励信号），再选 signal（哪个观测维度承载这个 role），再选 formula operator（用什么数学形式表达），最后写代码。
- 如果某个 role 需要的信号在观测空间中不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- reward_v1 以主学习信号和必要的稳定/安全约束为重点。效率、能耗、复杂门控和动态权重可以在后续迭代中按需加入，但不应因"模板没列"而排除合理的设计。

---

## 2. 信号完备性自查清单

在完成初始设计后，逐一检查以下信号类型是否被覆盖——不是每个任务都需要全部，但每一项的缺失应是有意选择：

- **主进展信号**：agent 朝任务目标前进时是否获得正向反馈？该信号是否每步都有梯度？
- **灾难性失败信号**：是否存在明确的终止惩罚（如摔倒、飞出边界）？如果观测中可推断失败状态，是否给予了足够强的负向信号？
- **效率/代价信号**：连续动作空间中是否有能量消耗或控制代价约束？离散动作空间中是否有不必要的动作惩罚？
- **任务完成信号**：终止条件中是否包含 success-like 条件？相应的观测是否可被用来构造任务完成的软近似信号？
- **健康/稳定约束**：agent 是否因缺少姿态/速度/位置约束而产生不安全行为？

---

## 3. Formula Operator Library

每个算子包含：数学形式、使用条件、适用证据。

### 3.1 dense_state_signal
数学形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2`
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
使用条件：该状态信号每步可观测，且与某项任务职责直接相关。
适用证据：
  - 凸化 → episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小（agent 满足于低水平稳态）。
  - hinge → 约束组件的 active_rate≈100%（全时惩罚）但 terminated 率仍高，说明 agent 在安全范围内也被持续惩罚，需要只在越界时生效的 hinge。
风险：线性正奖励在信号平台期无梯度；凸化权重过大可能诱导极端行为；hinge 的 threshold 需根据环境卡片的观测范围设定。

### 3.2 improvement_delta
数学形式：`old_measure - new_measure`（期望减少时）或 `next_value - current_value`（期望增加时）
使用条件：obs 和 next_obs 中存在可比较的标量度量，该度量沿最优路径应单调变化。
适用证据：有明确的进展度量（位置、距离、高度、角度等），且该度量的变化比瞬时速率更能反映真实进展。
与 dense_state_signal 的选择：如果要鼓励"处于某种好状态"，用 `w * signal`。如果要鼓励"朝好方向改变"，用 delta。delta 的优势是 agent 无法在好状态上停滞不前，必须持续改善。适合：agent 当前的绝对状态值不能完全反映进展（如位置——站在原点不动 vs. 走到终点但位置绝对值可能相同）。
注意：对观测中直接给出的速度信号（如 `horizontal_velocity`）不要做 delta——速度本身已经是变化率。对观测中的位置/角度/距离类信号优先考虑 delta。

### 3.3 potential_based_shaping
数学形式：`potential(next_obs) - potential(obs)`
使用条件：(1) 任务有一个可量化的进展度量（如位置、距离、高度）；(2) 该度量沿最优路径应单调变化；(3) 能从观测中构造一个标量的 potential function。
如何构造 potential：从观测中选择一个在任务完成时达到极值、且沿最优路径单调变化的信号（或信号组合）。potential 的计算只能依赖观测，不能依赖环境内部状态。
与 improvement_delta 的关系：两者数学上等价。potential_based_shaping 的优势在于允许将多个信号编码到一个 potential 中（如同时考虑位置和姿态），而 improvement_delta 通常用于单个度量。
风险：potential 若与任务目标不一致会系统性地误导策略。reward_v1 中如果存在天然的进展度量，优先使用 improvement_delta 的简单形式；当需要组合多个信号构造进展度量时，使用 potential_based_shaping。

### 3.4 quadratic_penalty
数学形式：`-w * error**2` 或 `-w * sum(action_i**2)`
使用条件：约束信号连续可观测，惩罚不应压制主学习信号。用于轻量抑制——需要约束但不至于触发终止的行为。
适用证据：某维度出现高频大幅波动或极端值但未触发终止。
与 hinge 的选择：如果约束有明确的安全边界（如身体倾角超过 X 度必摔），用 hinge（3.1）。如果只是希望"越小越好"没有硬边界（如控制代价、小幅抖动），用 quadratic。
风险：权重过大导致 agent 不敢行动。

### 3.5 soft_health_gate
数学形式：`main_reward * gate_factor`，gate_factor ∈ [0, 1] 在身体状态恶化时平滑衰减。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (safe_bound - current) / margin))`
使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
适用证据：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍 >0——agent 在"先冲后死"，需要在健康恶化时切断主奖励而非额外加罚。
风险：gate 太严格抑制探索；衰减区间应设在"接近危险但尚未终止"的范围内。

### 3.6 terminal_event
数学形式：`if failure_condition: reward = -PENALTY`（硬覆盖 per-step 奖励），或 `if success_condition: reward = +BONUS`
使用条件：(1) 存在可从观测推断的灾难性失败状态（如身体倾角超过阈值 + 接触地面）或任务完成状态；(2) 环境 info 为空因此无法直接读取终止原因。
如何构造：不要依赖 info 字段判断终止原因。可从观测推断：摔倒 → hull_angle 突然偏转 + 身体位置急剧下降；到达终点 → 持续前进中 episode 突然终止（truncated）；出界 → 位置坐标超出有效范围。
适用证据：agent 频繁触发某种终止模式，但当前奖励没有针对该模式提供差异化信号——比如所有终止回合 reward 都一样，agent 无法区分成功和失败。
与 hinge/gate 的区别：hinge 在越界前提供连续梯度，gate 在恶化时衰减主信号。terminal_event 在事件发生的那一刻提供硬信号——没有梯度，但语义明确（"这就是你应该避免/追求的结果"）。

### 3.7 action_efficiency
数学形式：`-w * sum(|action_i|)` 或 `-w * sum(action_i**2)`
使用条件：动作空间 ≥ 2 维连续控制，且任务包含隐含的效率需求（如 locomotion、manipulation）。
适用证据：agent 学会完成任务但动作幅度异常大、能耗高——说明缺效率约束。通常系数较小（主信号 per-step 的 1-5%），避免压制探索。
注意：离散动作空间通常不需要此算子，因为离散动作的选择隐含了代价。首次迭代可不加入，后续迭代若观察到无效动作频繁出现再考虑。

### 3.8 joint_condition_proxy
数学形式：`factor_1 * factor_2 * ...`（每个 factor 为连续 bounded 形式）或 `(f1 + f2 + ...) / n` 或 `(f1 * f2 * ...) ** (1/n)`
使用条件：没有显式 success flag，但有连续信号可构造任务完成的软近似。
适用证据：agent 能在各子条件分别取得进展但无法同时满足。
风险：乘积塌缩（一个 factor→0 则整体→0）；用几何平均或算术平均可缓解。

### 3.9 bounded_signal
数学形式：`x / (1 + abs(x))` 或 `1 / (1 + k * abs(error))` 或 `max(0, 1 - abs(error) / threshold)`
使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。用于压缩极端值而非施加约束。
与 hinge 的区别：bounded 是从两端压缩信号范围，hinge 是只在超出阈值时施加惩罚。如果目标是"值不应超过 X"，用 hinge；如果目标是"值不应该爆炸但无所谓具体范围"，用 bounded。

### 3.10 preview_conditioned_reward
数学形式：`main_reward * preview_factor`，preview_factor 基于观测中能反映**未来状态**的信号（如距离传感器、高度采样、前方地形探测），在不利前景下从 1 平滑衰减到下限。
使用条件：(1) 观测中存在提供前方/未来信息的维度；(2) 该维度可以映射到"前景好/坏"的连续度量；(3) agent 的失败模式与"无法提前调整行为以应对即将到来的状态变化"相关。
如何构造：从提供未来信息的观测中选择一个标量信号，设计一个在安全前景下接近 1、危险前景下接近下限（如 0.3-0.5）的衰减函数。下限不为零以避免完全抑制探索。
适用证据：agent 在相似的瞬时状态下表现差异大（同样的速度/姿态，有时成功有时失败），说明当前状态本身不足以区分好坏——缺少关于"接下来会发生什么"的信息。
与 soft_health_gate 的区别：gate 用当前的**身体状态**乘主奖励（"我已经歪了，别冲了"——被动响应）。preview 用**未来信息**乘主奖励（"前面是坑，别冲了"——主动预判）。两者可以共存：`main_reward * health_gate * preview_factor`。
风险：preview 信号若有噪声会导致主奖励波动；衰减下限设太低会抑制必要探索。

---

## 4. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"定位到合适的算子变换。
以数学语义和训练表现证据为准，不要求组件名完全匹配。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |



# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | fuel_penalty + goal_progress + stable_bonus | -92.75 | -92.75 | 0.00 | 68.75 | fuel_penalty=-0.002 goal_progress=0.016 stable_bonus=0.022 | new_best |
| 2 | approach_reward + fuel_penalty + goal_progress + stable_bonus | 194.87 | 194.87 | 0.00 | 661.90 | approach_reward=0.078 fuel_penalty=-0.007 goal_progress=0.002 stable_bonus=0.714 | new_best |
| 3 | approach_reward + fuel_penalty + goal_progress + stable_bonus | 163.45 | 194.87 | -31.42 | 896.45 | approach_reward=0.079 fuel_penalty=-0.007 goal_progress=0.002 stable_bonus=0.537 | no_meaningful_improvement |
| 4 | fuel_penalty + shaping | -119.85 | 194.87 | -314.72 | 68.40 | fuel_penalty=-0.003 shaping=0.098 | no_meaningful_improvement |
| 5 | angular_penalty + fuel_penalty + shaping | -91.37 | 194.87 | -286.24 | 91.40 | angular_penalty=-0.002 fuel_penalty=-0.005 shaping=0.092 | unsolved_high_achievement_continue_from_best |
| 6 | angular_vel_penalty + approach_delta + fuel_penalty + stable_proxy | -117.48 | 194.87 | -312.34 | 68.35 | angular_vel_penalty=-0.004 approach_delta=0.016 fuel_penalty=-0.002 stable_proxy=0.011 | no_meaningful_improvement |
| 7 | approach_delta + fuel_penalty + hinge_angle + hinge_x + success_proxy | -116.69 | 194.87 | -311.56 | 68.40 | approach_delta=0.016 fuel_penalty=-0.007 hinge_angle=0.004 hinge_x=0.000 success_proxy=0.024 | no_meaningful_improvement |
| 8 | angle_improvement + approach_delta + fuel_penalty + landing_proxy + speed_improvement + x_penalty | -103.69 | 194.87 | -298.56 | 87.45 | angle_improvement=-0.002 approach_delta=0.015 fuel_penalty=-0.007 landing_proxy=0.004 speed_improvement=0.005 | unsolved_high_achievement_continue_from_best |

```
