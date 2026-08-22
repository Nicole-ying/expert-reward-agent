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
- current_score: 165.623437
- gap_to_target: 34.376563
- target_achievement_ratio: 82.812%

# 2. 上一轮奖励函数代码（该轮得分: 165.623437）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 进度奖励：距离减小的量（鼓励靠近目标点）
    progress = current_dist - next_dist
    progress_reward = 2.0 * progress

    # 速度惩罚：抑制冲击速度（二次惩罚）
    velocity_penalty = 0.05 * (next_obs[2] ** 2 + next_obs[3] ** 2)

    # 姿态惩罚：抑制大幅倾斜（二次惩罚）
    angle_penalty = 0.1 * (next_obs[4] ** 2)

    # 软着陆近似奖励：同时满足双腿接触、靠近中心、低速、小角度时给予正向信号
    contact = next_obs[6] * next_obs[7]  # 1.0 仅当双腿都接触
    pos_factor = max(0.0, 1.0 - next_dist / 0.5)
    vel_sum = abs(next_obs[2]) + abs(next_obs[3])
    vel_factor = max(0.0, 1.0 - vel_sum / 0.5)
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.2)
    soft_landing = 0.5 * contact * pos_factor * vel_factor * angle_factor

    total_reward = progress_reward - velocity_penalty - angle_penalty + soft_landing
    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "soft_landing_proxy": soft_landing
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=165.623437, len=979.000000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[126.588596, 241.536025]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 325.849044 | 98.5% | 98.5% | 70.8% |
| progress_reward | 2.791908 | 0.8% | 0.9% | 99.8% |
| velocity_penalty | 1.642290 | 0.5% | 0.5% | 99.8% |
| angle_penalty | 0.428463 | 0.1% | 0.1% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个二维飞行器/着陆器轨迹优化任务。agent 从视窗上方中央附近以随机初始力开始，需要尽快、省油地**到达视窗中央的目标着陆平台，并以安全姿态稳定接触**（即实现软着陆）。  
核心是导航到目标并实现 safe and stable contact，附属优化是节省发动机推力（能量效率）和缩短耗时，但不改变核心目标。  
**不可混淆**：任务不是持续前行（没有前进方向），也不是纯粹的存活（没有存活计时器），而是**定点到达 + 停稳**。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（推测）  

维度说明（索引从 0 开始，均为可用信号，reward_usable 均为 true）：

- **obs[0]**: `x_position` — 飞行器质心相对目标着陆平台的水平坐标（单位未知，相对值）。reward_usable: true  
- **obs[1]**: `y_position` — 飞行器质心相对平台高度的垂直坐标（下正？待确认方向；通常上正，但可通过初始位置和下降过程推断方向）。rewars_usable: true  
- **obs[2]**: `x_velocity` — 水平线速度。reward_usable: true  
- **obs[3]**: `y_velocity` — 垂直线速度。reward_usable: true  
- **obs[4]**: `body_angle` — 机体方向角（弧度）。reward_usable: true  
- **obs[5]**: `angular_velocity` — 角速度。reward_usable: true  
- **obs[6]**: `left_support_contact` — 左支撑腿是否接触平台（布尔化 float: 1.0/0.0）。reward_usable: true  
- **obs[7]**: `right_support_contact` — 右支撑腿是否接触平台。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  

动作含义：
- **action 0**: `no_engine` — 不启动任何发动机（滑行）。
- **action 1**: `left_orientation_engine` — 点燃左定向发动机，产生侧向/旋转力矩，可改变机体角度并小幅移动。
- **action 2**: `main_engine` — 点燃主发动机，产生主要推力（推测在机体坐标系向上或向下，结合角度影响水平和垂直速度）。
- **action 3**: `right_orientation_engine` — 点燃右定向发动机，与左对称，改变旋转和侧向移动。

## 5. step 与终止条件分析
### 5.1 终止模式
环境给出三个终止条件，经抽象后为：
- `crash_or_body_contact` — 飞行器坠毁或身体其他部分（非支撑腿）接触地面/平台，属于 likely failure。
- `horizontal_position_outside_viewport` — 水平坐标超出视窗范围，显然为 failure。
- `body_not_awake_or_settled` — 飞行器“休眠”或已经稳定停靠，这**很可能对应成功软着陆**（双腿接触且速度、角度足够小后触发）。由于任务目标是到达并 settle，该条件可作为 success-like termination。

当前 info 字典为空，无任何 explicit success/failure flag。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: 无（info 为 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段（不可用）

**成功推断路径**（derived_possible）：  
当 episode 终止（terminated=True）且最后一次观测满足：  
 `left_support_contact == 1.0 && right_support_contact == 1.0`  
 并将 `abs(body_angle)`、`|x_velocity|`、`|y_velocity|` 控制在很小阈值内，且 `x_position` 和 `y_position` 接近零，则可认为发生了成功软着陆。  
**失败推断路径**：  
若终止时 `abs(x_position)` 很大（出界），或存在坠毁迹象（极端 body_angle 突变、两腿未接触），可判断为失败。由于缺少身体接触传感器，无法直接获得碰撞信号，角度过陡、速度冲击可作为间接证据。

## 7. 可用于奖励函数的信号
由于 info 不可用，reward 只能依赖 `obs`、`action` 和 `next_obs`。

- **位置**：`obs[0], obs[1]` 和 `next_obs[0], next_obs[1]`  
- **速度**：`obs[2], obs[3]` 和 `next_obs[2], next_obs[3]`  
- **姿态**：`obs[4]` (body_angle) 和 `next_obs[4]`  
- **角速度**：`obs[5]` 和 `next_obs[5]`  
- **接触**：`obs[6], obs[7]` 和 `next_obs[6], next_obs[7]` (双腿接触标志)  
- **动作**：`action` 值（离散 0-3），可用于动作效率惩罚/奖励

**可从观测间接推断的衍生信号**（derived_possible）：  
- 成功率线索：两腿接触 + 小速度 + 小倾角 + 接近零位置 → 可推断成功着陆  
- 坠毁线索：倾角突然超过安全阈值（如 abs(angle)>某一临界值）或速度骤变 → 可推断碰撞

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
| 1 | angle_penalty + progress_reward + soft_landing_proxy + velocity_penalty | 165.62 | 165.62 | 0.00 | 979.00 | angle_penalty=0.001 progress_reward=0.005 soft_landing_proxy=0.252 velocity_penalty=0.006 | new_best |

```
