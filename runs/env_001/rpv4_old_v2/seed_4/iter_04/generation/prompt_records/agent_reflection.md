# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。先用训练证据解释失败，再选择最小且可验证的干预。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率（旧日志中的`nonzero_rate`）。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。
- 不得仅因任务描述出现“跳跃、着陆、抓取”等语义，就断言对应状态量是缺失奖励职责。新增职责必须有轨迹行为、终止分布、组件激活或历史干预结果支持；证据不足时明确写“未知”，优先保持best主结构并提出最小可证伪修改。
- episode达到时间上限且失败终止很少时，首先判断现有主信号是否已经实现稳定行为、剩余差距是否来自效率或主目标强度；没有行为证据时，不为动作过程本身添加proxy。

# 唯一决策流程

按顺序完成，不能因知识库或工具命中某个变换就跳级。

## 1. 行为与历史诊断

User prompt 里的 #4 Subagent 诊断报告已经替你完成了三件事，从这里出发：

- **骨架演进表** — 并排了同一骨架各轮的 score、len、每个组件的 ep_sum 和 active_rate。纵向扫一眼就能看到："改了 A 那轮，B 塌了，score 跌了"。这是你的核心决策依据。
- **组件健康度** — 每个组件一行判定（健康/死组件/淹没/错位），告诉你哪个信号在驱动 agent、哪个形同虚设。
- **干预建议** — "最需要干预的组件是 X，因为 per-step 量级只有 Y 的 1/50"。

结合 per-step 量级（`ep_sum ÷ len`），回答两个问题：

1. **agent 学到了什么行为？**
2. **哪个组件最值得干预？** 不一定是变化最大的——如果 A 涨了 B 崩了，解释因果链。

一次只选择一个干预目标。current 明显差于 best 时，以 best 代码为基础做最小修改。

## 2. 信号完备性检查

检查当前奖励是否具有任务所需且可达的职责，而不是要求固定组件名称：

- 任务进展或可学习的过程引导；
- 必要的稳定、安全或动作约束；
- 当过程最优不等于任务完成时，能区分联合满足或完成状态的信号。

## 3. 选择干预层级

Subagent 诊断报告会明确告诉你哪个组件最需要干预。**它的推荐是你的起点。** 如果你不同意它的推荐，你必须在 response 里解释为什么不采纳。

先用 `episode_sum_mean ÷ len` 算出每个组件的 **per-step 量级**，对比谁在真正驱动行为。

然后逐组件诊断，每次只改一个组件：

**如果 per-step 量级明显失衡，但数学形态合理** → 调系数。强的组件在产生健康行为（score 涨、terminated 涨）就保留它、提弱组件；强的组件在驱动病态行为就削弱它。

**如果某个组件数学形态本身传达了错误激励** → 改这个组件的公式形式。典型信号：值在训练全程几乎不变（死组件）、乘积塌缩导致 active_rate 接近零、持续状态值无界累积导致 agent 赖着不走、全局惩罚全覆盖压制探索。参考变换方向：

| 证据 | 变换 |
|---|---|
| 事件几乎不触发 | 稀疏 → 连续 |
| 极端值支配 | 无界 → 有界 |
| 占据状态即获奖 | 状态值 → 改善量 |
| 乘积塌缩 | 乘积 → 加性/几何平均 |
| proxy 与外部分数脱钩 | 代理 → 任务完成对齐 |

**如果多轮迭代从未拿到可接受分数，且 Subagent 诊断和 Memory 表确认当前骨架方向错误** → 换主信号框架。选这一步前必须确认：历史上是否已有迭代拿到过不错的分数？如果有——你不是需要换框架，你需要在 best 基础上做上述两种修改之一。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。需要指数形式时使用`2.718281828 ** exponent`，或改用`1/(1+k*x)`、`max(0,1-x/D)`等无需库的bounded表达式。
- 除Level 3重建外，每轮只修改一个目标组件，不顺带调整其他组件。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件，不放total_reward和中间调制器。

# 输出

先用以下固定字段各写一句，不复述输入表格：

1. `evidence`：支持判断的外部结果、组件证据和上一轮结果；
2. `behavior_diagnosis`：策略当前的失败行为；
3. `signal_completeness`：必要职责是否完备、可达；
4. `selected_level`：Level 1、Level 2或Level 3及触发条件；
5. `selected_intervention`：唯一目标组件及具体修改；
6. `falsifiable_hypothesis`：为什么该修改应改善策略；
7. `expected_next_round`：下一轮哪些指标应如何变化；
8. `main_risk`：最可能引入的新漏洞。

然后立即输出完整Python代码。预期必须能被下一轮反馈证伪。
```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: 203.999462
- gap_to_target: -3.999462
- target_achievement_ratio: 102.000%

# 2. 上一轮奖励函数代码（该轮得分: 203.999462）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 1. Target approach: improvement delta ----------
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = dist_curr - dist_next           # positive when agent moves closer
    w_approach = 15.0                          # increased from 5.0 to amplify approach guidance
    reward_approach = w_approach * progress

    # ---------- 2. Soft landing speed: distance-gated velocity penalty ----------
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    sigma = 0.5                                # determines how sharply gate rises near target
    gate = 2.718281828 ** (-dist_next**2 / sigma**2)
    w_speed = 2.0
    reward_speed = -w_speed * speed_next * gate

    # ---------- 3. Stable orientation: quadratic penalties on tilt and rotation ----------
    w_angle = 1.0
    w_angvel = 0.1
    reward_orientation = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # ---------- 4. Contact completion: dual-leg contact gated by position and speed ----------
    left = left_contact_next
    right = right_contact_next
    beta_dist = 10.0
    beta_speed = 10.0
    # Soft proximity factors in [0,1] – near 1 only when distance/speed are near zero
    dist_factor = 1.0 / (1.0 + beta_dist * dist_next**2)
    speed_factor = 1.0 / (1.0 + beta_speed * speed_next**2)
    # Additive contact to give partial reward for single-leg touchdowns
    w_contact = 2.5
    reward_contact = w_contact * (left + right) * dist_factor * speed_factor

    total = reward_approach + reward_speed + reward_orientation + reward_contact

    components = {
        'approach_progress': reward_approach,
        'soft_landing_speed_penalty': reward_speed,
        'orientation_penalty': reward_orientation,
        'contact_completion': reward_contact,
    }
    return float(total), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 890.75 | 81.18 | ✅ |
| 2 | 骨架变化: approach_progress + contact_completion + orientati | — | 517.90 | 215.50 | ✅ |
| 3 | 骨架变化: approach_progress + contact_completion + orientati | — | 569.65 | 204.00 | ❓ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. Subagent 行为诊断报告

## 1. 行为事实
- 得分：score=203.999462，range=[68.170546, 245.278434]
- 终止方式：terminated 18/20，truncated 2/20
- 回合长度：len=569.650000
- Checkpoint 轨迹：无数据

## 2. 骨架演进表
| iter | score | len | term | contact_completion_ep_sum | contact_completion_active | soft_landing_speed_penalty_ep_sum | soft_landing_speed_penalty_active | approach_progress_ep_sum | approach_progress_active | orientation_penalty_ep_sum | orientation_penalty_active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 81.184393 | 890.750000 | 8t/12tr | 478.927441 | 15.8% | -73.990109 | 100.0% | 5.892979 | 100.0% | -5.213019 | 100.0% |
| 2 | 215.504114 | 517.900000 | 18t/2tr | 356.972250 | 18.4% | -60.296305 | 99.8% | 6.596181 | 97.7% | -2.671592 | 100.0% |
| 3 | 203.999462 | 569.650000 | 18t/2tr | 363.115059 | 16.5% | -71.365811 | 99.8% | 19.318184 | 97.8% | -3.520726 | 100.0% |

## 3. 数字标记
- **主导**：无（最大 magnitude_share 为 79.1%，未超过 80%）
- **死组件**：无（所有组件 active_rate > 2%）
- **淹没**：orientation_penalty（active_rate=100.0%，magnitude_share=0.8%<3%）
- **全覆盖惩罚**：soft_landing_speed_penalty（active_rate=99.8%，signed_share=-15.6%）；orientation_penalty（active_rate=100.0%，signed_share=-0.8%）
- **量级对比（每步均值大小）**：
  - 最大两个：contact_completion (0.6374)，soft_landing_speed_penalty (0.1253，绝对值)
  - 最小两个：orientation_penalty (0.0062，绝对值)，approach_progress (0.0339)
- **修改变化（iter_02 → iter_03）**：
  - contact_completion：ep_sum Δ=+6.14，active_rate Δ=-1.9%
  - soft_landing_speed_penalty：ep_sum Δ=-11.07，active_rate Δ=0.0%
  - approach_progress：ep_sum Δ=+12.72，active_rate Δ=+0.1%
  - orientation_penalty：ep_sum Δ=-0.85，active_rate Δ=0.0%

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 1. Target approach: improvement delta ----------
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = dist_curr - dist_next           # positive when agent moves closer
    w_approach = 5.0
    reward_approach = w_approach * progress

    # ---------- 2. Soft landing speed: distance-gated velocity penalty ----------
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    sigma = 0.5                                # determines how sharply gate rises near target
    gate = 2.718281828 ** (-dist_next**2 / sigma**2)
    w_speed = 2.0
    reward_speed = -w_speed * speed_next * gate

    # ---------- 3. Stable orientation: quadratic penalties on tilt and rotation ----------
    w_angle = 1.0
    w_angvel = 0.1
    reward_orientation = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # ---------- 4. Contact completion: dual-leg contact gated by position and speed ----------
    left = left_contact_next
    right = right_contact_next
    beta_dist = 10.0
    beta_speed = 10.0
    # Soft proximity factors in [0,1] – near 1 only when distance/speed are near zero
    dist_factor = 1.0 / (1.0 + beta_dist * dist_next**2)
    speed_factor = 1.0 / (1.0 + beta_speed * speed_next**2)
    # Modified: additive contact to give partial reward for single-leg touchdowns
    w_contact = 2.5
    reward_contact = w_contact * (left + right) * dist_factor * speed_factor

    total = reward_approach + reward_speed + reward_orientation + reward_contact

    components = {
        'approach_progress': reward_approach,
        'soft_landing_speed_penalty': reward_speed,
        'orientation_penalty': reward_orientation,
        'contact_completion': reward_contact,
    }
    return float(total), components
```

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| approach_progress | 0.010 | 0.037 | +0.027 | 99.8% | 100.0% | +0.2pp |
| contact_completion | 1.537 | 1.809 | +0.272 | 44.4% | 50.0% | +5.7pp |
| orientation_penalty | -0.029 | -0.026 | +0.003 | 100.0% | 100.0% | -0.0pp |
| soft_landing_speed_penalty | -0.082 | -0.092 | -0.010 | 99.5% | 100.0% | +0.5pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | 215.50 | 204.00 | -11.50 |
| 平均回合长度 | 517.9 | 569.6 | +51.8 |
| 终止回合数 | 18/20 | 18/20 | +0 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=203.999462, len=569.650000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[68.170546, 245.278434]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_completion | 363.115059 | 79.1% | 79.1% | 16.5% |
| soft_landing_speed_penalty | -71.365811 | -15.6% | 15.6% | 99.8% |
| approach_progress | 19.318184 | 4.2% | 4.5% | 97.8% |
| orientation_penalty | -3.520726 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
任务主体是一个2D飞行器/着陆器，初始从视野上方中央区域出发并带有随机初速。  
**主目标**：尽快飞到中央目标平台并稳定停靠（settle），要求位置精准、速度近乎为零、姿态稳定，并且两个支撑脚均安全接触地面。  
**次目标**：在此过程中尽可能减少引擎使用（节省燃料），并隐含要求快速完成（时间效率）。  
**不应混淆的目标**：燃料效率与快速到达是一对次要权衡，但不应喧宾夺主；安全平稳着陆优先于速度与能耗。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（推断）  
- 各维度定义与可用性：
  - obs[0]: x_position，水平相对目标平台的坐标，reward_usable: true  
  - obs[1]: y_position，垂直相对平台高度的坐标，reward_usable: true  
  - obs[2]: x_velocity，水平线速度，reward_usable: true  
  - obs[3]: y_velocity，垂直线速度，reward_usable: true  
  - obs[4]: body_angle，机体倾角，reward_usable: true  
  - obs[5]: angular_velocity，角速度，reward_usable: true  
  - obs[6]: left_support_contact，左支撑脚接地标志（1.0/0.0），reward_usable: true  
  - obs[7]: right_support_contact，右支撑脚接地标志（1.0/0.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：
  - action 0: no_engine (不做任何推进)  
  - action 1: left_orientation_engine (启动左侧姿态调整引擎)  
  - action 2: main_engine (启动主引擎，提供主要推力)  
  - action 3: right_orientation_engine (启动右侧姿态调整引擎)

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：出现碰撞或非预期的机体接触（典型失败）。  
- **horizontal_position_outside_viewport**：水平位置出界（失败）。  
- **body_not_awake_or_settled**：机体不再活跃/已稳定停靠。此条件**可能代表成功**（如果在目标位置附近、速度低、双足接地），但也可能因为其他原因“静默”失败（如卡在视口外、姿态错误而静止）。需结合状态判断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 返回 {}，不提供任何额外字段）  
- forbidden_or_uncertain_info_fields: info 内所有假设字段（如 “success”, “failure”, “termination_reason” 等）均不存在且不可用。

## 7. 可用于奖励函数的信号
全部来自观测：
- **位置**：x_position, y_position（目标参考零点）  
- **速度**：x_velocity, y_velocity  
- **姿态与角速度**：body_angle, angular_velocity  
- **接触信号**：left_support_contact, right_support_contact  
- **动作**：action（离散值 0-3）

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | approach_progress + contact_completion + orientation_penalty + soft_landing_speed_penalty | 81.18 | 81.18 | 0.00 | 890.75 | approach_progress=0.010 contact_completion=1.878 orientation_penalty=-0.020 soft_landing_speed_penalty=-0.080 | new_best |
| 2 | approach_progress + contact_completion + orientation_penalty + soft_landing_speed_penalty | 215.50 | 215.50 | 0.00 | 517.90 | approach_progress=0.010 contact_completion=1.537 orientation_penalty=-0.029 soft_landing_speed_penalty=-0.082 | target_solved_new_best |
| 3 | approach_progress + contact_completion + orientation_penalty + soft_landing_speed_penalty | 204.00 | 215.50 | -11.50 | 569.65 | approach_progress=0.037 contact_completion=1.809 orientation_penalty=-0.026 soft_landing_speed_penalty=-0.092 | target_solved_no_improvement |

```
