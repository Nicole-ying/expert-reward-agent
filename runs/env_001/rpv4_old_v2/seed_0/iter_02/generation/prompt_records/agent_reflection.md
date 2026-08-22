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
- current_score: 225.882125
- gap_to_target: -25.882125
- target_achievement_ratio: 112.941%

# 2. 上一轮奖励函数代码（该轮得分: 225.882125）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward function for the soft landing / docking task.

    Roles:
      - approach_target: continuous distance reward (bounded, positive)
      - safe_landing: speed penalty near pad + contact bonus for both legs
      - settlement_stabilization: penalty for velocity/angular movement after contact
    """
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # helpers
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    both_contacts = (left_contact > 0.5) and (right_contact > 0.5)

    # component A: approach_target – positive bounded distance reward
    # using 1/(1 + kp*dist) → max 1.0 when exactly on target
    kp = 1.0
    w_progress = 1.0
    progress_reward = w_progress * (1.0 / (1.0 + kp * dist))

    # component B: safe_landing – speed penalty when close to pad, contact bonus
    threshold_speed = 2.0        # apply speed penalty only within this distance
    w_speed = 0.5                # penalty weight for |vx|+|vy|
    if dist < threshold_speed:
        landing_speed_penalty = -w_speed * (abs(x_vel) + abs(y_vel))
    else:
        landing_speed_penalty = 0.0

    # contact bonus: both legs touching, higher when near center
    w_contact = 2.0
    kc = 1.0
    if both_contacts:
        contact_reward = w_contact * (1.0 / (1.0 + kc * dist))
    else:
        contact_reward = 0.0

    # component C: settlement_stabilization – after contact, encourage full stillness
    w_settle = 0.2
    if both_contacts:
        settlement_penalty = -w_settle * (
            abs(x_vel) + abs(y_vel) + abs(angular_vel) + abs(body_angle)
        )
    else:
        settlement_penalty = 0.0

    total_reward = progress_reward + landing_speed_penalty + contact_reward + settlement_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_speed_penalty": landing_speed_penalty,
        "contact_reward": contact_reward,
        "settlement_penalty": settlement_penalty,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. Subagent 行为诊断报告

## 1. 行为事实
- 得分：score=225.88，range=[199.06, 261.87]
- 终止方式：terminated=20/20（全部正常结束），truncated=0/20（无因步数上限截断）
- 回合长度：len=479.70
- Checkpoint 轨迹：无数据（本迭代未提供 Checkpoint 评估）

## 2. 骨架演进表
| iter | score | len | term | progress_reward_ep_sum | progress_reward_active | contact_reward_ep_sum | contact_reward_active | landing_speed_penalty_ep_sum | landing_speed_penalty_active | settlement_penalty_ep_sum | settlement_penalty_active |
|------|-------|-----|------|------------------------|------------------------|-----------------------|-----------------------|-----------------------------|------------------------------|---------------------------|---------------------------|
| 1    | 225.88 | 479.70 | terminated=20/20, truncated=0/20 | 346.76 | 100.0% | 70.21 | 7.8% | -56.49 | 99.8% | -0.33 | 7.8% |

## 3. 数字标记
- **全覆盖惩罚**：landing_speed_penalty 的 active_rate=99.8% (>95%)，signed_share=-11.9%（为负），符合全覆盖惩罚标记条件。
- **量级对比**（每步均值 = episode_sum_mean / len）：
  - 最大两个：progress_reward (0.72)、contact_reward (0.15)
  - 最小两个：settlement_penalty (0.00)、landing_speed_penalty (0.12)（按绝对值）
- 无“主导”组件（magnitude_share 最大值 73.2%，未突破 80%）
- 无“死组件”（最低 active_rate 为 7.8%，未低于 2%）
- 无“淹没”组件（active_rate>90% 的组件中仅 landing_speed_penalty，但其 magnitude_share=11.9%，未低于 3%）
- 无修改变化（仅有单轮历史，无法计算 Δ）

# 5. 组件逐项对比
（第一轮反思，无历史对比）

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=225.882125, len=479.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[199.061371, 261.871767]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 346.757918 | 73.2% | 73.2% | 100.0% |
| contact_reward | 70.211244 | 14.8% | 14.8% | 7.8% |
| landing_speed_penalty | -56.494578 | -11.9% | 11.9% | 99.8% |
| settlement_penalty | -0.333686 | -0.1% | 0.1% | 7.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个 2D 动态着陆/停靠控制问题。主体从接近画面顶部中央的位置出发，并带有随机初始冲量。核心目标是**让主体尽快到达并稳定停靠在画面中央的目标垫上**，同时**尽可能减少引擎推力使用**。自主学习到的策略应当先接近目标，逐渐减速，保持身体姿态稳定，最终轻柔与目标垫接触。注意：本任务不以“最快到达”为唯一指标，也不要求严格控制绝对姿态轨迹，而是以成功稳定着陆为主，燃料效率为次。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推断，实际 Box 默认为 float32)
- obs[0]: x_position，水平位置相对于目标垫的坐标，reward_usable: true
- obs[1]: y_position，垂直位置相对于目标垫高度的坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，主体朝向角度，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左侧支撑点接触标志 (0.0 or 1.0)，reward_usable: true
- obs[7]: right_support_contact，右侧支撑点接触标志 (0.0 or 1.0)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action/action_dim 0: no_engine，无操作/空闲
- action/action_dim 1: left_orientation_engine，点燃左侧姿态引擎
- action/action_dim 2: main_engine，点燃主推进引擎
- action/action_dim 3: right_orientation_engine，点燃右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` 可能表示成功着陆并稳定，但需要结合位置判断，未提供明确成功标签。
- failure-like termination: `crash_or_body_contact`（撞击或身体异常接触）和 `horizontal_position_outside_viewport`（水平离开视口），这两种明显为失败。
- ambiguous termination: `body_not_awake_or_settled` 在目标垫以外也可能发生（如坠地后静止），因此不能直接作为成功信号；是否为成功必须从观测与终止原因中进一步推断，但当前源码未提供终止原因字段。
- truncation: step 返回的第二个布尔值为 `False`，无截断标志，推测环境自然终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （info = {}，无任何字段可用）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用，因为源码返回空字典；禁止假设 info["success"]、info["failure"]、info["termination_reason"] 等。

## 7. 可用于奖励函数的信号
- position: x_position, y_position（到目标垫的相对位置，可构成距离惩罚/奖励）
- velocity: x_velocity, y_velocity（可用于鼓励减速、软着陆）
- orientation: body_angle（可用于稳定朝向）
- contact: left_support_contact, right_support_contact（检测着陆，鼓励两脚同时接触）
- action/engine: 动作选择（离散 ID，可计算燃料消耗）
- other: 无额外信息

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_reward + landing_speed_penalty + progress_reward + settlement_penalty | 225.88 | 225.88 | 0.00 | 479.70 | contact_reward=0.789 landing_speed_penalty=-0.114 progress_reward=0.717 settlement_penalty=-0.010 | target_solved_new_best |

```
