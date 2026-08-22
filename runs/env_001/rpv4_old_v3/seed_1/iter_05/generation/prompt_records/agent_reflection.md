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

## 0. 阅读你的修改自传

在做任何诊断之前，先读 User prompt 的 **#2 你的修改自传**。它记录了你过去每一轮改了什么组件、怎么改的、结果如何。回答三个问题再进入下一步：

1. **你过去改过哪些组件？每个改了几次？** 如果同一个组件已经被改了 4+ 次，你的下一个修改很可能是在重复过去。
2. **哪些修改让分数涨了？哪些让分数跌了？** 如果某个结构（如"近地引导+接触奖励"）曾经带来 +200 分的突破，你后来是否偏离了它？偏离的结果如何？
3. **你是否在同一个决策上来回切换？** 比如加了某机制 → 删掉 → 又加回来 → 又删掉。如果你发现自己在两个方向之间摇摆，说明你需要锁定一个已被证明有效的版本，只在系数层面微调。

读完自传后，如果你发现自己的修改历史有明显的循环或重复模式，在 `evidence` 字段里提及它。

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
- current_score: -111.061357
- gap_to_target: 311.061357
- target_achievement_ratio: -55.531%

# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `soft_landing` → 分数 -214.2 → -109.6（+104.6）✅ 大幅改善
- **你做了什么**：唯一目标组件为 soft_landing，将其从“低空门控二次惩罚”改为“全高度线性惩罚”，同时纳入姿态角度和角速度，使其在全程产生负向激励，引导 agent 学习减速与稳定姿态。
- **你为什么这么做**：加入全程速度/姿态惩罚后，agent 将学会使用主引擎抵消下沉、利用姿态引擎抑制翻滚，从而降低惩罚，并与 goal_approach 共同引导向中心垫缓慢降落，最终提高 score 并可能产生腿接触。
- **回合长度**：len: 182→68

### v3：修改 `crash_avoidance` → 分数 -109.6 → -115.2（-5.6）➖ 小幅退步
- **你做了什么**：唯一目标组件为 `crash_avoidance`，将其替换为 `landing_success`：当双腿接触且满足低速、直立、低角速度条件时，给予一次性正向大奖励。
- **你为什么这么做**：增加 100.0 的着陆成功奖励后，agent 将在 soft_landing 的引导下逐步探索到安全着陆状态，一旦成功即获得巨大正奖励，从而使平均分数由负转正；下一轮将观察到 landing_success 的 active_rate > 0 且 episode_sum_mean > 0，scor...
- **回合长度**：len: 68→68

### v4：修改 `landing_success` → 分数 -115.2 → -111.1（+4.1）➖ 小幅改善
- **你做了什么**：唯一目标组件为 landing_success，将其替换为 contact_reward（每步当 left_contact_next>0.5 且 right_contact_next>0.5 时给予 +1.0），并移除 crash_avoi...
- **你为什么这么做**：连续接触奖励会为 agent 提供直接梯度，促使其尽快放下双腿接触垫面，同时 soft_landing 仍会惩罚高速/倾斜，从而引导它学会在接触后继续减速并保持稳定，最终提高成功着陆率并拉升 score。
- **回合长度**：len: 68→68

## 你的修改轨迹（按组件分组）

- **`soft_landing`**：修改了 **1 次**，分数变化：v2(+105)
- **`crash_avoidance`**：修改了 **1 次**，分数变化：v3(-6)
- **`landing_success`**：修改了 **1 次**，分数变化：v4(+4)

**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 3. 上一轮奖励函数代码（该轮得分: -111.061357）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observations
    x_pos = obs[0]
    y_pos = obs[1]
    # Unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # 1. goal_approach: potential-based shaping (distance to center)
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # 2. soft_landing: full-range velocity, angle, angular velocity penalty
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. contact_reward: continuous per-step reward for having both legs on pad
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    if BOTH_LEGS:
        contact_reward = 1.0
    else:
        contact_reward = 0.0

    total_reward = goal_approach + soft_landing + contact_reward

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "contact_reward": contact_reward,
    }

    return float(total_reward), components
```

# 4. Subagent 行为诊断报告

# 行为事实
- 得分：score = -111.06，区间 [−130.75, −95.06]
- 终止方式：terminated 20/20（所有episode均因达到终止条件结束），truncated 0/20（无episode因步数上限截断）
- 回合长度：平均 len = 68.40 步
- Checkpoint轨迹（部分数据）：训练进度 40% 时官方得分 0.00，80% 时官方得分 0.00，保持持平

# 骨架演进表

| iter | score | len | term | goal_approach_ep_sum | goal_approach_active | soft_landing_ep_sum | soft_landing_active | crash_avoidance_ep_sum | crash_avoidance_active | landing_success_ep_sum | landing_success_active | contact_reward_ep_sum | contact_reward_active |
|------|--------|-------|------|------------------------|----------------------|----------------------|----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| 1 | -214.16 | 182.50 | 20/0 | 0.1524 | 1.00 | 0.0000 | 0.00 | 0.00 | 0.00 | — | — | — | — |
| 2 | -109.58 | 68.45 | 20/0 | 1.1202 | 1.00 | -19.0910 | 1.00 | 0.00 | 0.00 | — | — | — | — |
| 3 | -115.21 | 68.35 | 20/0 | 1.1188 | 1.00 | -20.7435 | 1.00 | — | — | 25.00 | 0.004 | — | — |
| 4 | -111.06 | 68.40 | 20/0 | 1.1204 | 1.00 | -19.2969 | 1.00 | — | — | — | — | 1.15 | 0.017 |

*注：term列格式为“terminated数/truncated数”；“—”表示该组件在本轮奖励函数中不存在。

# 数字标记（基于本轮 iter=4）

- **主导**：soft_landing magnitude_share = 89.3% > 80%
- **全覆盖惩罚**：soft_landing active_rate = 100% > 95% 且 signed_share = −89.3% 为负
- **死组件**：contact_reward active_rate = 1.7% < 2%
- **量级对比（每步均值）**  
  - 最大：contact_reward = 1.15/68.4 ≈ 0.01681，goal_approach = 1.1204/68.4 ≈ 0.01638  
  - 最小：soft_landing = −19.2969/68.4 ≈ −0.2821，goal_approach ≈ 0.01638  
- **修改变化（相对于 iter=3）**  
  - soft_landing：ep_sum Δ = +1.4466，active Δ = 0.00  
  - goal_approach：ep_sum Δ = +0.0016，active Δ = 0.00  
  - landing_success 已移除，contact_reward 为新增组件，无Δ计算

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observations
    x_pos = obs[0]
    y_pos = obs[1]
    # Unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # 1. goal_approach: potential-based shaping (distance to center)
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # 2. soft_landing: full-range velocity, angle, angular velocity penalty
    #    Encourage low speed, upright posture, and slow angular motion.
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. crash_avoidance: unchanged (penalise hard crash below pad without both legs)
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    CRASH_PENALTY = -10.0
    if y_pos_next < -0.2 and not BOTH_LEGS:
        crash_avoidance = CRASH_PENALTY
    else:
        crash_avoidance = 0.0

    total_reward = goal_approach + soft_landing + crash_avoidance

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "crash_avoidance": crash_avoidance,
    }

    return float(total_reward), components
```

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| contact_reward | — (新增) | 0.017 | 新增 | — | 1.7% | — |
| goal_approach | 0.016 | 0.016 | +0.000 | 100.0% | 100.0% | +0.0pp |
| landing_success | 0.536 | — (新增) | 已删除 | 0.5% | — | — |
| soft_landing | -0.290 | -0.281 | +0.009 | 100.0% | 100.0% | 0.0pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -115.21 | -111.06 | +4.15 |
| 平均回合长度 | 68.3 | 68.4 | +0.1 |
| 终止回合数 | 20/20 | 20/20 | +0 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-111.061357, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-130.746822, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | -19.296874 | -89.3% | 89.3% | 100.0% |
| goal_approach | 1.120424 | 5.2% | 5.4% | 100.0% |
| contact_reward | 1.150000 | 5.3% | 5.3% | 1.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：控制一个2D飞行器从视口顶部中央区域出发，精准降落到中央目标垫上，并保持稳定、安全的接触（两条支撑腿同时着垫、机体姿势竖直、速度极低）。  
次目标：在确保安全着陆的前提下，尽可能缩短总耗时并减少引擎燃料消耗。  
不应混淆的目标：不要单纯追求最快到达而牺牲着陆安全性；也不要只追求零燃料而无限盘旋。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64（或 float32，但观察含浮点标志位）
- obs[0]: x_position, 飞行器相对于目标垫的水平坐标 (m)，reward_usable: true
- obs[1]: y_position, 飞行器相对于垫面高度的垂直坐标 (m)，reward_usable: true
- obs[2]: x_velocity, 水平线速度 (m/s)，reward_usable: true
- obs[3]: y_velocity, 垂直线速度 (m/s)，reward_usable: true
- obs[4]: body_angle, 机体偏转角 (rad)，reward_usable: true
- obs[5]: angular_velocity, 角速度 (rad/s)，reward_usable: true
- obs[6]: left_support_contact, 左支撑腿接触标志 (0.0/1.0)，reward_usable: true
- obs[7]: right_support_contact, 右支撑腿接触标志 (0.0/1.0)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: do nothing (no_engine) – 不点火，仅靠惯性运动
- action 1: fire left orientation engine – 产生顺时针力矩，改变姿态
- action 2: fire main engine – 产生向上的推力（同时可能产生微小水平分力？此处按垂直主推力理解）
- action 3: fire right orientation engine – 产生逆时针力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 两腿同时接触目标垫且机体姿态、速度满足稳定条件 → 触发 body_not_awake_or_settled，可能伴随低速度/低角速度，判为成功。
- failure-like termination: ① 机体任何部分与地面或非垫区域接触 (crash_or_body_contact)；② 水平位置超出视口边界 (horizontal_position_outside_viewport)；③ 在未安全着陆的情况下触发休眠 (body_not_awake_or_settled) 也可能失败。
- ambiguous termination: 当 body_not_awake_or_settled 的触发阈值较宽时，可能涵盖成功着陆（两腿着垫、速度低）与卡在边缘/侧翻但速度为零的情况，需由观察进一步区分。
- truncation: 未观察到显式截断，False 表示不使用截断，episode 总是由上述条件终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} (info 为空字典，不可使用)
- forbidden_or_uncertain_info_fields: 任何 info 字段均禁止使用（因为未声明）。

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: current action (动作索引或 one-hot)，可计算是否使用主引擎/姿态引擎
- other: 无

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_avoidance + goal_approach + soft_landing | -214.16 | -214.16 | 0.00 | 182.50 | crash_avoidance=-0.001 goal_approach=0.001 soft_landing=-0.009 | new_best |
| 2 | crash_avoidance + goal_approach + soft_landing | -109.58 | -109.58 | 0.00 | 68.45 | crash_avoidance=-0.002 goal_approach=0.016 soft_landing=-0.282 | new_best |
| 3 | goal_approach + landing_success + soft_landing | -115.21 | -109.58 | -5.63 | 68.35 | goal_approach=0.016 landing_success=0.536 soft_landing=-0.290 | no_meaningful_improvement |
| 4 | contact_reward + goal_approach + soft_landing | -111.06 | -109.58 | -1.49 | 68.40 | contact_reward=0.017 goal_approach=0.016 soft_landing=-0.281 | no_meaningful_improvement |

```
