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
# ⚠️ REBUILD MODE
系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。
参考 #6 完整公式算子库选新的主信号框架，基于 #3 累积记录避开已失败的路径。
不要受上一轮代码结构约束。


# 1. Search objective
- target_score: 200.000000
- current_score: 124.271173
- gap_to_target: 75.728827
- target_achievement_ratio: 62.136%

# 2. 上一轮奖励函数代码（该轮得分: 124.271173）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 1.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (event_based landing bonus)
    #   Reward only the moment both legs first touch the pad.
    #   No continuing reward for staying on the pad.
    # ------------------------------------------------------------
    legs_touching_now = (obs[6] > 0.5) and (obs[7] > 0.5)
    legs_touching_next = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    if (not legs_touching_now) and legs_touching_next:
        safe_contact_reward = 20.0
    else:
        safe_contact_reward = 0.0

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return total_reward, components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 965.45 | 144.17 | ✅ |
| 2 | 骨架变化: orientation_stability_reward + progress_reward + s | — | 604.20 | 124.27 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. Subagent 行为诊断报告

## 1. 行为事实
- 得分：score=124.271173，range=[-87.308558, 244.895182]
- 终止方式：terminated 13/20，truncated 7/20
- 回合长度：len=604.20
- Checkpoint 轨迹：无数据（本轮未提供 checkpoint 评估）

## 2. 骨架演进表
| iter | score | len | term | safe_contact_ep_sum | safe_contact_active | orientation_stability_ep_sum | orientation_stability_active | progress_ep_sum | progress_active | velocity_damping_ep_sum | velocity_damping_active |
|------|--------|---------|------|---------------------|---------------------|------------------------------|------------------------------|-----------------|-----------------|-------------------------|-------------------------|
| 1    | 144.17 | 965.45  | 1/20 terminated, 19/20 truncated | 312.46              | 65.2%               | -3.95                        | 100.0%                       | 1.99            | 100.0%          | -1.29                   | 100.0%                  |
| 2    | 124.27 | 604.20  | 13/20 terminated, 7/20 truncated | 852.00              | 7.1%                | -9.48                        | 100.0%                       | 1.88            | 99.2%           | -1.44                   | 99.2%                   |

## 3. 数字标记
- **主导**：`safe_contact_reward` magnitude_share=98.5% > 80%，在奖励量级中占绝对主导。
- **淹没**：
  - `orientation_stability_reward` active_rate=100.0% > 90% 且 magnitude_share=1.1% < 3%
  - `progress_reward` active_rate=99.2% > 90% 且 magnitude_share=0.2% < 3%
  - `velocity_damping_reward` active_rate=99.2% > 90% 且 magnitude_share=0.2% < 3%
- **全覆盖惩罚**：
  - `orientation_stability_reward` active_rate=100.0% > 95% 且 signed_share=-1.1% （负）
  - `velocity_damping_reward` active_rate=99.2% > 95% 且 signed_share=-0.2% （负）
- **量级对比**（每步均值 = ep_sum ÷ len，按绝对值）：
  - 最大两个组件：`safe_contact_reward`（1.410）、`orientation_stability_reward`（0.0157）
  - 最小两个组件：`velocity_damping_reward`（0.00238）、`progress_reward`（0.00311）
- **修改变化**（iter=2 − iter=1）：
  - `safe_contact_reward`：ep_sum +539.54，active_rate −58.1 pp
  - `orientation_stability_reward`：ep_sum −5.53，active_rate 0.0 pp
  - `progress_reward`：ep_sum −0.108，active_rate −0.8 pp
  - `velocity_damping_reward`：ep_sum −0.145，active_rate −0.8 pp

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Potential = -(x^2 + y^2), reward = potential_curr - potential_next
    #   Positive when distance to pad decreases, zero when stationary.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 1.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    #   Penalise speed more strongly when near the pad.
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)          # alpha = 2.0
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Keep body upright and angular velocity small.
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (joint_condition_proxy)
    #   Give positive reward when both legs touch the pad,
    #   vehicle is at pad height, and vertical velocity is low.
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    # y position close to zero (pad height)
    proximity_to_pad = max(0.0, 1.0 - abs(next_obs[1]) / 0.2)   # threshold_y = 0.2
    # vertical velocity small
    soft_vel = max(0.0, 1.0 - abs(next_obs[3]) / 0.2)           # threshold_vy = 0.2
    safe_contact_reward = 0.5 * both_contacts * proximity_to_pad * soft_vel

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return total_reward, components
```

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| orientation_stability_reward | -0.019 | -0.030 | -0.011 | 100.0% | 100.0% | +0.0pp |
| progress_reward | 0.005 | 0.008 | +0.003 | 100.0% | 100.0% | +0.0pp |
| safe_contact_reward | 0.278 | 1.037 | +0.759 | 58.4% | 5.2% | -53.2pp |
| velocity_damping_reward | -0.007 | -0.013 | -0.006 | 99.9% | 100.0% | +0.0pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | 144.17 | 124.27 | -19.90 |
| 平均回合长度 | 965.5 | 604.2 | -361.2 |
| 终止回合数 | 1/20 | 13/20 | +12 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=124.271173, len=604.200000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-87.308558, 244.895182]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 852.000000 | 98.5% | 98.5% | 7.1% |
| orientation_stability_reward | -9.478931 | -1.1% | 1.1% | 100.0% |
| progress_reward | 1.877989 | 0.2% | 0.2% | 99.2% |
| velocity_damping_reward | -1.435409 | -0.2% | 0.2% | 99.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
一个 2D 飞行器从画面上方中心附近出发，受初始随机力作用。核心目标是**到达并稳定在中心目标平台上**（即精准着陆并停稳）。任务同时要求**尽可能快**地完成着陆，并且**尽可能少使用发动机推力**。在过程中，智能体需要学会接近目标、降低速度、保持姿态稳定并实现安全接触（避免硬撞击）。

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: (8,)  
- **dtype**: float32（推测）  
- **obs[0]**: `x_position`，水平坐标相对于目标 pad，可用于衡量横向距离，reward_usable: true  
- **obs[1]**: `y_position`，垂直坐标相对于 pad 高度，可用于衡量高度，reward_usable: true  
- **obs[2]**: `x_velocity`，水平线速度，用于速度控制，reward_usable: true  
- **obs[3]**: `y_velocity`，垂直线速度，用于软着陆控制，reward_usable: true  
- **obs[4]**: `body_angle`，机体角度，用于姿态稳定性，reward_usable: true  
- **obs[5]**: `angular_velocity`，角速度，用于姿态稳定，reward_usable: true  
- **obs[6]**: `left_support_contact`，左侧支撑接触标志（0/1），用于判断着陆状态，reward_usable: true  
- **obs[7]**: `right_support_contact`，右侧支撑接触标志（0/1），用于判断着陆状态，reward_usable: true

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  
- **action 0**: `no_engine`，无发动机动作（什么都不做）  
- **action 1**: `left_orientation_engine`，左姿态发动机，提供侧向推力或扭矩  
- **action 2**: `main_engine`，主发动机，提供向上推力  
- **action 3**: `right_orientation_engine`，右姿态发动机，提供另一侧推力或扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` – 可能表示机体不再活跃或已稳定（例如成功着陆后静止），但也可能包含坠落静止等失败情况。**无显式成功标志**。  
- **failure-like termination**:  
  - `crash_or_body_contact` – 机体与地面或障碍物碰撞导致的失败。  
  - `horizontal_position_outside_viewport` – 水平位置超出画面边界，视为失败。  
- **ambiguous termination**: 上述 settled 条件含义模糊，其设计意图是探测着陆稳定，但不能直接作为可靠的成功指标。  
- **truncation**: 未提供任何时间步或步数限制的截断信号。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**  
- `explicit_failure_flag_available`: **false**  
- `allowed_info_fields`: 环境返回空字典，无可用的额外信号。  
- `forbidden_or_uncertain_info_fields`: 所有未声明的 info 字段均不可用，且不应假设存在 `success`、`failure` 或 `termination_reason`。

## 7. 可用于奖励函数的信号
- **position**: `obs[0]`（x_position）、`obs[1]`（y_position），可计算到目标 pad 的距离；当两者趋近于 0 时表示到达。  
- **velocity**: `obs[2]`（x_velocity）、`obs[3]`（y_velocity），可用于惩罚或塑造速度，尤其在接近目标时。  
- **orientation**: `obs[4]`（body_angle）、`obs[5]`（angular_velocity），用于保持竖直与姿态稳定。  
- **contact**: `obs[6]`（left_support_contact）、`obs[7]`（right_support_contact），指示是否已接触环境，用于判断着陆状态与安全性。  
- **action/engine**: `action` 取值 1/2/3 代表发动机点火，可计算燃料消耗代价。  
- **other**: 无其他明确可用的信号。

# 8. Formula Operator Library（完整版，用于 Level 3 重建）
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



# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | orientation_stability_reward + progress_reward + safe_contact_reward + velocity_damping_reward | 144.17 | 144.17 | 0.00 | 965.45 | orientation_stability_reward=-0.019 progress_reward=0.005 safe_contact_reward=0.278 velocity_damping_reward=-0.007 | new_best |
| 2 | orientation_stability_reward + progress_reward + safe_contact_reward + velocity_damping_reward | 124.27 | 144.17 | -19.90 | 604.20 | orientation_stability_reward=-0.030 progress_reward=0.008 safe_contact_reward=1.037 velocity_damping_reward=-0.013 | no_meaningful_improvement |

```
