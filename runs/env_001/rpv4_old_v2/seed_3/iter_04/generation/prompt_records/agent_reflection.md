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
- current_score: 110.436884
- gap_to_target: 89.563116
- target_achievement_ratio: 55.218%

# 2. 上一轮奖励函数代码（该轮得分: 110.436884）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Encourage moving toward the pad center.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 2.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: landing_proxy (joint_condition_proxy, geometric mean)
    #   Reward simultaneous satisfaction of all landing conditions:
    #   feet contact, centered position, low velocity, low angle,
    #   low angular velocity.
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # bounded proximity factors: 1/(1 + k*|error|)
    proximity_x = 1.0 / (1.0 + 10.0 * abs(next_obs[0]))   # target radius ~0.1
    proximity_y = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))   # pad height ~0
    low_xvel    = 1.0 / (1.0 + 5.0 * abs(next_obs[2]))    # tolerable x speed
    low_yvel    = 1.0 / (1.0 + 5.0 * abs(next_obs[3]))    # tolerable y speed
    low_angle   = 1.0 / (1.0 + 10.0 * abs(next_obs[4]))   # upright angle
    low_angvel  = 1.0 / (1.0 + 2.0 * abs(next_obs[5]))    # angular velocity

    num_factors = 7
    product = (both_contacts * proximity_x * proximity_y *
               low_xvel * low_yvel * low_angle * low_angvel)
    # geometric mean to prevent collapse of a single factor
    landing_proxy_reward = 4.0 * (product ** (1.0 / num_factors))

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Suppress large angles and angular velocities at all times.
    # ------------------------------------------------------------
    angle_penalty = 2.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.5 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: action_penalty (action_efficiency)
    #   Small fuel penalty for any engine use (actions 1,2,3).
    # ------------------------------------------------------------
    action_penalty = -0.03 * (1.0 if action != 0 else 0.0)

    total_reward = (progress_reward +
                    landing_proxy_reward +
                    orientation_stability_reward +
                    action_penalty)

    components = {
        "progress_reward": progress_reward,
        "landing_proxy_reward": landing_proxy_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "action_penalty": action_penalty,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 965.45 | 144.17 | ✅ |
| 2 | 骨架变化: orientation_stability_reward + progress_reward + s | — | 604.20 | 124.27 | ❌ |
| 3 | 骨架变化: action_penalty + landing_proxy_reward + orientatio | — | 528.55 | 110.44 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. Subagent 行为诊断报告

# 训练数据报告

## 1. 行为事实
- 评估得分：110.436884（范围 -51.736246 ~ 219.059345）
- 终止方式：成功着陆 (terminated) 18/20，时间耗尽 (truncated) 2/20
- 回合平均长度：528.55 步
- 无过早终止 (early_terminal) 事件（0/20）
- 无 reward 计算错误（reward_errors=0）
- Checkpoint 评估数据：未提供，无法写出轨迹

## 2. 骨架演进表

| iter | score | len | term | progress_reward_ep_sum | progress_reward_active | orientation_stability_reward_ep_sum | orientation_stability_reward_active | safe_contact_reward_ep_sum | safe_contact_reward_active | velocity_damping_reward_ep_sum | velocity_damping_reward_active | landing_proxy_reward_ep_sum | landing_proxy_reward_active | action_penalty_ep_sum | action_penalty_active |
|------|-------|-----|------|-------------------------|------------------------|-------------------------------------|-------------------------------------|----------------------------|----------------------------|-------------------------------|-------------------------------|-----------------------------|-----------------------------|-----------------------|-----------------------|
| 1    | 144.17 | 965.45 | T:1/20, Tr:19/20 | 1.986 | 100.0% | -3.952 | 100.0% | 312.464 | 65.2% | -1.291 | 100.0% | - | - | - | - |
| 2    | 124.27 | 604.20 | T:13/20, Tr:7/20 | 1.878 | 99.2% | -9.479 | 100.0% | 852.000 | 7.1% | -1.435 | 99.2% | - | - | - | - |
| 3    | 110.44 | 528.55 | T:18/20, Tr:2/20 | 3.644 | 99.8% | -15.551 | 100.0% | - | - | - | - | 359.975 | 24.5% | -15.683 | 98.9% |

（“-”表示该组件在本轮未定义）

## 3. 数字标记

### 主导
- `landing_proxy_reward` magnitude_share = 91.1% (>80%)，主导奖励构成。

### 死组件
- 无死组件（所有组件 active_rate ≥ 24.5%，无低于 2% 的情况）。

### 淹没
- `progress_reward` active_rate = 99.8% (>90%)，magnitude_share = 1.0% (<3%)，被淹没。

### 全覆盖惩罚
- `action_penalty` active_rate = 98.9% (>95%)，signed_share = -4.0%（负），标记为全覆盖惩罚。
- `orientation_stability_reward` active_rate = 100.0% (>95%)，signed_share = -3.9%（负），标记为全覆盖惩罚。

### 量级对比（每步均值 ep_sum/len）
- 最大两个：
  `landing_proxy_reward` ≈ 0.681  
  `progress_reward` ≈ 0.007
- 最小两个（最负）：
  `action_penalty` ≈ -0.0297  
  `orientation_stability_reward` ≈ -0.0294

### 修改变化（iter_2 → iter_3）
- 消失组件：
  `safe_contact_reward` ep_sum Δ = -852.000， active_rate Δ = -7.1 ppt  
  `velocity_damping_reward` ep_sum Δ = +1.435， active_rate Δ = -99.2 ppt
- 新增组件：
  `landing_proxy_reward` ep_sum Δ = +359.975， active_rate Δ = +24.5 ppt  
  `action_penalty` ep_sum Δ = -15.683， active_rate Δ = +98.9 ppt
- 保留组件变化：
  `progress_reward` ep_sum Δ = +1.766， active_rate Δ = +0.6 ppt  
  `orientation_stability_reward` ep_sum Δ = -6.072， active_rate Δ = 0 ppt

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
| action_penalty | — (新增) | -0.020 | 新增 | — | 67.1% | — |
| landing_proxy_reward | — (新增) | 1.192 | 新增 | — | 37.7% | — |
| orientation_stability_reward | -0.030 | -0.078 | -0.048 | 100.0% | 100.0% | -0.0pp |
| progress_reward | 0.008 | 0.021 | +0.013 | 100.0% | 99.9% | -0.1pp |
| safe_contact_reward | 1.037 | — (新增) | 已删除 | 5.2% | — | — |
| velocity_damping_reward | -0.013 | — (新增) | 已删除 | 100.0% | — | — |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | 124.27 | 110.44 | -13.83 |
| 平均回合长度 | 604.2 | 528.5 | -75.7 |
| 终止回合数 | 13/20 | 18/20 | +5 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=110.436884, len=528.550000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-51.736246, 219.059345]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy_reward | 359.974666 | 91.1% | 91.1% | 24.5% |
| action_penalty | -15.682500 | -4.0% | 4.0% | 98.9% |
| orientation_stability_reward | -15.550877 | -3.9% | 3.9% | 100.0% |
| progress_reward | 3.643578 | 0.9% | 1.0% | 99.8% |

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

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | orientation_stability_reward + progress_reward + safe_contact_reward + velocity_damping_reward | 144.17 | 144.17 | 0.00 | 965.45 | orientation_stability_reward=-0.019 progress_reward=0.005 safe_contact_reward=0.278 velocity_damping_reward=-0.007 | new_best |
| 2 | orientation_stability_reward + progress_reward + safe_contact_reward + velocity_damping_reward | 124.27 | 144.17 | -19.90 | 604.20 | orientation_stability_reward=-0.030 progress_reward=0.008 safe_contact_reward=1.037 velocity_damping_reward=-0.013 | no_meaningful_improvement |
| 3 | action_penalty + landing_proxy_reward + orientation_stability_reward + progress_reward | 110.44 | 144.17 | -33.74 | 528.55 | action_penalty=-0.020 landing_proxy_reward=1.192 orientation_stability_reward=-0.078 progress_reward=0.021 | no_meaningful_improvement |

```
