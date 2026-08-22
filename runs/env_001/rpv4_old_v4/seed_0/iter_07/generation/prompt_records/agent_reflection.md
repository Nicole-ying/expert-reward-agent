# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的工作是：根据 user prompt 中的调查指引，自主查阅历史记录，诊断问题，做出一个语义方向上的修改，输出新代码。

# 可用工具

以下工具随时可用，在诊断过程中主动调用：

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action。快速掌握全貌。
- `get_component_history(component_name)` — 某个组件在所有迭代中的 ep_sum_mean 和 active_rate 演化轨迹。
- `read_training_feedback(iteration)` — 某轮训练的详细反馈：精确分数、每个组件的 ep_sum/active_rate/magnitude_share。
- `read_past_reflection(iteration)` — 你过去某轮的诊断和干预记录。动手前查询以避免重复失败的修改。
- `read_reward_code(iteration)` — 某轮的完整奖励函数代码。对比代码差异时使用。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。

# 输出

输出两个部分：

**1. 诊断报告**（在代码之前）：

- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的失败行为
- `signal_completeness`：必要职责是否完备，缺什么信号
- `selected_level`：L1 / L2 / L3 及选择理由
- `selected_intervention`：目标组件及具体修改内容
- `falsifiable_hypothesis`：为什么该修改应改善策略
- `expected_next_round`：下一轮哪些指标应如何变化
- `main_risk`：最可能引入的新问题

**2. 完整 Python 代码**（```python ... ``` 代码块）。

```

## User Prompt

```markdown
第 6 轮迭代训练已经结束，得分 -119.139204（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(6)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

3. **查全局历史避免打转**——用 `read_memory()` 看所有迭代。**关键判断：你是不是在同一个语义功能上反复修改但无突破？**（注意：组件名字可能变了，但语义功能相同——比如 success_landing_bonus 和 success_bonus 都是"着陆成功信号"。）如果某个语义已经改了 2-3 轮都不好，不要再改它了——换一个语义方向。

4. **参考 best code 但要验证**——用 `read_reward_code(best_iter)` 看历史最佳奖励函数的代码。它的哪些组件可能促成了当时的最高分？但不要盲目照搬——结合 best 那轮的 feedback（`read_training_feedback(best_iter)`）看：它的得分分布、终止率、组件 activation 是否健康。best 只是记录最高分，不一定代表正确的引导方式。

5. **Checkpoint 是训练过程的证据**——如果某个组件在 checkpoint 40%→80% 期间 ep_sum 持续为 0 或不变，说明它没有提供有效梯度，agent 在训练中忽略了它。

### 修改原则

- **单一语义修改**：每轮只解决一个问题。通常是一个组件的修改，但如果需要联动（如改一个组件的公式 + 调另一个组件的权重来配合），只要服务于同一语义也算一次修改。
- **L1**：只调系数/权重，不改数学表达式。适用于组件方向对但力度不对。
- **L2**：改一个组件的数学形态（稀疏→连续、状态值→改善量、均匀→高度感知等）。适用于组件的引导逻辑有问题（如均匀惩罚导致无近地减速梯度）。
- **L3**：换主信号框架。仅当多轮迭代从未达标，且历史所有语义方向都已穷尽。
- **⚠️ 组件被 exploit 时必须干预**：如果一个组件 magnitude_share > 90%，即使它刚被激活，也必须修改它——加触发条件限制、加每 episode 次数上限、或收紧阈值。"刚激活所以只做 L1"不适用于 exploit 场景。
- **避免打转**：如果 `read_past_reflection` 显示你之前已经改过同一语义且结果不好，换方向。
- **接近目标时优先微调**：如果当前得分或 best 得分已经接近目标（差距 < 10%），**强烈推荐 L1 微调**（只调系数）。历史尝试中反复出现同一模式：分数已接近目标，却选择 L2 改组件形态，结果越改越差，连续多轮倒退。架构已有基本有效性时，大规模改动风险远大于收益。

### 你需要的信息都在这里

实验记录位于 `rpv4_old_v4/seed_0/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_6, score=-119.139204)
# Training Feedback

## Final-policy outcome
score=-119.139204, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-143.625829, -90.798148]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | -66.671958 | -86.2% | 86.2% | 100.0% |
| vel_penalty | -10.485636 | -13.6% | 13.6% | 99.7% |
| orientation_penalty | -0.144697 | -0.2% | 0.2% | 100.0% |
| fuel_penalty | -0.047250 | -0.1% | 0.1% | 13.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估
# Checkpoint 评估（训练过程中各组件的变化趋势）

每个 checkpoint 在训练到 N% 时用当前模型评估 20 集，记录每个组件的 ep_sum_mean。
**趋势比单点数值更重要：**
- 组件全程为 0 → 死组件，agent 在训练中忽略了它
- 组件从 0 逐渐增长 → 正在引导学习
- 组件骤升 → 可能被 exploit
- 组件骤降 → agent 在学习抑制它

## Overall score trend
| cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -116.6 | -121.4 | -111.0 | -108.1 | -122.7 | -116.0 | -113.8 | -109.4 | -117.6 | -121.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | -67.392 | -66.672 | -66.673 | -67.670 | -66.609 | -67.075 | -66.700 | -66.687 | -66.671 | -66.642 |
| fuel_penalty | -0.000 | -0.121 | -0.094 | -0.094 | -0.055 | -0.106 | -0.108 | -0.090 | -0.060 | -0.044 |
| orientation_penalty | -0.363 | -0.106 | -0.135 | -0.546 | -0.176 | -0.139 | -0.100 | -0.095 | -0.143 | -0.156 |
| vel_penalty | -10.528 | -10.386 | -10.411 | -10.551 | -10.408 | -10.553 | -10.365 | -10.397 | -10.458 | -10.435 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_penalty | 0.1% | 35.2% | 27.4% | 21.0% | 16.2% | 30.2% | 31.6% | 26.1% | 17.5% | 12.7% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vel_penalty | 99.9% | 99.7% | 99.6% | 99.6% | 99.7% | 99.8% | 99.6% | 99.4% | 99.6% | 99.7% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 1. Approach target (main progress) ──
    distance = (x * x + y * y) ** 0.5
    approach_reward = -1.0 * distance

    # ── 2. Soft landing gate ──
    gate = 1.0 / (1.0 + 3.0 * distance)   # close to 1 when near pad

    vel_penalty = -0.3 * gate * (vx * vx + vy * vy + 0.1 * angular_vel * angular_vel)
    orientation_penalty = -0.2 * gate * (angle * angle)

    # ── 3. Fuel efficiency (light penalty) ──
    if action == 2:               # main engine
        fuel_penalty = -0.02
    elif action == 1 or action == 3:  # orientation engines
        fuel_penalty = -0.005
    else:                         # no engine
        fuel_penalty = 0.0

    total_reward = approach_reward + vel_penalty + orientation_penalty + fuel_penalty

    components = {
        'approach_reward': approach_reward,
        'vel_penalty': vel_penalty,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components
```

# Subagent 数据报告
# 训练数据报告 — iter 06

## 1. 行为事实

- **得分**：score=-119.14，range=[-143.63, -90.80]
- **终止方式**：terminated=20/20（全部提前失败），truncated=0/20（无时间截断）
- **回合长度**：len=68.45
- **Checkpoint 轨迹**：
  - 得分趋势：cp10%=-116.6 → cp100%=-121.5（基本稳定在 -120 附近，无显著改善）
  - approach_reward ep_sum：始终在 -67 左右，全程扁平
  - vel_penalty ep_sum：始终在 -10.4 左右，全程扁平
  - orientation_penalty ep_sum：全程接近 0（-0.4 到 -0.1）
  - fuel_penalty ep_sum：全程接近 0（-0.000 到 -0.121），active_rate 从 cp10% 的 0.1% 逐渐下降到 cp100% 的 12.7%

## 2. 骨架演进表

| iter | score | len | term | progress/approach_ep_sum | progress/approach_active | orientation_penalty_ep_sum | orientation_penalty_active | contact/landing_ep_sum | contact/landing_active | vel_damping/vel_penalty_ep_sum | vel_damping/vel_penalty_active | fuel_penalty_ep_sum | fuel_penalty_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -159.49 | 303.35 | T20 | 0.104 | 100% | -0.034 | 100% | 0.000 | 0% | 0.000 | 0% | — | — |
| 2 | -36.38 | 1000.00 | S20 | 1.037 | 100% | -0.302 | 100% | 2704.041 | 100% | -26.778 | 95.3% | — | — |
| 3 | -434.87 | 967.20 | T1/S19 | -1.975 | 100% | -0.874 | 100% | 25.542 | 68.3% | 0.000 | 0% | — | — |
| 4 | -59.19 | 977.90 | T1/S19 | 0.837 | 100% | -0.558 | 100% | 4731.133 | 100% | -21.756 | 92.1% | — | — |
| 5 | -124.22 | 68.35 | T20 | 1.114 | 100% | -1.330 | 100% | 2.316 | 100% | -11.708 | 45.6% | -0.110 | 3.2% |
| 6 | -119.14 | 68.45 | T20 | -66.672 | 100% | -0.145 | 100% | — | — | -10.486 | 99.7% | -0.047 | 13.6% |

**表注**：
- iter 1–5 的进度组件为 `progress_reward`（delta型），iter 6 为 `approach_reward`（状态值型）
- iter 1 的着陆组件为 `contact_reward`（乘积连续型），iter 2–5 为 `landing_proximity_reward`（状态Gaussian型），iter 6 无此组件
- iter 1 的速度组件为 `vel_damping`（门控惩罚型，死组件），iter 2–5 为 `vel_damping`（门控惩罚型），iter 6 为 `vel_penalty`（门控惩罚型，公式形态不同）
- T20=terminated 20/20，S20=truncated 20/20

## 3. 数字标记

- **主导**：`approach_reward` — magnitude_share=86.2%（>80%）
- **全覆盖惩罚**：`approach_reward` — active_rate=100%（>95%）且 signed_share=-86.2%（为负）
- **接近淹没**：`orientation_penalty` — active_rate=100%（>90%）但 magnitude_share=0.2%（<3%），未达到严格淹没阈值但极低
- **死组件**：`fuel_penalty` — active_rate=13.6%（未达到<2%的严格死组件标准，但活性较低）
- **量级对比**（每步均值 = ep_sum ÷ len）：
  - 最大：`approach_reward` = -0.974 / step；`vel_penalty` = -0.153 / step
  - 最小：`orientation_penalty` = -0.002 / step；`fuel_penalty` = -0.001 / step
- **修改变化（iter 5 → iter 6）**：
  - `progress_reward` → `approach_reward`：ep_sum 从 +1.114 → -66.672（Δ=-67.786），active_rate 保持 100%
  - `orientation_penalty`：ep_sum 从 -1.330 → -0.145（Δ=+1.185），active_rate 保持 100%
  - `vel_damping` → `vel_penalty`：ep_sum 从 -11.708 → -10.486（Δ=+1.222），active_rate 从 45.6% → 99.7%（Δ=+54.1pp）
  - `fuel_penalty`：ep_sum 从 -0.110 → -0.047（Δ=+0.063），active_rate 从 3.2% → 13.6%（Δ=+10.4pp）
  - `landing_proximity_reward`：已移除（iter 5 ep_sum=+2.316, active_rate=100%）

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1_progress_reward(改善量) → v2_progress_reward(改善量) → v3_progress_reward(改善量) → v4_progress_reward(改善量) → v5_progress_reward(改善量) → v6_approach_reward(状态值) | -159 → -36 → -435 → -59 → -124 → -119 | 改善量→状态值：1次切换（v5→v6） |
| 着陆/接触信号 | v1_contact_reward(死) → v2_landing_proximity(全主导99%) → v3_landing_proximity(活性68%) → v4_landing_proximity(全主导99.5%) → v5_landing_proximity(小额13.9%) → v6_无 | -159 → -36 → -435 → -59 → -124 → -119 | 组件存活→移除：1次（v5→v6）；形态：乘积连续→状态Gaussian（v1→v2），1次；宽度：宽sigma(2.0)→窄(0.7)→宽(0.7保持)（v2→v4），1次；加权：状态值→进度加权→状态值（v2→v4），2次切换。合计在该功能上发生至少5次设计振荡 |
| 稳定性/约束信号 | v1_vel_damping(死) → v2_vel_damping(门控1.0%) → v3_vel_damping(死) → v4_vel_damping(门控0.5%) → v5_vel_damping(门控70.5% dominate) → v6_vel_penalty(门控13.6%) | -159 → -36 → -435 → -59 → -124 → -119 | 活性：死→活→死→活→主导→次要，5次切换；公式：abs求和→平方求和（v5→v6），1次；惩罚对象：vx+vy → vx²+vy²+0.1*angvel²（v5→v6），1次 |
| 完成/成功代理 | v5_fuel_penalty(稀疏3.2%) → v6_fuel_penalty(稀疏13.6%) | -124 → -119 | 无方向切换，权重降低（-0.2→-0.02主发动机） |
| 燃料效率（交叉功能） | v5_fuel_penalty(稀疏惩罚) → v6_fuel_penalty(稀疏惩罚) | -124 → -119 | 权重从 -0.2/-0.05/-0.0 → -0.02/-0.005/-0.0，降低了10倍，仍在稀疏使用 |

# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v2: replaced dead contact_reward with landing_proximity_reward —
        continuous proximity+stability reward that does not depend on contact flags.
    """
    # Extract relevant signals
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # 1. Progress reward (improvement_delta on Euclidean distance)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # positive when approaching
    w_progress = 1.0
    progress_reward = w_progress * progress

    # 2. Orientation penalty (quadratic_penalty)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Landing proximity reward (continuous soft-landing proxy)
    #    Rewards being close to target, slow, and upright — no contact flags needed.
    #    Gaussian product peaks at the ideal landing state (dist=0, speed=0, angle=0).
    speed = (vx ** 2 + vy ** 2) ** 0.5
    sigma_dist = 2.0
    sigma_speed = 1.0
    sigma_angle = 0.3

    dist_factor = 2.718281828 ** (- (dist_curr ** 2) / (sigma_dist ** 2))
    speed_factor = 2.718281828 ** (- (speed ** 2) / (sigma_speed ** 2))
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))

    w_landing = 3.0
    landing_proximity_reward = w_landing * dist_factor * speed_factor * angle_factor

    # 4. Near-pad velocity damping (hinge / gated penalty)
    dist_thresh = 1.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 at far, 1 at origin
    w_vel_damp = 0.5
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    total_reward = progress_reward + orientation_penalty + landing_proximity_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_proximity_reward": landing_proximity_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components
```

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float (实际为连续值，左/右支撑为 0.0 或 1.0)
- 各维度含义：
  - obs[0] (x_position): 相对于目标着陆垫的水平坐标，reward_usable: true  
  - obs[1] (y_position): 相对于着陆垫高度的垂直坐标，reward_usable: true  
  - obs[2] (x_velocity): 水平线速度，reward_usable: true  
  - obs[3] (y_velocity): 垂直线速度，reward_usable: true  
  - obs[4] (body_angle): 机体角度（方向），reward_usable: true  
  - obs[5] (angular_velocity): 角速度，reward_usable: true  
  - obs[6] (left_support_contact): 左侧支撑接触标志（1.0表示接触，0.0表示未接触），reward_usable: true  
  - obs[7] (right_support_contact): 右侧支撑接触标志，reward_usable: true

所有观测维度均可在自定义奖励中安全使用。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - action 0: no_engine – 什么都不做，滑翔/惯性飞行
  - action 1: left_orientation_engine – 点燃左侧姿态发动机（产生逆时针或顺时针旋转力矩）
  - action 2: main_engine – 点燃主发动机（提供向机头方向的推力，通常向上或减速）
  - action 3: right_orientation_engine – 点燃右侧姿态发动机（与左侧相反的旋转力矩）

注意：该飞行器没有横向直接推进器，水平控制主要通过倾斜机身+主发动机实现。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 中的 “settled” 部分 —— 推测为飞行器稳定停靠在目标着陆垫上（满足速度、角度阈值且支撑接触）
- failure-like termination: crash_or_body_contact（当不属于成功着陆时的身体/地面碰撞）和 horizontal_position_outside_viewport
- ambiguous termination: body_not_awake_or_settled 内部 —— 由于任务描述未区分“settled”与“not awake”，但根据任务目标“reach and settle”可推断出成功情况；需结合接触标志、位置接近度进一步解析
- truncation: 无（step 返回的 truncated 恒为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，终止后无明确标志）
- explicit_failure_flag_available: false 
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均未提供，不可依赖。

成功/失败只能通过下一状态的观测（next_obs）与 done 组合推断：例如，若终止时左右支撑接触均为 1，且 x、y 位置接近 0，速度很小，角度接近 0（或垂直），则可以假定为成功着陆。

## 7. 可用于奖励函数的信号
- 位置信号：
  - next_obs[0] (x_position): 相对于目标着陆垫的水平偏差
  - next_obs[1] (y_position): 相对于着陆垫高度的垂直偏差（越高越正？）
- 速度信号：
  - next_obs[2] (x_velocity): 水平速度（靠近着陆垫时应接近 0）
  - next_obs[3] (y_velocity): 垂直速度（靠近着陆垫时应接近 0 或一个小负值以表示软下降）
- 姿态信号：
  - next_obs[4] (body_angle): 机体角度，用于维持稳定姿态（接近 0 表示竖直）
  - next_obs[5] (angular_velocity): 角速度，用于惩罚大幅旋转
- 接触信号：
  - next_obs[6] (left_support_contact): 左侧支撑接触 (0/1)
  - next_obs[7] (right_support_contact): 右侧支撑接触 (0/1)
  可用于判断着陆状态、两侧接触平衡、软/硬着陆
- 动作/发动机信号：
  - action: 离散动作，可用于惩罚推力使用（action 0 为无推力，action 1/3 为姿态推力，action 2 为主发动机推力）
- 其他衍生物：
  - 距离到目标：sqrt(x_position² + y_position²)
  - 接近速度/方向
  - 着陆瞬间的状态组合（位置、速度、角度）
```
