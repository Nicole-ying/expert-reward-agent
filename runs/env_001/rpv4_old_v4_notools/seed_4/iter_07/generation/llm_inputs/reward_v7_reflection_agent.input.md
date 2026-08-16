# Duplicate reward retry
The previous generation duplicated iter 6 (runs/env_001/rpv4_old_v4_notools/seed_4/iter_06/generation/reward_v1.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- extract signals ---
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- helper: distance to target pad centre ---
    dist_obs = (x_obs ** 2 + y_obs ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # === 1. progress reward (improvement_delta) ===
    progress = dist_obs - dist_next   # positive when distance shrinks
    w_progress = 50.0
    progress_reward = w_progress * progress

    # === 2. attitude stabilisation (quadratic penalty) ===
    w_angle = 1.0
    w_angvel = 0.5
    attitude_penalty = -w_angle * (abs(angle_next) ** 2) - w_angvel * (angvel_next ** 2)

    # === 3. fuel / engine usage penalty ===
    if action == 0:
        fuel_cost = 0.0
    elif action == 2:   # main engine
        fuel_cost = 0.2
    else:               # left or right orientation engine
        fuel_cost = 0.1
    fuel_reward = -fuel_cost

    # === 4. soft landing component: bonus for dual contact near centre, penalty for high speed ===
    dist_threshold = 0.2
    proximity = max(0.0, 1.0 - dist_next / dist_threshold)   # 1 when dist=0, 0 when dist >= threshold
    dual_contact = left_contact * right_contact               # 1 if both legs touch, else 0
    velocity_mag = (vx_next ** 2 + vy_next ** 2) ** 0.5
    v_safe = 0.5   # maximum allowable landing speed for full bonus; beyond this gives penalty
    speed_factor = 1.0 - velocity_mag / v_safe                # can be negative if speed > v_safe
    w_landing = 40.0
    landing_reward = w_landing * dual_contact * proximity * speed_factor

    # --- total reward ---
    total_reward = progress_reward + attitude_penalty + fuel_reward + landing_reward

    components = {
        "progress": progress_reward,
        "attitude": attitude_penalty,
        "fuel": fuel_reward,
        "landing": landing_reward,
    }

    return float(total_reward), components
```

下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 第一原则：黑盒实验，不是代码重构

你无法预知新奖励函数会引导出什么行为，只能训练完后看结果。
**看到好趋势 → 先理解为什么好，再精准微调。**
大规模重写等于丢弃已有实验证据，重新掷骰子。

**硬约束：如果本轮得分比上一轮改善超过 30 分，你必须先回答：**
上一轮的哪个修改促成了这个改善？证据是什么？
**如果你回答不了，本轮只做 L1，不改数学形态。**

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

### 本轮数据

# 上一轮训练反馈 (iter_6, score=-111.912905)
# Training Feedback

## Final-policy outcome
score=-111.912905, len=69.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-196.679836, 17.571949]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 55.745291 | 66.5% | 69.0% | 100.0% |
| attitude | -14.950602 | -17.8% | 17.8% | 100.0% |
| landing | -5.677077 | -6.8% | 12.4% | 0.6% |
| fuel | -0.610000 | -0.7% | 0.7% | 6.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


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
| -115.7 | -114.6 | -116.1 | -106.2 | -108.8 | -115.8 | -117.5 | -107.0 | -106.7 | -109.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude | -16.088 | -16.173 | -16.651 | -14.538 | -11.806 | -16.104 | -16.562 | -15.085 | -11.259 | -14.551 |
| fuel | -0.090 | -0.250 | -0.425 | -0.435 | -0.380 | -0.375 | -0.305 | -0.430 | -0.335 | -0.615 |
| landing | -5.673 | -5.671 | -5.682 | -5.679 | -5.683 | -5.685 | -5.682 | -5.608 | -5.662 | -5.716 |
| progress | 55.723 | 55.707 | 55.671 | 56.466 | 55.725 | 55.683 | 55.656 | 56.487 | 55.831 | 55.759 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel | 1.3% | 3.6% | 6.1% | 6.1% | 5.3% | 5.2% | 4.4% | 5.3% | 4.3% | 6.9% |
| landing | 0.6% | 0.6% | 0.6% | 0.6% | 0.6% | 0.6% | 0.6% | 0.6% | 0.6% | 0.6% |
| progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- extract signals ---
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- helper: distance to target pad centre ---
    dist_obs = (x_obs ** 2 + y_obs ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # === 1. progress reward (improvement_delta) ===
    progress = dist_obs - dist_next   # positive when distance shrinks
    w_progress = 50.0
    progress_reward = w_progress * progress

    # === 2. attitude stabilisation (quadratic penalty) ===
    w_angle = 1.0
    w_angvel = 0.5
    attitude_penalty = -w_angle * (abs(angle_next) ** 2) - w_angvel * (angvel_next ** 2)

    # === 3. fuel / engine usage penalty ===
    if action == 0:
        fuel_cost = 0.0
    elif action == 2:   # main engine
        fuel_cost = 0.2
    else:               # left or right orientation engine
        fuel_cost = 0.1
    fuel_reward = -fuel_cost

    # === 4. soft landing component: bonus for dual contact near centre, penalty for high speed ===
    dist_threshold = 0.2
    proximity = max(0.0, 1.0 - dist_next / dist_threshold)   # 1 when dist=0, 0 when dist >= threshold
    dual_contact = left_contact * right_contact               # 1 if both legs touch, else 0
    velocity_mag = (vx_next ** 2 + vy_next ** 2) ** 0.5
    v_safe = 0.5   # maximum allowable landing speed for full bonus; beyond this gives penalty
    speed_factor = 1.0 - velocity_mag / v_safe                # can be negative if speed > v_safe
    w_landing = 40.0
    landing_reward = w_landing * dual_contact * proximity * speed_factor

    # --- total reward ---
    total_reward = progress_reward + attitude_penalty + fuel_reward + landing_reward

    components = {
        "progress": progress_reward,
        "attitude": attitude_penalty,
        "fuel": fuel_reward,
        "landing": landing_reward,
    }

    return float(total_reward), components
```

---

以上是本轮数据。现在开始调查，做出一个语义方向上的修改。

### 输出要求

**1. 诊断报告**（在代码之前）：
- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的学到的行为模式是什么，是不健康的吗？结合 score、episode_length、terminated/truncated 判断。
- `signal_completeness`：必要职责是否完备？缺什么信号？
- `selected_level`：L1 / L2 / L3 及选择理由
- `selected_intervention`：目标组件及具体修改内容
- `falsifiable_hypothesis`：为什么该修预计改善策略？
- `expected_next_round`：下一轮策略学习到的行为模式预计是什么？

**2. 完整 Python 代码**（```python ... ``` 代码块）。