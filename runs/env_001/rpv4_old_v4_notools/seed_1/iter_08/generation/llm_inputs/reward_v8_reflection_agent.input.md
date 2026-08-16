# Duplicate reward retry
The previous generation duplicated iter 7 (runs/env_001/rpv4_old_v4_notools/seed_1/iter_07/generation/reward_v7.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress  : dense signal – progress towards origin (target at (0,0))
      attitude_penalty    : penalty for tilt and angular velocity
      speed_penalty       : penalty for linear velocity (encourages slow approach)
      touchdown_reward    : dense reward for high-quality leg contact during soft landing
      soft_landing_bonus  : event-based – one-time +200 when a stable landing is first achieved
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty ----------
    vx = next_obs[2]
    vy = next_obs[3]
    speed_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # ---------- touchdown reward (leg contact quality) ----------
    # Only reward leg contact when the vehicle is stable (low speed, low tilt, near target)
    leg1 = next_obs[6]
    leg2 = next_obs[7]
    leg_contact = (leg1 + leg2) > 0.5   # at least one leg touching
    if leg_contact:
        dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
        speed = (vx ** 2 + vy ** 2) ** 0.5
        ang = abs(angle)
        # quality: 1.0 for each condition fully satisfied, linearly down to 0
        quality = max(0.0, 1.0 - dist / 0.3) * max(0.0, 1.0 - speed / 0.3) * max(0.0, 1.0 - ang / 0.2)
        touchdown_reward = 10.0 * quality
    else:
        touchdown_reward = 0.0

    # ---------- soft landing bonus (one-time success event) ----------
    def success(ob):
        leg_contact_s = (ob[6] + ob[7]) > 1.5           # both legs must touch
        pos_ok = abs(ob[0]) < 0.3                       # horizontal offset
        speed_ok = (ob[2]**2 + ob[3]**2)**0.5 < 0.3     # low linear speed
        angle_ok = abs(ob[4]) < 0.2                      # low tilt
        return leg_contact_s and pos_ok and speed_ok and angle_ok

    old_success = success(obs)
    new_success = success(next_obs)
    landing_event = (not old_success) and new_success
    soft_landing_bonus = 200.0 if landing_event else 0.0

    # ---------- total ----------
    total_reward = (proximity_progress
                    + attitude_penalty
                    + speed_penalty
                    + touchdown_reward
                    + soft_landing_bonus)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "touchdown_reward": touchdown_reward,
        "soft_landing_bonus": soft_landing_bonus,
    }
    return float(total_reward), components
```

下面是环境描述：
主要目标：控制一个带有主引擎和两个姿态引擎的 2D 车体，从视口顶部中心尽快降落到中央目标垫上并稳定停靠。稳定停靠需要满足位置接近垫中心、速度接近零、姿态保持水平且两个支撑腿安全接触垫面。

次要目标：在确保成功着陆的前提下，尽可能减少引擎使用（节约燃料），同时尽量缩短飞行时间。

不应混淆的目标：该任务不是单纯的速度追踪或跟踪轨迹，也不是平衡存活问题；引擎节油和时间最优是辅助，核心是安全到达目标区域。

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

# 上一轮训练反馈 (iter_7, score=-51.020801)
# Training Feedback

## Final-policy outcome
score=-51.020801, len=72.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-101.254818, -24.178921]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 70.000000 | 71.6% | 71.6% | 0.5% |
| proximity_progress | 22.172338 | 22.7% | 23.7% | 100.0% |
| touchdown_reward | 1.779902 | 1.8% | 1.8% | 0.6% |
| speed_penalty | -1.722602 | -1.8% | 1.8% | 99.9% |
| attitude_penalty | -1.101576 | -1.1% | 1.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 9/20
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
| -115.1 | -62.2 | -37.8 | -10.7 | -86.4 | -69.1 | -59.3 | -39.6 | -63.0 | -55.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | -2.174 | -1.236 | -1.067 | -1.306 | -1.134 | -1.128 | -1.174 | -0.933 | -0.969 | -0.812 |
| proximity_progress | 22.412 | 22.332 | 22.176 | 20.850 | 22.358 | 22.325 | 22.296 | 21.921 | 22.288 | 22.127 |
| soft_landing_bonus | 40.000 | 80.000 | 70.000 | 960.000 | 70.000 | 70.000 | 50.000 | 80.000 | 70.000 | 70.000 |
| speed_penalty | -1.859 | -1.715 | -1.644 | -1.485 | -1.834 | -1.781 | -1.764 | -1.685 | -1.782 | -1.740 |
| touchdown_reward | 1.137 | 1.663 | 1.614 | 127.015 | 2.072 | 1.609 | 1.638 | 1.444 | 1.504 | 1.922 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity_progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| soft_landing_bonus | 0.3% | 0.6% | 0.5% | 2.4% | 0.5% | 0.5% | 0.4% | 0.5% | 0.5% | 0.5% |
| speed_penalty | 99.9% | 99.8% | 99.9% | 100.0% | 99.6% | 99.7% | 99.8% | 99.9% | 99.8% | 99.8% |
| touchdown_reward | 0.4% | 0.6% | 0.5% | 18.9% | 0.7% | 0.6% | 0.6% | 0.5% | 0.5% | 0.6% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress  : dense signal – progress towards origin (target at (0,0))
      attitude_penalty    : penalty for tilt and angular velocity
      speed_penalty       : penalty for linear velocity (encourages slow approach)
      touchdown_reward    : dense reward for high-quality leg contact during soft landing
      soft_landing_bonus  : event-based – one-time +200 when a stable landing is first achieved
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty ----------
    vx = next_obs[2]
    vy = next_obs[3]
    speed_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # ---------- touchdown reward (leg contact quality) ----------
    # Only reward leg contact when the vehicle is stable (low speed, low tilt, near target)
    leg1 = next_obs[6]
    leg2 = next_obs[7]
    leg_contact = (leg1 + leg2) > 0.5   # at least one leg touching
    if leg_contact:
        dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
        speed = (vx ** 2 + vy ** 2) ** 0.5
        ang = abs(angle)
        # quality: 1.0 for each condition fully satisfied, linearly down to 0
        quality = max(0.0, 1.0 - dist / 0.3) * max(0.0, 1.0 - speed / 0.3) * max(0.0, 1.0 - ang / 0.2)
        touchdown_reward = 10.0 * quality
    else:
        touchdown_reward = 0.0

    # ---------- soft landing bonus (one-time success event) ----------
    def success(ob):
        leg_contact_s = (ob[6] + ob[7]) > 1.5           # both legs must touch
        pos_ok = abs(ob[0]) < 0.3                       # horizontal offset
        speed_ok = (ob[2]**2 + ob[3]**2)**0.5 < 0.3     # low linear speed
        angle_ok = abs(ob[4]) < 0.2                      # low tilt
        return leg_contact_s and pos_ok and speed_ok and angle_ok

    old_success = success(obs)
    new_success = success(next_obs)
    landing_event = (not old_success) and new_success
    soft_landing_bonus = 200.0 if landing_event else 0.0

    # ---------- total ----------
    total_reward = (proximity_progress
                    + attitude_penalty
                    + speed_penalty
                    + touchdown_reward
                    + soft_landing_bonus)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "touchdown_reward": touchdown_reward,
        "soft_landing_bonus": soft_landing_bonus,
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