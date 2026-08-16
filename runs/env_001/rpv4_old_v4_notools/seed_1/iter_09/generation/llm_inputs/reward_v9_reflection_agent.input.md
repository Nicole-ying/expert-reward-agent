# Duplicate reward retry
The previous generation duplicated iter 8 (runs/env_001/rpv4_old_v4_notools/seed_1/iter_08/generation/reward_v8.py). Retry 2: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python

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

# 上一轮训练反馈 (iter_8, score=-24.389467)
# Training Feedback

## Final-policy outcome
score=-24.389467, len=919.600000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-106.995986, 189.654465]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| touchdown_reward | 51.135346 | 59.8% | 59.8% | 2.1% |
| proximity_progress | 20.576698 | 24.1% | 36.0% | 100.0% |
| attitude_penalty | -2.690323 | -3.1% | 3.1% | 100.0% |
| speed_penalty | -0.896170 | -1.0% | 1.0% | 100.0% |
| landing_quality_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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
| -47.9 | -106.6 | -70.6 | -125.7 | -141.6 | -121.6 | 59.1 | 12.5 | -156.5 | -0.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | -1.142 | -1.181 | -1.308 | -0.862 | -0.995 | -1.495 | -4.458 | -2.720 | -1.323 | -2.758 |
| landing_quality_reward | 3.000 | 103.000 | 5.500 | 0.000 | 0.000 | 0.000 | 222.000 | 0.000 | 0.000 | 120.000 |
| proximity_progress | 21.941 | 9.639 | 13.660 | 9.934 | 6.294 | 8.164 | 19.225 | 22.632 | 7.781 | 21.239 |
| speed_penalty | -3.717 | -1.919 | -1.852 | -2.217 | -1.937 | -2.039 | -1.106 | -0.758 | -1.387 | -0.909 |
| touchdown_reward | 8.841 | 272.558 | 163.235 | 0.000 | 0.000 | 18.001 | 628.061 | 56.559 | 0.000 | 153.669 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_quality_reward | 0.4% | 2.4% | 0.1% | 0.0% | 0.0% | 0.0% | 3.1% | 0.0% | 0.0% | 1.3% |
| proximity_progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 99.9% | 100.0% | 100.0% | 100.0% |
| touchdown_reward | 4.0% | 14.2% | 5.8% | 0.0% | 0.0% | 0.9% | 17.1% | 2.4% | 0.0% | 3.5% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress  : dense - progress towards origin (target at (0,0))
      attitude_penalty    : penalty for tilt and angular velocity
      speed_penalty       : stage‑gated penalty on linear velocity (loose when high, strict near ground)
      touchdown_reward    : additive quality reward for any leg contact
      landing_quality_reward : per‑step reward while fully stable on the pad (replaces one‑time bonus)
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty (stage‑gated) ----------
    y = next_obs[1]                   # vertical coordinate, decreasing towards ground
    vx = next_obs[2]
    vy = next_obs[3]
    speed_sq = vx ** 2 + vy ** 2
    if y < 0.5:                       # near ground – strict braking
        speed_penalty = -0.1 * speed_sq
    else:                             # in flight – allow faster descent
        speed_penalty = -0.01 * speed_sq

    # ---------- touchdown reward (leg contact, additive quality) ----------
    leg1 = next_obs[6]
    leg2 = next_obs[7]
    leg_contact = (leg1 + leg2) > 0.5    # at least one leg touching
    if leg_contact:
        dist = new_dist
        speed = speed_sq ** 0.5
        ang = abs(angle)
        # additive factors: each independently rewards closeness to target conditions
        q_dist  = max(0.0, 1.0 - dist / 0.3)
        q_speed = max(0.0, 1.0 - speed / 0.3)
        q_angle = max(0.0, 1.0 - ang / 0.2)
        quality = q_dist + q_speed + q_angle   # range 0..3
        touchdown_reward = 3.0 * quality
    else:
        touchdown_reward = 0.0

    # ---------- landing quality (per‑step stable‑landing reward) ----------
    leg_contact_stable = (next_obs[6] + next_obs[7]) > 1.5      # both legs firmly
    pos_ok   = abs(next_obs[0]) < 0.3
    speed_ok = speed_sq ** 0.5 < 0.3
    angle_ok = abs(angle) < 0.2
    if leg_contact_stable and pos_ok and speed_ok and angle_ok:
        landing_quality_reward = 10.0
    else:
        landing_quality_reward = 0.0

    # ---------- total ----------
    total_reward = (proximity_progress
                    + attitude_penalty
                    + speed_penalty
                    + touchdown_reward
                    + landing_quality_reward)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "touchdown_reward": touchdown_reward,
        "landing_quality_reward": landing_quality_reward,
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