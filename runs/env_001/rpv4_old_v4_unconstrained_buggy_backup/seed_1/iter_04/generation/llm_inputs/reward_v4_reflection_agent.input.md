下面是环境描述：
主要目标：控制一个带有主引擎和两个姿态引擎的 2D 车体，从视口顶部中心尽快降落到中央目标垫上并稳定停靠。稳定停靠需要满足位置接近垫中心、速度接近零、姿态保持水平且两个支撑腿安全接触垫面。

次要目标：在确保成功着陆的前提下，尽可能减少引擎使用（节约燃料），同时尽量缩短飞行时间。

不应混淆的目标：该任务不是单纯的速度追踪或跟踪轨迹，也不是平衡存活问题；引擎节油和时间最优是辅助，核心是安全到达目标区域。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_3, score=128.439068)
# Training Feedback

## Final-policy outcome
score=128.439068, len=573.600000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-35.507392, 266.659277]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 535.413352 | 91.6% | 91.6% | 5.8% |
| proximity_progress | 32.604367 | 5.6% | 7.6% | 99.3% |
| attitude_penalty | -4.378923 | -0.7% | 0.7% | 100.0% |

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
| -19.6 | -105.8 | 16.1 | 70.2 | 40.6 | 97.3 | 118.6 | 52.0 | 133.0 | 154.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | -1.749 | -0.952 | -5.030 | -4.973 | -6.974 | -3.911 | -5.174 | -6.205 | -3.503 | -4.022 |
| proximity_progress | 29.986 | 10.840 | 24.927 | 27.632 | 23.811 | 31.482 | 32.217 | 29.851 | 33.409 | 34.097 |
| soft_landing_bonus | 5.806 | 1.999 | 260.733 | 210.071 | 246.151 | 1592.086 | 751.083 | 836.787 | 1247.480 | 575.613 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity_progress | 100.0% | 99.8% | 100.0% | 99.6% | 99.7% | 99.4% | 99.0% | 99.8% | 98.9% | 98.8% |
| soft_landing_bonus | 0.4% | 0.2% | 2.7% | 3.0% | 3.6% | 11.4% | 6.7% | 6.7% | 10.0% | 6.1% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v3 — Fixes soft_landing_bonus multi-fire exploit with binary contact threshold.

    Changes from v2:
    - Contact detection: continuous product → thresholded binary (0.5 cutoff)
      Prevents partial transition events when contact values fluctuate.
    - Coefficient: 20.0 → 30.0 to maintain meaningful one-shot landing incentive
      now that the bonus truly fires only once per episode.
    - proximity_progress and attitude_penalty unchanged (both healthy).
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: TRUE one-shot reward on first safe contact ----------
    # Binary threshold prevents exploit from continuous contact-value fluctuations
    was_both_contact = 1.0 if (obs[6] > 0.5 and obs[7] > 0.5) else 0.0
    is_both_contact  = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    new_contact_event = (1.0 - was_both_contact) * is_both_contact  # 1.0 ONLY on 0→1 transition

    # Quality gates — soft thresholds preserve gradient for learning
    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 30.0 * new_contact_event * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```

---

以上是本轮数据。以下信息可通过工具自主查阅：
- `read_memory()` — 全局记忆表，简单记录每一轮迭代iteration的socre，len长度，奖励函数组件组成
- `read_training_feedback(N)` — 任意迭代的详细训练反馈，了解以前的iteration有过什么好的或者坏的行为，可以避免重蹈覆辙
- `read_reward_code(N)` — 任意迭代的奖励函数代码，方便你深入查看比较好的奖励函数的具体设计
- `read_past_reflection(N)` — 任意迭代的诊断和干预记录，可以查看由好到坏或者由坏到好的记录，吸收学习经验
- `get_component_history(name)` — 组件在所有迭代中的演化，能看清楚某一组件的好的或者坏的演化过程
- `read_environment_card()` — 环境的观察空间、动作空间、终止条件，用来不确定的时候理解每个输入的观测和动作空间的维度信息
- `read_checkpoint_trend(N)` — 训练过程中各组件的趋势，可以看到从训练过程中看出组件是否发挥作用，这样就不仅只看最终反馈，更全面

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

现在开始调查，分析后输出诊断和代码。

### 输出要求

**1. 诊断报告**（在代码之前）：
- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的学到的行为模式是什么，是不健康的吗？结合 score、episode_length、terminated/truncated 判断。

**2. 完整 Python 代码**（```python ... ``` 代码块）。