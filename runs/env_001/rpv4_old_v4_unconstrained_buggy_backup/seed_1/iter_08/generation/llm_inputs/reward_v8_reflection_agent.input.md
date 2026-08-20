下面是环境描述：
主要目标：控制一个带有主引擎和两个姿态引擎的 2D 车体，从视口顶部中心尽快降落到中央目标垫上并稳定停靠。稳定停靠需要满足位置接近垫中心、速度接近零、姿态保持水平且两个支撑腿安全接触垫面。

次要目标：在确保成功着陆的前提下，尽可能减少引擎使用（节约燃料），同时尽量缩短飞行时间。

不应混淆的目标：该任务不是单纯的速度追踪或跟踪轨迹，也不是平衡存活问题；引擎节油和时间最优是辅助，核心是安全到达目标区域。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_7, score=-116.436182)
# Training Feedback

## Final-policy outcome
score=-116.436182, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.774654, -95.638888]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_progress | 44.728742 | 88.6% | 91.6% | 100.0% |
| attitude_penalty | -2.288445 | -4.5% | 4.5% | 100.0% |
| soft_landing_bonus | 1.487163 | 2.9% | 2.9% | 1.8% |
| fuel_penalty | -0.450000 | -0.9% | 0.9% | 4.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
| -120.4 | -122.4 | -125.9 | -117.8 | -113.8 | -121.0 | -119.5 | -118.3 | -119.2 | -115.9 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | -3.308 | -3.418 | -3.231 | -2.680 | -2.421 | -2.549 | -2.378 | -2.362 | -2.233 | -2.089 |
| fuel_penalty | -0.480 | -0.480 | -0.458 | -0.510 | -0.375 | -0.397 | -0.412 | -0.360 | -0.383 | -0.450 |
| proximity_progress | 44.604 | 44.535 | 44.482 | 44.600 | 44.629 | 44.602 | 44.636 | 44.646 | 44.674 | 44.715 |
| soft_landing_bonus | 0.860 | 1.046 | 0.850 | 1.004 | 1.034 | 1.061 | 1.139 | 1.040 | 1.140 | 1.458 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_penalty | 4.6% | 4.7% | 4.5% | 4.9% | 3.6% | 3.9% | 4.0% | 3.5% | 3.7% | 4.4% |
| proximity_progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| soft_landing_bonus | 1.2% | 1.2% | 1.1% | 1.4% | 1.4% | 1.4% | 1.5% | 1.3% | 1.3% | 1.7% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v7 — Adds fuel_penalty to v4 skeleton, encouraging efficient settling.

    Diagnosis from v4-v6 history:
    - v4 (135.22, 19/20 landed) is the only working framework, but soft_landing_bonus
      explodes late in training (cp100%=431.5). Agent bounces and re-lands to farm bonus.
    - v5 (+proximity, -landing) → high-speed crash diving (len=68, all terminated early).
    - v6 (+velocity_penalty) → permanent hovering (len=1000, 0/20 terminated,
      soft_landing_bonus 0% active). Global speed penalty kills landing incentive.
    - The v4 skeleton must be preserved; only v4 achieved reliable landings.

    Single semantic change: NEW fuel_penalty component.
    - Penalizes engine actions (1/2/3) with -0.15 per step.
    - Action 0 (no engine) costs nothing — gravity alone can descend.
    - Once landed and settled, agent can stay on action 0 indefinitely → natural settling incentive.
    - Bouncing requires engine re-fire → bouncing exploit is now taxed per re-fire.
    - Aligns with stated secondary objective: fuel efficiency.

    All other components unchanged from v4:
    - proximity_progress=40.0, soft_landing_bonus=4.0, attitude_penalty same.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 40.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: continuous per-step reward for safe contact ----------
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at theta=0, → 0.17 at |theta|=1

    soft_landing_bonus = 4.0 * both_contact * velocity_quality * angle_quality

    # ---------- fuel_penalty: discourage unnecessary engine use (NEW) ----------
    # action 0 = no engine (free), actions 1/2/3 = engine fire (cost -0.15)
    fuel_penalty = 0.0 if action == 0 else -0.15

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus + fuel_penalty

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "fuel_penalty": fuel_penalty
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