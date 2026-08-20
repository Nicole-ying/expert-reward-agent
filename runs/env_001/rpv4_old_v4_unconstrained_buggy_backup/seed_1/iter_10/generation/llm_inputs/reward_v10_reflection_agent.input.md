# Duplicate reward retry
The previous generation duplicated iter 4 (runs/env_001/rpv4_old_v4_unconstrained/seed_1/iter_04/generation/reward_v4.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v10 — Revert to v4 skeleton: binary contact + proximity=40.

    Diagnosis from v9:
    - Graduated contact (0.5× single-leg) did NOT improve soft_landing_bonus
      active_rate (3.4% in v9 vs 3.4% in v8). The agent crashes before reaching
      the pad because proximity=35 is too weak to reliably drive approach.
    - v4 (proximity=40, binary contact, landing=4.0) is the ONLY configuration
      that achieved high soft_landing active_rate (23.5%) and reliable landing
      (19/20, score=135.22).
    - v5-v9 all deviated from v4 and all failed: proximity>40→crash diving,
      proximity<40→hovering, graduated contact→no improvement.

    Single semantic change (L2 + L1, same direction):
    - soft_landing_bonus: graduated contact_level → binary both_contact gate.
      Binary gate forces the agent to fully commit to landing before receiving
      the bonus, which v4 proved creates a strong "discover and stay" incentive.
    - proximity_progress: 35.0 → 40.0 (restore v4 value). This is the minimum
      gradient required to reliably reach the pad without causing crash-diving.
    - soft_landing_bonus coefficient: 5.0 → 4.0 (restore v4 value).
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 40.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: binary contact + continuous per-step quality ----------
    # Binary gate: both legs must be firmly on the pad (proven in v4)
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # Quality gates — soft thresholds preserve gradient for learning
    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 4.0 * both_contact * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```

下面是环境描述：
主要目标：控制一个带有主引擎和两个姿态引擎的 2D 车体，从视口顶部中心尽快降落到中央目标垫上并稳定停靠。稳定停靠需要满足位置接近垫中心、速度接近零、姿态保持水平且两个支撑腿安全接触垫面。

次要目标：在确保成功着陆的前提下，尽可能减少引擎使用（节约燃料），同时尽量缩短飞行时间。

不应混淆的目标：该任务不是单纯的速度追踪或跟踪轨迹，也不是平衡存活问题；引擎节油和时间最优是辅助，核心是安全到达目标区域。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_9, score=-102.307853)
# Training Feedback

## Final-policy outcome
score=-102.307853, len=68.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.531851, -73.251336]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_progress | 39.184449 | 88.8% | 92.0% | 100.0% |
| soft_landing_bonus | 2.308037 | 5.2% | 5.2% | 3.4% |
| attitude_penalty | -1.245961 | -2.8% | 2.8% | 100.0% |

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
| -122.3 | -107.4 | -107.1 | -77.8 | -77.9 | -93.6 | -104.8 | -103.1 | -101.6 | -101.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | -2.313 | -1.175 | -1.474 | -0.829 | -1.197 | -1.046 | -1.282 | -1.062 | -1.231 | -1.500 |
| proximity_progress | 39.396 | 39.222 | 39.253 | 38.650 | 39.098 | 39.154 | 39.214 | 39.201 | 39.189 | 39.183 |
| soft_landing_bonus | 1.779 | 2.343 | 2.047 | 2.851 | 2.427 | 2.766 | 2.459 | 2.461 | 2.310 | 2.351 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity_progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| soft_landing_bonus | 3.1% | 3.1% | 3.2% | 4.1% | 3.2% | 3.2% | 3.1% | 3.1% | 3.3% | 3.3% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v9 — Graduated contact gate for soft_landing_bonus.

    Diagnosis from v8:
    - soft_landing_bonus active_rate = 3.4% — far too sparse for reliable learning.
      The binary gate (both legs must contact simultaneously) means the agent almost
      never experiences the landing reward signal.
    - Checkpoint trend shows the agent discovers landing mid-training (cp50-60%)
      but then forgets it (cp70-100%) — classic sparse-reward collapse as exploration
      decays. Agent retreats to safe hovering where proximity_progress gives steady
      ~25/episode without risk.
    - v4 proved the skeleton works (score=135, 19/20 landed) but needed proximity=40
      to force approach. At proximity=30 the approach gradient is too weak.

    Single semantic change (L2):
    - soft_landing_bonus: binary both_contact → graduated contact_level.
      Single-leg contact now earns half credit (0.5×), creating a stepping-stone
      gradient: "touch the pad" → "land with both legs." This directly addresses
      the sparsity problem by increasing the effective active_rate.
    - Proximity nudged 30→35 and bonus coeff 4.0→5.0 to support the same semantic
      direction (making landing more discoverable and rewarding).
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 35.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: graduated contact for progressive learning ----------
    left_contact = 1.0 if next_obs[6] > 0.5 else 0.0
    right_contact = 1.0 if next_obs[7] > 0.5 else 0.0
    contact_level = 0.5 * left_contact + 0.5 * right_contact   # 0.0, 0.5, or 1.0

    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 5.0 * contact_level * velocity_quality * angle_quality

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