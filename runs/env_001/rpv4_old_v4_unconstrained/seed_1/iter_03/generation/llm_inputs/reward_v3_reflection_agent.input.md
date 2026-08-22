下面是环境描述：
主要目标：控制一个带有主引擎和两个姿态引擎的 2D 车体，从视口顶部中心尽快降落到中央目标垫上并稳定停靠。稳定停靠需要满足位置接近垫中心、速度接近零、姿态保持水平且两个支撑腿安全接触垫面。

次要目标：在确保成功着陆的前提下，尽可能减少引擎使用（节约燃料），同时尽量缩短飞行时间。

不应混淆的目标：该任务不是单纯的速度追踪或跟踪轨迹，也不是平衡存活问题；引擎节油和时间最优是辅助，核心是安全到达目标区域。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_2, score=-125.317262)
# Training Feedback

## Final-policy outcome
score=-125.317262, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-160.183067, -64.030748]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -11.353440 | -46.3% | 46.3% | 100.0% |
| proximity_progress | 5.560546 | 22.7% | 23.5% | 100.0% |
| altitude_reward | 2.422990 | 9.9% | 9.9% | 100.0% |
| attitude_penalty | -2.149320 | -8.8% | 8.8% | 100.0% |
| step_penalty | -1.367000 | -5.6% | 5.6% | 100.0% |
| contact_reward | 1.199342 | 4.9% | 4.9% | 3.7% |
| fuel_penalty | -0.287500 | -1.2% | 1.2% | 1.7% |

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
| -116.0 | -115.5 | -115.4 | -113.8 | -115.8 | -120.9 | -122.5 | -126.0 | -124.6 | -125.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| altitude_reward | 2.589 | 2.582 | 2.582 | 2.588 | 2.680 | 2.441 | 2.447 | 2.420 | 2.418 | 2.423 |
| attitude_penalty | -3.328 | -3.312 | -3.300 | -3.194 | -3.743 | -2.104 | -2.391 | -2.112 | -1.915 | -2.129 |
| contact_reward | 1.648 | 1.664 | 1.680 | 1.670 | 1.635 | 1.422 | 1.327 | 1.256 | 1.366 | 1.200 |
| fuel_penalty | -0.100 | -0.350 | -0.375 | -0.375 | -0.825 | -0.350 | -0.237 | -0.375 | -0.375 | -0.287 |
| proximity_progress | 5.572 | 5.569 | 5.570 | 5.570 | 5.658 | 5.573 | 5.566 | 5.568 | 5.568 | 5.561 |
| speed_penalty | -11.473 | -11.458 | -11.459 | -11.474 | -11.715 | -11.321 | -11.354 | -11.336 | -11.316 | -11.352 |
| step_penalty | -1.399 | -1.398 | -1.398 | -1.399 | -1.421 | -1.370 | -1.371 | -1.366 | -1.366 | -1.367 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| altitude_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_reward | 5.3% | 5.2% | 5.3% | 5.3% | 5.9% | 3.8% | 4.0% | 3.4% | 3.5% | 3.7% |
| fuel_penalty | 0.6% | 2.0% | 2.1% | 2.1% | 4.6% | 2.0% | 1.4% | 2.2% | 2.2% | 1.7% |
| proximity_progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| step_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2 — restructured to fix hovering and landing-sparsity problems.

    Core changes from v1:
    - Added altitude_reward: continuous incentive to descend (fixes hovering).
    - Added fuel_penalty + step_penalty: make lingering near the pad costly.
    - contact_reward now uses continuous n_contact (0/1/2) not binary product,
      so partial contact provides gradient. Per-step value is modest to prevent
      bounce-exploit (fuel+step costs outweigh repeated tiny contact gains).
    - Reduced proximity_progress weight — was 82.9% magnitude share, too dominant.
    - All components balanced so net reward near pad w/o landing is negative.
    """
    # ── current state ──────────────────────────────────────────────
    x  = next_obs[0]          # horizontal position relative to pad center
    y  = next_obs[1]          # vertical position (height above pad)
    vx = next_obs[2]          # horizontal velocity
    vy = next_obs[3]          # vertical velocity
    angle      = next_obs[4]  # body tilt angle
    angular_vel = next_obs[5] # angular velocity
    left_contact  = next_obs[6]  # 1.0 if left leg on ground
    right_contact = next_obs[7]  # 1.0 if right leg on ground

    ox = obs[0]               # previous x
    oy = obs[1]               # previous y

    # ── 1. proximity_progress ──────────────────────────────────────
    # Reward closing distance to the pad centre. Reduced weight vs v1
    # (was 30.0, now 5.0) so it doesn't drown out landing signals.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    proximity_progress = 5.0 * (old_dist - new_dist)

    # ── 2. altitude_reward ─────────────────────────────────────────
    # Continuous reward for being at low height.  Creates the *gradient*
    # to descend that v1 was missing.  Max 0.15/step at y=0.
    altitude_reward = 0.15 / (1.0 + 5.0 * (y ** 2) ** 0.5)

    # ── 3. speed_penalty ───────────────────────────────────────────
    # Discourage high speed generally; extra weight when close to ground
    # (ground_proximity factor) so the agent slows down for landing.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    ground_proximity = 1.0 / (1.0 + 5.0 * (y ** 2) ** 0.5)
    speed_penalty = -0.12 * speed * (1.0 + ground_proximity)

    # ── 4. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating.  Same form as v1, slightly
    # reduced angular-vel coefficient.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 5. contact_reward ──────────────────────────────────────────
    # Reward leg-ground contact (0, 1, or 2 legs), scaled by quality:
    # low speed and upright attitude.  *Not* binary — partial credit
    # for single-leg contact provides a learning gradient.
    n_contact = left_contact + right_contact          # 0.0, 1.0, or 2.0
    contact_quality = 1.0 / ((1.0 + 3.0 * speed) * (1.0 + 2.0 * (angle ** 2) ** 0.5))
    contact_reward = 1.5 * n_contact * contact_quality

    # ── 6. fuel_penalty ────────────────────────────────────────────
    # Action 0 = no engine; actions 1,2,3 fire at least one engine.
    # Discourages unnecessary thrust and makes hovering expensive.
    fuel_penalty = -0.25 if action != 0 else 0.0

    # ── 7. step_penalty ────────────────────────────────────────────
    # Slight time pressure — makes indefinite hovering net-negative.
    step_penalty = -0.02

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        proximity_progress
        + altitude_reward
        + speed_penalty
        + attitude_penalty
        + contact_reward
        + fuel_penalty
        + step_penalty
    )

    components = {
        "proximity_progress": proximity_progress,
        "altitude_reward":    altitude_reward,
        "speed_penalty":      speed_penalty,
        "attitude_penalty":   attitude_penalty,
        "contact_reward":     contact_reward,
        "fuel_penalty":       fuel_penalty,
        "step_penalty":       step_penalty,
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