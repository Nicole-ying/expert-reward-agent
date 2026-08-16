# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。根据训练反馈，修改奖励函数以改善外部任务表现。

# 可用工具

以下工具随时可用，在诊断过程中主动调用：

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action。
- `get_component_history(component_name)` — 某个组件在所有迭代中的 ep_sum_mean 和 active_rate 演化轨迹。
- `read_training_feedback(iteration)` — 某轮训练的详细反馈：精确分数、每个组件的 ep_sum/active_rate/magnitude_share。
- `read_past_reflection(iteration)` — 你过去某轮的诊断和干预记录。
- `read_reward_code(iteration)` — 某轮的完整奖励函数代码。
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件。
- `read_checkpoint_trend(iteration)` — 查看某轮训练中各组件随训练的演化轨迹。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。


```

## User Prompt

```markdown
下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_5, score=175.744060)
# Training Feedback

## Final-policy outcome
score=175.744060, len=588.650000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-4.793600, 279.181440]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 1177.023470 | 78.0% | 78.0% | 5.8% |
| settled_reward | 297.461290 | 19.7% | 19.7% | 38.3% |
| landing_approach | 14.621929 | 1.0% | 1.0% | 100.0% |
| goal_proximity | -9.891942 | -0.7% | 0.7% | 100.0% |
| velocity_penalty | -5.268320 | -0.3% | 0.3% | 99.3% |
| fuel_penalty | -1.782250 | -0.1% | 0.1% | 60.6% |
| orientation_penalty | -1.407820 | -0.1% | 0.1% | 100.0% |
| contact_speed_penalty | -0.556098 | -0.0% | 0.0% | 38.3% |

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
| -6.1 | -62.9 | 156.9 | 161.8 | 249.0 | 229.4 | 239.0 | 114.1 | 144.1 | 180.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 1185.072 | 463.765 | 399.688 | 64.484 | 21.900 | 462.330 | 83.401 | 2454.477 | 2711.412 | 25.639 |
| contact_speed_penalty | -1.002 | -0.179 | -0.876 | -0.620 | -0.656 | -1.881 | -0.379 | -0.927 | -0.794 | -0.380 |
| fuel_penalty | -2.429 | -3.243 | -2.356 | -1.112 | -0.636 | -1.019 | -1.052 | -2.025 | -1.759 | -1.625 |
| goal_proximity | -12.389 | -13.025 | -9.648 | -11.486 | -9.278 | -9.161 | -10.353 | -10.061 | -9.426 | -9.449 |
| landing_approach | 14.646 | 10.481 | 17.431 | 2.886 | 2.492 | 4.726 | 4.469 | 16.239 | 18.498 | 13.697 |
| orientation_penalty | -2.434 | -2.903 | -1.709 | -1.895 | -1.235 | -1.519 | -1.953 | -1.970 | -1.495 | -1.395 |
| settled_reward | 161.053 | 55.176 | 687.426 | 33.194 | 40.517 | 92.524 | 101.667 | 336.399 | 358.746 | 289.270 |
| velocity_penalty | -11.842 | -9.273 | -7.065 | -7.738 | -5.235 | -6.484 | -7.446 | -6.604 | -6.935 | -5.136 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 5.6% | 3.0% | 2.3% | 2.6% | 0.5% | 5.5% | 2.3% | 10.7% | 10.5% | 0.2% |
| contact_speed_penalty | 18.3% | 9.8% | 63.0% | 16.5% | 15.2% | 31.0% | 36.6% | 38.3% | 35.9% | 38.1% |
| fuel_penalty | 67.2% | 95.1% | 69.8% | 69.9% | 54.9% | 62.9% | 58.8% | 60.8% | 52.9% | 61.7% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_approach | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| settled_reward | 18.3% | 9.8% | 63.5% | 18.1% | 18.4% | 32.8% | 38.6% | 38.4% | 36.0% | 38.7% |
| velocity_penalty | 99.6% | 100.0% | 99.4% | 97.0% | 94.8% | 97.0% | 97.3% | 99.4% | 99.1% | 99.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5: restore one-shot contact_bonus as primary landing driver; gut landing_approach;
        widen settled_reward gates; squared-distance proximity.

    Core thesis (from iter_1→4 evidence):
    - contact_bonus is the ONLY signal that has ever produced landing behavior.
      iter_2: per-step contact_bonus → 20/20 landing. iter_3: one-shot → 9/20 + hover.
      iter_4: removed → score collapse to 10.
    - landing_approach at high weight causes hover parasitism (iter_3: 304 ep_sum).
    - Fix: HUGE one-shot contact_bonus + TINY landing_approach + discoverable settled_reward.

    Changes from v4:
    - RESTORED one-shot transition contact_bonus, w=50 (was 8 in v3, removed in v4).
    - landing_approach: 0.20 → 0.05 (anti-hover).
    - settled_reward: wider gates (spatial 0.04→0.10, speed 0.03→0.08), w 3.5→2.0.
    - goal_proximity: squared distance for stronger far-field gradient.
    - contact_speed_penalty: kept, w reduced 2.0→1.2.
    """

    # ── unpack ──────────────────────────────────────────────────────
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_c_cur, right_c_cur = obs[6], obs[7]
    left_c_nxt, right_c_nxt = next_obs[6], next_obs[7]

    # ── derived ─────────────────────────────────────────────────────
    dist_cur_sq = x_cur ** 2 + y_cur ** 2
    dist_nxt_sq = x_nxt ** 2 + y_nxt ** 2
    dist_cur = dist_cur_sq ** 0.5
    dist_nxt = dist_nxt_sq ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2
    both_legs = left_c_nxt * right_c_nxt        # 0.0 or 1.0

    # ══════════════════════════════════════════════════════════════════
    # 1. goal_proximity — squared-distance penalty + approach bonus
    #    Squared gives stronger gradient far away, gentler near target.
    # ══════════════════════════════════════════════════════════════════
    w_dist = 0.10
    proximity_penalty = -w_dist * dist_nxt_sq       # -2.5 at dist=5, -0.1 at dist=1

    w_approach = 1.0
    delta_dist = dist_cur - dist_nxt                 # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ══════════════════════════════════════════════════════════════════
    # 2. velocity_penalty — gated speed penalty + anti-slam near ground
    # ══════════════════════════════════════════════════════════════════
    w_v = 0.30
    k_gate = 5.0
    gate = 1.0 / (1.0 + k_gate * dist_nxt)          # ~1 near pad, ~0 far away

    # anti-slam: penalise downward velocity when very close to ground
    w_slam = 1.5
    slam_gate = 1.0 / (1.0 + 25.0 * dist_nxt)        # tight: only |dist| < ~0.15 matters
    downward_v = max(0.0, -vy_nxt)                   # positive = moving downward

    velocity_penalty = (
        -w_v * gate * speed_sq
        - w_slam * slam_gate * (downward_v ** 2)
    )

    # ══════════════════════════════════════════════════════════════════
    # 3. orientation_penalty — stay upright, suppress spin
    # ══════════════════════════════════════════════════════════════════
    w_ang = 0.05
    w_angvel = 0.01
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ══════════════════════════════════════════════════════════════════
    # 4. landing_approach — TINY continuous shaping, anti-hover
    #    Weight gutted from 0.20 (v4) / 0.60 (v3) → 0.05.
    #    Only provides gradient; NOT profitable as a steady state.
    # ══════════════════════════════════════════════════════════════════
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)
    slow     = 2.718281828 ** (-speed_sq / 0.60)
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30)
    landing_approach = 0.05 * near_y * near_x * slow * upright

    # ══════════════════════════════════════════════════════════════════
    # 5. contact_bonus — ONE-SHOT transition bonus (RESTORED)
    #    The essential landing driver. Fires ONCE on first touchdown.
    #    Tight spatial + speed gates prevent bounce-exploit:
    #    a re-trigger after bouncing has worse position & speed → small bonus.
    # ══════════════════════════════════════════════════════════════════
    w_contact = 50.0
    had_contact = left_c_cur * right_c_cur             # were we already touching?
    has_contact = both_legs                            # are we touching now?
    new_contact = has_contact * (1.0 - had_contact)    # 1 ONLY on transition step

    # tight gates: only a clean landing at the pad gets full bonus
    spatial_gate = 2.718281828 ** (-dist_nxt_sq / 0.06)    # σ²=0.06 → tight
    speed_gate   = 2.718281828 ** (-speed_sq / 0.08)       # σ²=0.08 → soft landing
    contact_bonus = w_contact * new_contact * spatial_gate * speed_gate

    # ══════════════════════════════════════════════════════════════════
    # 6. settled_reward — per-step reward for STAYING landed
    #    Wider gates than v4 so agent can discover it.
    #    Only active when both legs are in sustained contact.
    # ══════════════════════════════════════════════════════════════════
    w_settled = 2.0
    s_spatial  = 2.718281828 ** (-dist_nxt_sq / 0.10)   # was 0.04 in v4
    s_slow     = 2.718281828 ** (-speed_sq / 0.08)      # was 0.03 in v4
    s_upright  = 2.718281828 ** (-abs(angle_nxt) / 0.15) # was 0.10 in v4
    settled_reward = w_settled * both_legs * s_spatial * s_slow * s_upright

    # ══════════════════════════════════════════════════════════════════
    # 7. contact_speed_penalty — anti-bounce: punish speed while in contact
    # ══════════════════════════════════════════════════════════════════
    w_contact_spd = 1.2
    contact_speed_penalty = -w_contact_spd * both_legs * speed_sq

    # ══════════════════════════════════════════════════════════════════
    # 8. fuel_penalty — small regularisation
    # ══════════════════════════════════════════════════════════════════
    w_fuel = 0.005
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ══════════════════════════════════════════════════════════════════
    # assemble
    # ══════════════════════════════════════════════════════════════════
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + contact_bonus
        + settled_reward
        + contact_speed_penalty
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'contact_bonus': contact_bonus,
        'settled_reward': settled_reward,
        'contact_speed_penalty': contact_speed_penalty,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components
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
```
