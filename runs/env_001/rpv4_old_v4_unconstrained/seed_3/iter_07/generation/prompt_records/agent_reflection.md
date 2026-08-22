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

# 上一轮训练反馈 (iter_6, score=57.644809)
# Training Feedback

## Final-policy outcome
score=57.644809, len=845.800000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-44.608090, 252.584578]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| pre_landing_bonus | 186.823219 | 61.9% | 61.9% | 96.8% |
| goal_proximity | -63.820861 | -21.1% | 21.1% | 100.0% |
| settled_reward | 27.320883 | 9.1% | 9.1% | 3.2% |
| landing_approach | 9.305849 | 3.1% | 3.1% | 100.0% |
| velocity_penalty | -4.531410 | -1.5% | 1.5% | 99.7% |
| fuel_penalty | -4.098750 | -1.4% | 1.4% | 96.9% |
| orientation_penalty | -2.414153 | -0.8% | 0.8% | 100.0% |
| contact_bonus | 2.190262 | 0.7% | 0.7% | 0.1% |
| contact_break_penalty | -1.200000 | -0.4% | 0.4% | 0.0% |
| contact_speed_penalty | -0.121308 | -0.0% | 0.0% | 2.9% |

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
| -464.3 | -50.1 | -19.6 | -11.3 | -18.9 | 66.2 | -6.5 | 10.2 | 8.0 | 27.4 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.000 | 0.052 | 1.594 | 0.513 | 0.009 | 3.617 | 0.165 | 0.372 | 0.878 | 0.726 |
| contact_break_penalty | 0.000 | -0.300 | -0.600 | -0.300 | -0.600 | -0.900 | 0.000 | 0.000 | -0.300 | 0.000 |
| contact_speed_penalty | 0.000 | -0.920 | -0.837 | -0.406 | -0.465 | -1.011 | -0.241 | -0.595 | -0.235 | -0.029 |
| fuel_penalty | -0.354 | -0.448 | -1.485 | -3.381 | -3.396 | -3.065 | -4.214 | -4.550 | -4.581 | -4.480 |
| goal_proximity | -18.821 | -22.529 | -31.301 | -40.514 | -47.249 | -39.068 | -59.318 | -61.594 | -46.261 | -64.201 |
| landing_approach | 0.002 | 0.104 | 4.066 | 10.099 | 8.041 | 9.901 | 9.851 | 10.557 | 15.361 | 10.492 |
| orientation_penalty | -4.287 | -1.120 | -1.498 | -1.507 | -2.311 | -1.804 | -2.134 | -2.397 | -2.620 | -2.266 |
| pre_landing_bonus | 0.000 | 0.818 | 72.530 | 198.831 | 182.746 | 171.922 | 202.849 | 223.272 | 260.600 | 214.462 |
| settled_reward | 0.000 | 0.185 | 20.407 | 12.274 | 0.061 | 48.363 | 1.033 | 9.210 | 18.432 | 7.083 |
| velocity_penalty | -23.712 | -10.672 | -7.202 | -6.350 | -8.559 | -7.119 | -6.217 | -5.865 | -6.030 | -4.268 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.0% | 0.5% | 0.2% | 0.0% | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% |
| contact_break_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| contact_speed_penalty | 0.0% | 3.3% | 2.8% | 0.6% | 0.1% | 1.9% | 0.6% | 2.9% | 2.4% | 1.3% |
| fuel_penalty | 90.2% | 79.7% | 84.9% | 93.6% | 92.4% | 92.2% | 99.2% | 99.5% | 96.9% | 99.2% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_approach | 96.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| pre_landing_bonus | 85.5% | 96.7% | 96.7% | 99.4% | 99.9% | 97.7% | 99.4% | 97.1% | 97.6% | 98.6% |
| settled_reward | 0.0% | 3.3% | 3.3% | 0.6% | 0.1% | 2.3% | 0.6% | 2.9% | 2.4% | 1.4% |
| velocity_penalty | 100.0% | 100.0% | 99.4% | 99.9% | 100.0% | 99.3% | 100.0% | 100.0% | 99.9% | 99.9% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v6: anti-bounce overhaul.

    Diagnosis (from iter_5 checkpoint):
    - contact_bonus spiked to 2454-2711 at cp80-90% (active 10.5%), then collapsed to 26.
      Agent discovered bouncing exploit: lose contact → regain → re-trigger one-shot bonus.
    - settled_reward at w=2.0 couldn't compete with bounce profits (~25-40 per bounce cycle).
    - goal_proximity was dead (~-10 flat across all checkpoints).
    - Score peaked at 249 (cp50%) then crashed to 114 (cp80%) → exploitative policy collapsed.

    Changes from v5:
    1. contact_bonus: w 50→15 — reduce bounce profit; still clear landing signal.
    2. settled_reward: w 2.0→4.0 — make staying put clearly better than bouncing.
    3. NEW contact_break_penalty: -6.0 when losing contact — direct anti-bounce cost.
    4. goal_proximity: linear distance (not squared), remove approach_bonus (oscillation risk).
    5. NEW pre_landing_bonus: reward good posture near pad WITHOUT contact — bridges approach→landing.
    6. landing_approach: 0.05→0.03 — further anti-hover.
    7. velocity_penalty: gated, stronger anti-slam near ground.
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
    dist_nxt_sq = x_nxt ** 2 + y_nxt ** 2
    dist_nxt = dist_nxt_sq ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2
    speed = speed_sq ** 0.5
    had_contact = left_c_cur * right_c_cur          # 0.0 or 1.0
    has_contact = left_c_nxt * right_c_nxt           # 0.0 or 1.0
    both_legs = has_contact

    # ══════════════════════════════════════════════════════════════════
    # 1. goal_proximity — linear distance penalty (simple, effective)
    #    Removed approach_bonus: it rewarded oscillation (approach→overshoot→re-approach).
    #    Linear gives consistent gradient: stronger far, gentle near.
    # ══════════════════════════════════════════════════════════════════
    w_dist = 0.25
    goal_proximity = -w_dist * dist_nxt               # -1.25 at dist=5, -0.25 at dist=1

    # ══════════════════════════════════════════════════════════════════
    # 2. velocity_penalty — gated speed cost + strong anti-slam near ground
    #    Gate: speed penalty only matters near the pad (allows free motion far away).
    #    Anti-slam: harsh penalty for downward speed when very close to ground.
    # ══════════════════════════════════════════════════════════════════
    k_gate = 4.0
    proximity_gate = 1.0 / (1.0 + k_gate * dist_nxt)   # ~1 at pad, ~0.2 at dist=1

    w_v = 0.40
    gated_speed_penalty = -w_v * proximity_gate * speed_sq

    # anti-slam: penalise downward velocity near ground (prevents crash-landing)
    w_slam = 3.0
    slam_gate = 2.718281828 ** (-dist_nxt_sq / 0.03)    # tight: only dist < ~0.2 matters
    downward_v = max(0.0, -vy_nxt)                      # positive = moving down
    anti_slam = -w_slam * slam_gate * (downward_v ** 2)

    velocity_penalty = gated_speed_penalty + anti_slam

    # ══════════════════════════════════════════════════════════════════
    # 3. orientation_penalty — stay upright, suppress spin
    # ══════════════════════════════════════════════════════════════════
    w_ang = 0.06
    w_angvel = 0.015
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ══════════════════════════════════════════════════════════════════
    # 4. landing_approach — micro shaping, anti-hover
    #    Further reduced from 0.05 → 0.03. Only a whisper of gradient.
    # ══════════════════════════════════════════════════════════════════
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)
    slow     = 2.718281828 ** (-speed_sq / 0.60)
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30)
    landing_approach = 0.03 * near_y * near_x * slow * upright

    # ══════════════════════════════════════════════════════════════════
    # 5. pre_landing_bonus — reward good approach posture near pad
    #    Bridges the gap between "接近目标" and "着陆接触".
    #    ONLY active when NOT in contact — once landed, settled_reward takes over.
    #    This gives the agent a reason to slow down and align BEFORE touching down.
    # ══════════════════════════════════════════════════════════════════
    w_preland = 0.40
    near_pad = 2.718281828 ** (-dist_nxt_sq / 0.15)       # σ²=0.15: active within ~0.5 dist
    pre_slow = 2.718281828 ** (-speed_sq / 0.25)           # reward speed < ~0.5
    pre_upright = 2.718281828 ** (-abs(angle_nxt) / 0.25)  # reward angle < ~0.5
    not_contacted = 1.0 - has_contact                       # 0 when in contact
    pre_landing_bonus = w_preland * near_pad * pre_slow * pre_upright * not_contacted

    # ══════════════════════════════════════════════════════════════════
    # 6. contact_bonus — ONE-SHOT transition bonus (REDUCED)
    #    w reduced from 50→15. Still a clear signal to land, but no longer
    #    profitable to bounce. Bounce cycle: +15*gates - 6(break_penalty) < 0.
    #    Tight spatial + speed gates ensure only clean landings get full bonus.
    # ══════════════════════════════════════════════════════════════════
    w_contact = 15.0
    new_contact = has_contact * (1.0 - had_contact)       # 1 ONLY on transition step

    spatial_gate = 2.718281828 ** (-dist_nxt_sq / 0.05)   # σ²=0.05: tight
    speed_gate   = 2.718281828 ** (-speed_sq / 0.06)      # σ²=0.06: soft landing only
    contact_bonus = w_contact * new_contact * spatial_gate * speed_gate

    # ══════════════════════════════════════════════════════════════════
    # 7. contact_break_penalty — NEW: punish losing contact
    #    Direct anti-bounce signal. When agent was in contact and loses it,
    #    apply a meaningful penalty. This makes bouncing unprofitable:
    #    bounce_profit ≈ 15*0.5 - 6 = +1.5 at best (with bad gates, net negative).
    #    Staying settled: ~4.0/step — clearly superior.
    # ══════════════════════════════════════════════════════════════════
    w_break = 6.0
    lost_contact = had_contact * (1.0 - has_contact)      # 1 when contact→no contact
    contact_break_penalty = -w_break * lost_contact

    # ══════════════════════════════════════════════════════════════════
    # 8. settled_reward — per-step reward for STAYING landed (INCREASED)
    #    w doubled from 2.0→4.0. Now the dominant positive signal:
    #    ~4.0 * 0.9³ ≈ 2.9/step when well-settled. Over 150 settled steps ≈ 435.
    #    This dwarfs any bounce profit and anchors the policy in stable landing.
    # ══════════════════════════════════════════════════════════════════
    w_settled = 4.0
    s_spatial  = 2.718281828 ** (-dist_nxt_sq / 0.10)
    s_slow     = 2.718281828 ** (-speed_sq / 0.08)
    s_upright  = 2.718281828 ** (-abs(angle_nxt) / 0.15)
    settled_reward = w_settled * has_contact * s_spatial * s_slow * s_upright

    # ══════════════════════════════════════════════════════════════════
    # 9. contact_speed_penalty — anti-bounce: punish speed while in contact
    #    Retained as a secondary anti-bounce measure. Works alongside
    #    contact_break_penalty: speed while contacted → penalty; losing contact → penalty.
    # ══════════════════════════════════════════════════════════════════
    w_contact_spd = 1.5
    contact_speed_penalty = -w_contact_spd * has_contact * speed_sq

    # ══════════════════════════════════════════════════════════════════
    # 10. fuel_penalty — small regularisation
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
        + pre_landing_bonus
        + contact_bonus
        + contact_break_penalty
        + settled_reward
        + contact_speed_penalty
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'pre_landing_bonus': pre_landing_bonus,
        'contact_bonus': contact_bonus,
        'contact_break_penalty': contact_break_penalty,
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
