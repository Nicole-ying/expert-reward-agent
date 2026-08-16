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

# 上一轮训练反馈 (iter_7, score=24.815189)
# Training Feedback

## Final-policy outcome
score=24.815189, len=687.850000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-267.150515, 262.620642]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settled_reward | 326.066588 | 76.3% | 76.3% | 11.1% |
| goal_proximity | -57.383571 | -13.4% | 13.4% | 100.0% |
| contact_bonus | 24.360957 | 5.7% | 5.7% | 0.7% |
| landing_approach | 8.433396 | 2.0% | 2.0% | 100.0% |
| velocity_penalty | -5.138310 | -1.2% | 1.2% | 99.7% |
| fuel_penalty | -2.854000 | -0.7% | 0.7% | 83.0% |
| orientation_penalty | -1.981543 | -0.5% | 0.5% | 100.0% |
| contact_speed_penalty | -1.345398 | -0.3% | 0.3% | 10.9% |

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
| -395.4 | 59.3 | -102.7 | -55.5 | 88.9 | -12.6 | 102.0 | 44.4 | 55.2 | 83.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.000 | 1.360 | 0.009 | 0.487 | 1.279 | 0.134 | 1.201 | 0.441 | 1.201 | 30.892 |
| contact_speed_penalty | -0.119 | -2.873 | -3.206 | -0.861 | -0.932 | -0.470 | -0.473 | -1.263 | -0.463 | -0.446 |
| fuel_penalty | -0.253 | -1.506 | -1.357 | -1.458 | -2.043 | -1.614 | -2.269 | -1.578 | -3.360 | -2.373 |
| goal_proximity | -18.322 | -41.498 | -45.085 | -49.962 | -47.520 | -58.491 | -60.703 | -52.359 | -66.355 | -48.714 |
| landing_approach | 0.003 | 1.401 | 0.708 | 1.181 | 6.357 | 2.178 | 3.936 | 1.628 | 7.140 | 6.141 |
| orientation_penalty | -4.845 | -3.167 | -2.055 | -1.875 | -2.688 | -1.838 | -2.086 | -1.980 | -2.116 | -1.789 |
| settled_reward | 0.000 | 129.846 | 4.550 | 22.575 | 201.214 | 14.236 | 74.999 | 49.778 | 61.289 | 455.298 |
| velocity_penalty | -25.927 | -12.730 | -11.451 | -11.456 | -6.033 | -5.040 | -7.066 | -8.334 | -4.727 | -4.439 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.0% | 1.3% | 0.4% | 0.2% | 0.2% | 0.2% | 0.2% | 0.3% | 0.1% | 1.7% |
| contact_speed_penalty | 0.1% | 18.0% | 6.8% | 6.7% | 14.8% | 3.1% | 7.4% | 9.3% | 3.7% | 20.5% |
| fuel_penalty | 64.1% | 96.6% | 97.4% | 90.6% | 70.3% | 75.6% | 86.6% | 85.7% | 90.5% | 85.5% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_approach | 99.5% | 100.0% | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| settled_reward | 0.1% | 18.1% | 6.8% | 6.8% | 15.7% | 3.5% | 8.4% | 10.2% | 4.1% | 21.0% |
| velocity_penalty | 100.0% | 99.4% | 100.0% | 99.8% | 98.9% | 99.3% | 98.5% | 99.0% | 99.4% | 99.5% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7: return to contact_bonus-driven architecture (proven in iter_5, score=175.7)
        with structural anti-bounce fixes.

    Diagnosis (from iter_5→6 evidence):
    - iter_5 succeeded with one-shot contact_bonus (w=50) + settled_reward (w=2),
      but cp80-90% showed bounce exploit: agent learned tiny hops to re-trigger
      contact_bonus at near-perfect gates → 2454-2711 ep_sum, score crashed.
    - iter_6 tried pre_landing_bonus to bridge approach→landing, but the
      not_contacted gate created a hover trap: agent hovered indefinitely near
      pad (96.8% active, 62% share) without landing. Only 7/20 terminated.
    - contact_break_penalty was a dead letter (0.0% active, -1.2 ep_sum).

    Key changes from v6:
    1. REMOVED pre_landing_bonus — the hover trap root cause.
    2. REMOVED contact_break_penalty — never activated, no evidence of efficacy.
    3. contact_bonus: w 15→40, ADDED approach_scale: bonus scales with how far the
       agent approached FROM. Genuine landing from distance → full bonus. Tiny hop
       re-trigger (dist_cur ≈ 0.05) → ~10% bonus. This structurally prevents bounce.
    4. settled_reward: w 4→6, wider gates (spatial σ² 0.10→0.25, speed 0.08→0.15,
       angle 0.15→0.30). Makes sustained contact clearly dominant (~5/step vs ~0.2
       for any residual bounce).
    5. contact_speed_penalty: w 1.5→2.0 — stronger speed suppression while contacted.
    6. goal_proximity: keep linear distance (same as v6), no approach_bonus.
    7. velocity_penalty: keep gated + anti-slam (same as v6).
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
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt_sq = x_nxt ** 2 + y_nxt ** 2
    dist_nxt = dist_nxt_sq ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2
    had_contact = left_c_cur * right_c_cur          # 0.0 or 1.0
    has_contact = left_c_nxt * right_c_nxt           # 0.0 or 1.0

    # ══════════════════════════════════════════════════════════════════
    # 1. goal_proximity — linear distance penalty
    #    Simple, monotonic, consistent gradient. No approach_bonus (avoids
    #    oscillation: approach→overshoot→re-approach cycles).
    # ══════════════════════════════════════════════════════════════════
    w_dist = 0.25
    goal_proximity = -w_dist * dist_nxt               # -1.25 at dist=5, -0.25 at dist=1

    # ══════════════════════════════════════════════════════════════════
    # 2. velocity_penalty — gated speed cost + anti-slam near ground
    #    Gating allows free motion far from pad. Anti-slam harshly penalises
    #    downward velocity when very close to ground (prevents crash-landing).
    # ══════════════════════════════════════════════════════════════════
    k_gate = 4.0
    proximity_gate = 1.0 / (1.0 + k_gate * dist_nxt)   # ~1 at pad, ~0.2 at dist=1

    w_v = 0.40
    gated_speed_penalty = -w_v * proximity_gate * speed_sq

    w_slam = 3.0
    slam_gate = 2.718281828 ** (-dist_nxt_sq / 0.03)    # tight: only dist < ~0.2 matters
    downward_v = max(0.0, -vy_nxt)
    anti_slam = -w_slam * slam_gate * (downward_v ** 2)

    velocity_penalty = gated_speed_penalty + anti_slam

    # ══════════════════════════════════════════════════════════════════
    # 3. orientation_penalty — stay upright, suppress spin
    # ══════════════════════════════════════════════════════════════════
    w_ang = 0.06
    w_angvel = 0.015
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ══════════════════════════════════════════════════════════════════
    # 4. landing_approach — micro-shaping, anti-hover whisper
    #    Weight kept tiny (0.03). Provides soft gradient toward slow+near+upright
    #    without creating a profitable steady state.
    # ══════════════════════════════════════════════════════════════════
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)
    slow     = 2.718281828 ** (-speed_sq / 0.60)
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30)
    landing_approach = 0.03 * near_y * near_x * slow * upright

    # ══════════════════════════════════════════════════════════════════
    # 5. contact_bonus — ONE-SHOT transition bonus with anti-bounce gate
    #
    #    ANTI-BOUNCE MECHANISM: approach_scale
    #    The bonus is scaled by 1 - exp(-dist_cur / 0.50), where dist_cur is
    #    the agent's distance from pad BEFORE making contact.
    #    - Genuine landing from far away (dist_cur=3.0): scale ≈ 0.998 → full bonus
    #    - Tiny hop re-trigger (dist_cur=0.05): scale ≈ 0.095 → only ~10% bonus
    #    This structurally prevents the bounce exploit seen in iter_5 cp80-90%.
    #    A bounce cycle nets ~3-4 bonus minus penalties → clearly inferior to
    #    settled_reward's ~5-6 per step.
    # ══════════════════════════════════════════════════════════════════
    w_contact = 40.0
    new_contact = has_contact * (1.0 - had_contact)       # 1 ONLY on transition step

    spatial_gate = 2.718281828 ** (-dist_nxt_sq / 0.04)   # σ²=0.04: tight, at y=0.05 gate=0.94
    speed_gate   = 2.718281828 ** (-speed_sq / 0.04)      # σ²=0.04: tight, at speed=0.3 gate=0.11
    approach_scale = 1.0 - 2.718281828 ** (-dist_cur / 0.50)  # anti-bounce: near→low, far→high

    contact_bonus = w_contact * new_contact * spatial_gate * speed_gate * approach_scale

    # ══════════════════════════════════════════════════════════════════
    # 6. settled_reward — per-step reward for STAYING landed
    #
    #    BOOSTED: w 4→6, wider gates. The dominant positive signal when landed.
    #    ~6 * 0.96 * 0.94 * 0.97 ≈ 5.25/step when well-settled.
    #    Over 200 settled steps ≈ 1050 — dwarfs any residual bounce profit.
    #    Wide gates ensure agent can discover this reward even with imperfect landing.
    # ══════════════════════════════════════════════════════════════════
    w_settled = 6.0
    s_spatial  = 2.718281828 ** (-dist_nxt_sq / 0.25)    # σ²=0.25: tolerant
    s_slow     = 2.718281828 ** (-speed_sq / 0.15)       # σ²=0.15: tolerant
    s_upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30) # σ²=0.30: tolerant
    settled_reward = w_settled * has_contact * s_spatial * s_slow * s_upright

    # ══════════════════════════════════════════════════════════════════
    # 7. contact_speed_penalty — anti-bounce: punish any speed while contacted
    #    Strengthened from 1.5→2.0. Even a tiny bounce requires lifting off,
    #    which incurs speed while (briefly) in contact → penalty.
    # ══════════════════════════════════════════════════════════════════
    w_contact_spd = 2.0
    contact_speed_penalty = -w_contact_spd * has_contact * speed_sq

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
