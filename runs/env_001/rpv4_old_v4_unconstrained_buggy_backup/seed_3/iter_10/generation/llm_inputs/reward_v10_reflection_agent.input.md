下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_9, score=219.649315)
# Training Feedback

## Final-policy outcome
score=219.649315, len=474.150000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[48.871007, 283.784657]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 1031.500000 | 99.2% | 99.2% | 12.8% |
| velocity_penalty | -4.178173 | -0.4% | 0.4% | 97.5% |
| goal_proximity | 2.618898 | 0.3% | 0.3% | 97.2% |
| orientation_penalty | -1.607024 | -0.2% | 0.2% | 100.0% |

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
| 59.8 | -150.9 | -175.6 | -42.4 | -228.5 | 131.3 | 139.8 | 226.8 | 124.1 | 223.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 708.900 | 39.100 | 0.000 | 171.950 | 0.000 | 874.400 | 2038.300 | 736.500 | 705.500 | 1407.350 |
| goal_proximity | 2.124 | 0.352 | -0.039 | 1.667 | -1.092 | 2.307 | 2.578 | 2.591 | 2.139 | 2.576 |
| orientation_penalty | -0.641 | -1.024 | -1.118 | -1.464 | -1.249 | -2.114 | -1.500 | -1.150 | -2.108 | -1.574 |
| velocity_penalty | -10.225 | -6.052 | -3.851 | -3.247 | -4.138 | -3.602 | -3.373 | -3.984 | -4.695 | -4.308 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 20.9% | 0.8% | 0.0% | 1.6% | 0.0% | 17.5% | 12.5% | 12.6% | 18.1% | 11.7% |
| goal_proximity | 98.8% | 100.0% | 100.0% | 100.0% | 100.0% | 98.9% | 99.1% | 96.8% | 98.5% | 97.1% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 98.8% | 100.0% | 100.0% | 100.0% | 100.0% | 99.0% | 99.2% | 97.1% | 98.6% | 97.4% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v9: transition-based contact reward — fix per-step over-stay degradation

    Diagnosis from v8 (score=154, cp80% peaked at 240 then degraded to 160):
    - Per-step contact_bonus (w=65/step) created a persistent over-stay incentive.
      Agent learned to land then stayed collecting ~57 steps of bonus (3695 total),
      drowning out efficiency signals (goal_proximity ~2.3, velocity ~-6.6).
    - cp80%→cp100%: contact rebounded +17%, approach efficiency dropped -17%,
      score fell from 240→160. Classic per-step reward exploitation.
    - Weight tuning alone (v6=80, v7=40→collapsed, v8=65) cannot fix this:
      contact must be strong enough to drive landing, but per-step always over-stays.

    Changes from v8:
    1. contact_bonus: restructured from per-step accumulation to transition-based:
       - landing_bonus (500, one-time): rewards the FIRST step achieving safe contact.
         Equivalent to ~8 old steps — concentrated signal for learning to land.
       - sustain_bonus (3/step, per-step): tiny gradient to stay on pad until
         natural termination. 3 vs old 65 = 22x weaker over-stay incentive.
    2. All other components unchanged from v8 proven backbone.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations (current + next) ---
    x_cur, y_cur = obs[0], obs[1]
    left_cont_cur, right_cont_cur = obs[6], obs[7]

    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: potential-based progress reward ---
    # Rewards reduction in Euclidean distance to target (0,0).
    # Positive when approaching, negative when retreating.
    # Weight 2.0 (v6/v8 proven): consistent approach gradient.
    w_pot = 2.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized. Weight 0.3 (v6/v8 proven).
    w_v = 0.3
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Weights unchanged from v6/v8 — stable across all iterations.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: transition-based landing reward ---
    # KEY CHANGE (v9): switch from per-step accumulation to event-based.
    #
    # Problem: per-step bonus created over-stay exploitation.
    #   Old: +65 every step on pad → 3695 total, 99.7% magnitude share.
    #
    # Solution: one-time landing bonus + tiny sustain.
    #   - landing_bonus: rewards the TRANSITION into safe contact (first touch).
    #     Concentrated 500-point signal — strong enough to drive landing learning.
    #   - sustain_bonus: 3/step for maintaining contact. Provides gentle gradient
    #     to stay on pad until natural termination, but 22x weaker than old 65/step
    #     so over-stay is no longer the dominant strategy.
    #
    # Safe contact definition: both legs on pad, near target center.
    w_land = 500.0      # one-time landing bonus
    w_sustain = 3.0     # per-step sustain (was 65 in v8 → 22x reduction)

    near_target_cur = (abs(x_cur) < 0.5) and (abs(y_cur) < 0.3)
    both_legs_cur = (left_cont_cur > 0.5) and (right_cont_cur > 0.5)
    was_safe = near_target_cur and both_legs_cur

    near_target_nxt = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs_nxt = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    is_safe = near_target_nxt and both_legs_nxt

    # Transition: first step entering safe contact
    landing_bonus = w_land if (is_safe and not was_safe) else 0.0
    # Sustain: every step maintaining safe contact
    sustain_bonus = w_sustain if is_safe else 0.0

    contact_bonus = landing_bonus + sustain_bonus

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
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