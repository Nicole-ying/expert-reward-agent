下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_4, score=-109.533802)
# Training Feedback

## Final-policy outcome
score=-109.533802, len=68.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-137.319938, -67.062972]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 142.400000 | 92.4% | 92.4% | 1.8% |
| velocity_penalty | -7.935640 | -5.2% | 5.2% | 99.6% |
| goal_proximity | -3.328360 | -2.2% | 2.2% | 100.0% |
| orientation_penalty | -0.387919 | -0.3% | 0.3% | 100.0% |

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
| -115.7 | -119.6 | -114.2 | -111.9 | -127.4 | -91.4 | -70.6 | -90.4 | -107.7 | -113.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 162.300 | 152.100 | 162.300 | 152.600 | 101.100 | 152.200 | 142.800 | 152.100 | 142.100 | 142.000 |
| goal_proximity | -3.322 | -3.323 | -3.321 | -3.325 | -3.331 | -3.357 | -3.367 | -3.337 | -3.330 | -3.329 |
| orientation_penalty | -0.246 | -0.329 | -0.200 | -0.175 | -0.730 | -0.412 | -0.475 | -0.341 | -0.396 | -0.362 |
| velocity_penalty | -7.934 | -7.977 | -7.934 | -7.940 | -7.915 | -7.847 | -7.500 | -7.737 | -7.940 | -7.933 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 1.7% | 1.5% | 1.7% | 1.9% | 0.8% | 1.6% | 2.0% | 1.5% | 1.5% | 1.5% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.8% | 99.9% | 99.9% | 99.7% | 99.9% | 99.8% | 99.9% | 100.0% | 99.6% | 99.5% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: one-time landing bonus + direct distance gradient

    Key changes from v3:
    1. goal_proximity: revert from potential-based (delta squared distance)
       back to direct distance penalty — the only approach proven stable (v2).
       Provides consistent gradient toward (0,0) at all positions.
    2. contact_bonus: switch from per-step accumulation (100/step, which
       caused 99.6% magnitude dominance in v2) to a ONE-TIME transition bonus
       (+200 on first pad contact) plus a tiny per-step settling incentive (+2).
       This eliminates the "stay and collect" exploit while retaining the
       landing anchor.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_cur = obs[6]
    right_cont_cur = obs[7]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: direct distance penalty (v2 approach, proven stable) ---
    # Consistent gradient toward (0,0) at every step, regardless of movement.
    # Weight 0.05 means at typical start distance ~8, penalty ≈ -0.4/step;
    # over ~200-step approach accumulates ~-80, clearly beaten by +200 contact.
    w_dist = 0.05
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = -w_dist * dist_nxt

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Stronger damping near ground to prevent high-speed crashes.
    w_v = 0.15
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: ONE-TIME transition bonus + tiny settling incentive ---
    # The core fix: instead of 100/step (which dominated everything in v2
    # and caused training collapse in v3), reward the landing EVENT once,
    # then provide a gentle per-step incentive to stay until body settles.
    #
    # Detect whether agent was already on the pad last step (prevents repeat).
    near_target_cur = (abs(x_cur) < 0.5) and (abs(y_cur) < 0.3)
    both_legs_cur = (left_cont_cur > 0.5) and (right_cont_cur > 0.5)
    was_on_pad = near_target_cur and both_legs_cur

    near_target_nxt = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs_nxt = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    is_on_pad = near_target_nxt and both_legs_nxt

    # Transition: first step entering the pad region with both legs down.
    just_landed = is_on_pad and not was_on_pad

    w_contact_one_time = 200.0   # one-time reward for the landing event
    w_settle_per_step = 2.0      # tiny per-step incentive to maintain contact

    contact_bonus = 0.0
    if just_landed:
        contact_bonus += w_contact_one_time
    if is_on_pad:
        contact_bonus += w_settle_per_step

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