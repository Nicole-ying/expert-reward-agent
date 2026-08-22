下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_6, score=154.328659)
# Training Feedback

## Final-policy outcome
score=154.328659, len=415.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-62.691647, 260.370380]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 4548.000000 | 99.7% | 99.7% | 13.7% |
| velocity_penalty | -6.637381 | -0.1% | 0.1% | 98.1% |
| goal_proximity | 2.296536 | 0.1% | 0.1% | 98.0% |
| orientation_penalty | -2.097178 | -0.0% | 0.0% | 100.0% |

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
| -92.1 | -31.7 | -29.9 | 70.9 | 63.4 | 149.0 | 112.0 | 240.2 | 197.6 | 160.2 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 104.000 | 1536.000 | 6160.000 | 9816.000 | 8988.000 | 6880.000 | 7928.000 | 3888.000 | 4316.000 | 4540.000 |
| goal_proximity | 2.228 | 2.173 | 2.160 | 2.300 | 2.421 | 2.513 | 2.394 | 2.703 | 2.383 | 2.242 |
| orientation_penalty | -0.428 | -0.928 | -1.566 | -1.999 | -2.135 | -2.003 | -2.301 | -1.400 | -1.719 | -2.129 |
| velocity_penalty | -15.465 | -10.796 | -12.672 | -12.131 | -10.957 | -8.283 | -11.311 | -8.401 | -6.675 | -6.194 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 1.9% | 3.6% | 10.3% | 18.5% | 13.6% | 12.9% | 16.0% | 14.1% | 16.3% | 13.7% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 99.8% | 100.0% | 98.9% | 99.2% | 96.1% | 97.0% | 98.2% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.5% | 100.0% | 100.0% | 99.8% | 100.0% | 99.0% | 99.3% | 96.6% | 97.3% | 98.3% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v6: potential-based progress reward + per-step contact bonus (iter_2 skeleton)

    Core diagnosis from v5 failure:
    - v5's always-negative distance penalty (goal_proximity=-36.24, 82.2% share)
      created a pure-penalty landscape. Agent learned "hover far away, do nothing"
      because every step toward the target was punished.
    - contact_bonus (25/step) was too small to overcome the penalty gradient;
      agent found landing at cp60% but then abandoned it.
    - Only iter_2's per-step contact_bonus (100) ever successfully activated
      landing behavior. All attempts to "fix" the 99.6% dominance killed it.

    Changes from v5:
    1. goal_proximity: switched from distance penalty (-w*dist) to potential-based
       progress reward: w_pot * (dist_cur - dist_nxt). Positive when approaching,
       negative when retreating. This ensures every step toward target is net-positive
       regardless of contact, restoring exploration gradient.
    2. contact_bonus: per-step weight raised 25→80 — close to iter_2's proven 100,
       but slightly moderated since progress reward now also provides positive signal.
    3. velocity_penalty: slightly reduced 0.4→0.3 — progress reward already encodes
       a soft speed incentive (approaching requires some velocity).

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: potential-based progress reward ---
    # Rewards reduction in Euclidean distance to target (0,0).
    # Positive when approaching, negative when retreating.
    # Unlike distance penalty (always negative), this makes every approach step
    # net-beneficial, restoring the exploration gradient toward the target.
    # Weight 2.0 means approaching 0.1 units/step yields +0.2 reward.
    w_pot = 2.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized so progress reward can drive initial approach.
    w_v = 0.3
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Weights unchanged — these have been stable across all iterations.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Per-step design (iter_2 proven): rewards every step with both legs on pad
    # near target. Creates natural curriculum from first touch → stable settle.
    # Weight 80: close to iter_2's 100 (which worked), moderated because progress
    # reward now provides positive approach gradient.
    w_contact = 80.0
    near_target = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    contact_bonus = w_contact if (near_target and both_legs) else 0.0

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