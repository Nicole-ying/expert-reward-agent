下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_8, score=154.328659)
# Training Feedback

## Final-policy outcome
score=154.328659, len=415.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-62.691647, 260.370380]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 3695.250000 | 99.7% | 99.7% | 13.7% |
| velocity_penalty | -6.637381 | -0.2% | 0.2% | 98.1% |
| goal_proximity | 2.296536 | 0.1% | 0.1% | 98.0% |
| orientation_penalty | -2.097178 | -0.1% | 0.1% | 100.0% |

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
| contact_bonus | 84.500 | 1248.000 | 5005.000 | 7975.500 | 7302.750 | 5590.000 | 6441.500 | 3159.000 | 3506.750 | 3688.750 |
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
    v8: v6 backbone with modest contact reduction (80→65) to curb late-training
    over-stay degradation while staying safely above the activation threshold.

    Diagnosis from v7 (score=-121, total regression from v6's 154):
    - contact 80→40 collapsed landing behavior (active 13.7%→0.1%).
      History shows contact must be ≥80 to reliably activate (iter_2=100✓,
      iter_6=80✓, iter_7=40✗, iter_5=25✗).
    - velocity 0.3→0.5 + potential 2→6 created conflicting near-ground signals
      that made approach net-positive but landing net-negative vs staying aloft.

    Diagnosis from v6 (best=154, cp80%=240, cp100%=160):
    - Architecture is sound: potential-based progress + per-step contact.
    - cp80%→cp100% degradation: agent over-stayed on pad (+8 contact steps),
      approach efficiency dropped (goal_proximity 2.703→2.242).
    - Contact at 99.7% magnitude_share drowns efficiency signals in late training.

    Changes from v7 (effectively reverting toward v6):
    1. w_contact: 40 → 65 — modest reduction from v6's 80 (19% cut, not 50%).
       Still well above the failure threshold seen at 40 and 25.
       Reduces per-step incentive to over-stay without killing landing motivation.
    2. w_pot: 6.0 → 2.0 — revert to v6's proven approach gradient.
       v7's 6.0 created conflicting incentives near ground.
    3. w_v: 0.5 → 0.3 — revert to v6's velocity damping.
       v7's 0.5 was too punitive near ground, discouraging final approach.
    4. w_ang: 0.08 → 0.05 — revert to v6's orientation weight.

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
    # Weight 2.0 (v6 proven): provides consistent approach gradient
    # without conflicting with contact incentive near ground.
    w_pot = 2.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized. Weight 0.3 (v6 proven): enough to deter
    # high-speed crashes without discouraging final approach.
    w_v = 0.3
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Weights unchanged from v6 — stable across all iterations.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Per-step design (v6/v2 proven): rewards every step with both legs
    # on pad near target. Weight 65: modest reduction from v6's 80 to
    # reduce late-training over-stay incentive while staying safely above
    # the ~40-50 threshold where landing behavior collapses.
    w_contact = 65.0
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