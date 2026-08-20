下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_3, score=-50.454204)
# Training Feedback

## Final-policy outcome
score=-50.454204, len=754.300000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-210.441144, 205.911676]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 35.000000 | 84.6% | 84.6% | 0.0% |
| velocity_penalty | -3.533354 | -8.5% | 8.5% | 99.9% |
| orientation_penalty | -1.511864 | -3.7% | 3.7% | 100.0% |
| goal_proximity | 1.015898 | 2.5% | 3.2% | 99.9% |

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
| -124.8 | -116.7 | 100.4 | -97.1 | -93.7 | -77.4 | -81.5 | -74.3 | -77.0 | -98.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 95.000 | 130.000 | 5670.000 | 315.000 | 145.000 | 790.000 | 750.000 | 1285.000 | 30.000 | 35.000 |
| goal_proximity | 1.160 | 1.161 | 1.136 | 0.914 | 0.942 | 0.982 | 0.956 | 0.925 | 0.993 | 0.976 |
| orientation_penalty | -0.464 | -0.302 | -0.930 | -1.266 | -1.361 | -1.411 | -1.422 | -1.457 | -1.537 | -1.460 |
| velocity_penalty | -10.622 | -10.573 | -6.334 | -4.387 | -3.932 | -3.945 | -3.771 | -3.744 | -3.524 | -3.805 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 1.4% | 1.9% | 22.9% | 0.5% | 0.2% | 1.1% | 1.0% | 1.9% | 0.0% | 0.0% |
| goal_proximity | 100.0% | 100.0% | 98.4% | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.7% | 99.8% | 98.4% | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: potential-based approach shaping + proven sparse contact bonus
    
    Key change from v2: goal_proximity switched from direct distance penalty
    (always negative, weak gradient) to potential-based improvement reward
    (positive when approaching target, negative when retreating).
    This gives the agent meaningful gradient during the ~88% of steps before contact,
    while the proven contact_bonus remains as the terminal anchor.
    
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

    # --- 1. goal_proximity: potential-based improvement reward ---
    # Rewards reduction in squared distance to target (0,0).
    # delta > 0 means approaching; delta < 0 means retreating.
    # Squared distance gives naturally stronger gradient when far away
    # and gentler when close, encouraging rapid initial approach then precision.
    # Weights moderate to avoid dominating the contact_bonus anchor.
    w_x = 0.3
    w_y = 0.6
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive = approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive = approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping: stronger near ground ---
    # Quadratic penalty on speed, gated by proximity to pad.
    # Slightly increased weight (0.15→0.20) to counterbalance the stronger
    # approach gradient and discourage high-speed dives.
    w_v = 0.20
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep the lander upright.
    # Weights unchanged from v2 — these worked adequately.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus: large sparse reward for successful landing ---
    # Unchanged from v2 — relaxed thresholds + large per-step bonus.
    # This is the proven anchor that makes landing the global optimum.
    # Per-step nature rewards staying on the pad until body settles.
    w_contact = 100.0
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