下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_5, score=-96.022755)
# Training Feedback

## Final-policy outcome
score=-96.022755, len=938.350000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-211.704988, -21.881394]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -36.237845 | -82.2% | 82.2% | 100.0% |
| velocity_penalty | -5.799167 | -13.2% | 13.2% | 100.0% |
| orientation_penalty | -2.022030 | -4.6% | 4.6% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -68.9 | -112.8 | -76.8 | -103.8 | -78.0 | 15.3 | -38.8 | -37.5 | -66.5 | -92.9 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 46.250 | 185.000 | 365.000 | 87.500 | 0.000 | 458.750 | 137.500 | 0.000 | 0.000 | 0.000 |
| goal_proximity | -5.367 | -18.868 | -21.845 | -34.034 | -36.524 | -20.436 | -26.535 | -30.292 | -38.114 | -35.453 |
| orientation_penalty | -0.416 | -1.002 | -1.092 | -1.284 | -1.328 | -1.547 | -1.524 | -1.419 | -1.576 | -1.945 |
| velocity_penalty | -18.323 | -7.656 | -6.419 | -5.119 | -4.926 | -5.365 | -4.686 | -4.423 | -4.567 | -5.698 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 1.4% | 1.5% | 2.4% | 0.4% | 0.0% | 2.2% | 0.6% | 0.0% | 0.0% | 0.0% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5: per-step contact bonus (v2 skeleton) with reduced weight + stronger velocity damping

    Key diagnosis from v4 failure:
    - v4's one-time transition bonus (+200) created a flat reward landscape:
      contact_bonus stayed at ~142 across ALL checkpoints with zero growth.
      Agent learned "dive fast, grab bonus, die" (len=68.55, all early terminate).
    - v2's per-step bonus grew from 125→2820 during training — reward scaled
      naturally with landing quality, creating a curriculum from touch to settle.

    Changes from v4:
    1. contact_bonus: revert from one-time transition (+200) + per-step settling (+2)
       back to pure per-step bonus (v2 approach). Weight reduced 100→25 to address
       the 99.6% magnitude dominance that plagued v2.
    2. velocity_penalty: strengthened 0.15→0.4 (v4 had 0.15) to provide meaningful
       gradient against high-speed approaches, now that contact_bonus is smaller.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: direct distance penalty (v2/v4 approach, proven stable) ---
    # Consistent gradient toward (0,0) at every step.
    w_dist = 0.05
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = -w_dist * dist_nxt

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Strengthened (0.15→0.4) to deter high-speed crashes now that contact_bonus
    # is moderate. Gate ensures far-field maneuvering isn't overly penalized.
    w_v = 0.4
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Keep lander upright, suppress wobble. Unchanged from v2/v4.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Reverted from one-time transition (v4) back to per-step (v2 skeleton).
    # Weight 25 (vs v2's 100) keeps contact as the anchor signal while giving
    # velocity/orientation penalties room to shape behavior.
    # Per-step design means longer settled stays → higher reward, creating
    # natural curriculum from first touch to stable landing.
    w_contact = 25.0
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