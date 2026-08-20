下面是环境描述：
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_2, score=-99.872139)
# Training Feedback

## Final-policy outcome
score=-99.872139, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-131.358588, -57.127609]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 3792.866119 | 99.3% | 99.3% | 100.0% |
| vel_damping | -25.982364 | -0.7% | 0.7% | 100.0% |
| progress_reward | 0.398749 | 0.0% | 0.0% | 100.0% |
| orientation_penalty | -0.409954 | -0.0% | 0.0% | 100.0% |

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
| -2016.6 | -1946.4 | -494.8 | -156.3 | -227.9 | -248.8 | -82.1 | -119.6 | -116.7 | -126.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| orientation_penalty | -4.466 | -0.484 | -0.495 | -0.416 | -0.477 | -0.451 | -0.445 | -0.452 | -0.403 | -0.386 |
| progress_reward | -14.198 | -14.653 | -2.887 | -0.167 | -0.839 | -0.695 | 0.578 | 0.198 | 0.240 | 0.115 |
| soft_landing_proxy | 68.992 | 106.346 | 619.645 | 2788.229 | 2255.796 | 1556.581 | 4026.231 | 3551.977 | 3384.548 | 3370.071 |
| vel_damping | -11.063 | -10.581 | -13.048 | -15.423 | -19.626 | -21.778 | -24.125 | -22.297 | -19.580 | -24.506 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| soft_landing_proxy | 67.4% | 75.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vel_damping | 36.1% | 29.7% | 49.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v2: proximity-based soft landing proxy replaces dead contact_reward;
        widened vel_damping gate so it actually activates.
    """
    # Extract relevant signals
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # 1. Progress reward (improvement_delta on Euclidean distance)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # positive when approaching
    w_progress = 1.0
    progress_reward = w_progress * progress

    # 2. Orientation penalty (quadratic_penalty)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Soft landing proxy (proximity-based, continuous — replaces dead contact_reward)
    # Rewards being in the "landing zone": close to pad + low speed + upright.
    # Creates a smooth gradient from approach → hover → touch down.
    sigma_dist = 1.5                                    # landing zone radius
    sigma_vel = 0.5                                     # acceptable landing speed
    sigma_angle = 0.3                                   # acceptable tilt

    dist = dist_curr
    speed = (vx ** 2 + vy ** 2) ** 0.5

    proximity_factor = 2.718281828 ** (- (dist ** 2) / (2.0 * sigma_dist ** 2))
    vel_factor = 2.718281828 ** (- (speed ** 2) / (2.0 * sigma_vel ** 2))
    angle_factor = 2.718281828 ** (- (angle ** 2) / (2.0 * sigma_angle ** 2))

    w_landing = 5.0
    soft_landing_proxy = w_landing * proximity_factor * vel_factor * angle_factor

    # 4. Near-pad velocity damping (wider gate so it activates)
    dist_thresh = 3.0                                   # was 1.0 — far too narrow
    gate = max(0.0, 1.0 - dist / dist_thresh)           # 0 far away, 1 at origin
    w_vel_damp = 0.3
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    total_reward = progress_reward + orientation_penalty + soft_landing_proxy + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "vel_damping": vel_damping
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