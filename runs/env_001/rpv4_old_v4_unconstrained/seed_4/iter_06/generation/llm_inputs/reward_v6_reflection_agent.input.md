下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_5, score=-98.299635)
# Training Feedback

## Final-policy outcome
score=-98.299635, len=68.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-118.010880, -68.230143]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | -38.574747 | -95.4% | 95.4% | 100.0% |
| velocity_penalty | -1.829619 | -4.5% | 4.5% | 100.0% |
| angle_penalty | -0.030607 | -0.1% | 0.1% | 100.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -131.1 | -112.4 | -110.0 | -114.6 | -113.8 | -112.1 | -114.3 | -101.7 | -107.8 | -97.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angle_penalty | -0.831 | -0.127 | -0.021 | -0.019 | -0.017 | -0.017 | -0.023 | -0.018 | -0.018 | -0.031 |
| approach_reward | -38.829 | -38.639 | -38.574 | -38.552 | -38.563 | -38.557 | -38.558 | -38.576 | -38.563 | -38.576 |
| boundary_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| landing_bonus | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| velocity_penalty | -1.845 | -1.859 | -1.857 | -1.849 | -1.854 | -1.854 | -1.855 | -1.852 | -1.851 | -1.818 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angle_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| boundary_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| velocity_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- unpack observations ---
    x_pos = obs[0]           # horizontal distance to target
    y_pos = obs[1]           # vertical distance to target
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    # --- hyperparameters ---
    w_dist = 0.5            # approach reward weight
    w_vel = 0.02            # velocity penalty weight
    w_angle = 0.1           # body angle penalty weight
    w_angvel = 0.05         # angular velocity penalty weight
    w_boundary = 0.5        # horizontal out-of-bounds penalty
    w_landing = 1.0         # landing quality bonus weight

    safe_x_bound = 1.0       # x beyond this is considered unsafe
    vel_thresh = 0.5         # velocity below which landing is considered stable
    angle_thresh = 0.2       # body angle below which landing is considered upright

    # 1. Approach reward (main learning signal)
    dist_sq = x_pos**2 + y_pos**2
    approach = -w_dist * dist_sq

    # 2. Stability penalties (soft constraints)
    vel_penalty = -w_vel * (x_vel**2 + y_vel**2)
    angle_penalty = -w_angle * body_angle**2 - w_angvel * ang_vel**2

    # 3. Horizontal boundary penalty (prevent drifting out of viewport)
    if abs(x_pos) > safe_x_bound:
        boundary_penalty = -w_boundary * (abs(x_pos) - safe_x_bound)**2
    else:
        boundary_penalty = 0.0

    # 4. Landing quality bonus (joint condition proxy)
    both_legs = left_contact * right_contact  # = 1.0 only if both legs touch
    # bounded stability factors
    vel_stable = max(0.0, 1.0 - abs(x_vel) / vel_thresh)
    yvel_stable = max(0.0, 1.0 - abs(y_vel) / vel_thresh)
    angle_stable = max(0.0, 1.0 - abs(body_angle) / angle_thresh)
    landing_quality = both_legs * vel_stable * yvel_stable * angle_stable
    landing_bonus = w_landing * landing_quality

    total_reward = approach + vel_penalty + angle_penalty + boundary_penalty + landing_bonus

    components = {
        "approach_reward": approach,
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "boundary_penalty": boundary_penalty,
        "landing_bonus": landing_bonus
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