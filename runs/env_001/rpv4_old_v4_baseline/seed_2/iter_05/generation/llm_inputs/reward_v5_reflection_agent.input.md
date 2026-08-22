下面是环境描述：
本环境是一个 2D 飞行器/着陆器轨迹优化任务。智能体（飞行器）起始于视口顶部中央附近，需要通过学习控制其主引擎与姿态引擎，**到达并稳定停靠在中央目标平台上**。
**主目标：尽可能快地到达目标平台并实现稳定、安全的接触（着陆）。**
**重要次目标（效率约束）：在完成主目标的过程中，尽量减少引擎推力使用（节约燃料）。**
**不该混淆的目标：** 此任务的核心是导航到达，而非单纯的存活平衡或持续前进。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (score=248.710617)
# Training Feedback

## Final-policy outcome
score=248.710617, len=343.050000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[122.259898, 300.447991]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 11.975929 | 69.1% | 69.1% | 28.0% |
| progress | 3.275544 | 18.9% | 20.4% | 96.1% |
| orientation_penalty | -1.586832 | -9.2% | 9.2% | 100.0% |
| velocity_penalty | -0.245900 | -1.4% | 1.4% | 95.6% |

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
| -92.9 | -54.0 | -22.0 | -61.2 | 16.2 | 163.4 | 184.4 | 207.8 | 224.4 | 242.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_guidance | 0.109 | 2.516 | 5.332 | 4.868 | 8.362 | 29.558 | 36.190 | 27.208 | 15.912 | 15.000 |
| orientation_penalty | -0.340 | -0.560 | -0.565 | -1.392 | -1.681 | -2.383 | -2.135 | -1.575 | -1.399 | -1.678 |
| progress | 1.896 | 1.896 | 1.713 | 1.782 | 2.138 | 2.825 | 3.070 | 3.211 | 3.200 | 3.271 |
| velocity_penalty | -0.553 | -0.517 | -0.458 | -0.538 | -0.377 | -0.311 | -0.252 | -0.243 | -0.221 | -0.254 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_guidance | 1.0% | 17.2% | 19.3% | 10.2% | 17.9% | 44.7% | 54.1% | 43.5% | 41.7% | 32.2% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 99.3% | 98.6% | 99.3% | 97.5% | 96.8% |
| velocity_penalty | 99.9% | 99.9% | 99.9% | 100.0% | 99.8% | 99.0% | 98.4% | 99.1% | 97.1% | 96.3% |
# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # distance to target (target at origin (0,0))
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # velocity components
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # orientation penalty: body angle and angular velocity
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # contact guidance: reward both legs touching, but only near the target and at low speed
    contact_val = float(next_obs[6] + next_obs[7])   # range 0.0 .. 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 3.0)
    velocity_ok = max(0.0, 1.0 - abs(vy) / 0.5)
    contact_guidance = 0.07 * contact_val * proximity_factor * velocity_ok

    # combine all components
    total_reward = (2.5 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 2.5 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```

# 反思规则

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

分析完每个组件后再写代码。

### 输出要求

Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
