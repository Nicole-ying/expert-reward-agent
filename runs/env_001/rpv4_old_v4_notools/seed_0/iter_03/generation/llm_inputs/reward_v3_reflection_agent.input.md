下面是环境描述：
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

目标得分：200

### 第一原则：黑盒实验，不是代码重构

你无法预知新奖励函数会引导出什么行为，只能训练完后看结果。
**看到好趋势 → 先理解为什么好，再精准微调。**
大规模重写等于丢弃已有实验证据，重新掷骰子。

**硬约束：如果本轮得分比上一轮改善超过 30 分，你必须先回答：**
上一轮的哪个修改促成了这个改善？证据是什么？
**如果你回答不了，本轮只做 L1，不改数学形态。**

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

### 本轮数据

# 上一轮训练反馈 (iter_2, score=-168.021945)
# Training Feedback

## Final-policy outcome
score=-168.021945, len=387.200000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-264.295058, -124.453488]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | -0.071833 | -36.4% | 93.7% | 100.0% |
| orientation_penalty | -0.012515 | -6.3% | 6.3% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| vel_damping | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
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
| -758.2 | -755.6 | -259.7 | -253.9 | -188.9 | -202.6 | -185.0 | -173.3 | -168.7 | -166.4 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_reward | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| orientation_penalty | -2.573 | -2.531 | -0.508 | -0.098 | -0.018 | -0.032 | -0.028 | -0.032 | -0.012 | -0.010 |
| progress_reward | -1.602 | -1.601 | -0.330 | -0.421 | -0.183 | -0.191 | -0.115 | -0.091 | -0.082 | -0.084 |
| vel_damping | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_reward | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vel_damping | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: 修复死组件 vel_damping——改为基于高度的速度惩罚，让 agent 在接近地面时减速，
    同时降低 progress 权重以抑制其噪声幅度。
    contact_reward 暂时保留原样，等待 vel_damping 激活后看能否自然触发。
    """
    # 提取信号
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on Euclidean distance) —— 降权
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # 正值 = 接近
    w_progress = 0.5   # 从 1.0 降低
    progress_reward = w_progress * progress

    # 2. Orientation penalty (保持不变)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Soft contact reward (保持不变，等待着陆出现)
    contact_flag = left_contact * right_contact
    sigma_angle = 0.1
    sigma_vel = 0.2
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))
    vel_factor = 2.718281828 ** (- (vx ** 2 + vy ** 2) / (sigma_vel ** 2))
    soft_contact = contact_flag * angle_factor * vel_factor
    w_contact = 2.0
    contact_reward = w_contact * soft_contact

    # 4. 接近地面速度惩罚 (重写 vel_damping) —— 基于高度，不再是全局距离
    altitude = y_curr                   # 假设地面 y=0
    alt_thresh = 1.0                    # 高度 < 1.0 时开始惩罚
    gate_alt = max(0.0, 1.0 - altitude / alt_thresh)   # 0 在高处，1 在地面
    w_vel_damp = 1.0                    # 提升权重以增强信号
    # 主要惩罚垂直速度，也对横向速度适度惩罚
    vel_damping = -w_vel_damp * gate_alt * (abs(vy) + 0.3 * abs(vx))

    total_reward = progress_reward + orientation_penalty + contact_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components
```

---

以上是本轮数据。现在开始调查，做出一个语义方向上的修改。

### 输出要求

**1. 诊断报告**（在代码之前）：
- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的学到的行为模式是什么，是不健康的吗？结合 score、episode_length、terminated/truncated 判断。
- `signal_completeness`：必要职责是否完备？缺什么信号？
- `selected_level`：L1 / L2 / L3 及选择理由
- `selected_intervention`：目标组件及具体修改内容
- `falsifiable_hypothesis`：为什么该修预计改善策略？
- `expected_next_round`：下一轮策略学习到的行为模式预计是什么？

**2. 完整 Python 代码**（```python ... ``` 代码块）。