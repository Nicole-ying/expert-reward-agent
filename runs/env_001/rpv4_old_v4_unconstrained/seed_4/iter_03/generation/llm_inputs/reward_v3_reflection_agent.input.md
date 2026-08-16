下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_2, score=-119.337430)
# Training Feedback

## Final-policy outcome
score=-119.337430, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-143.445027, -98.057243]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settling_stability | -121.498283 | -51.4% | 52.9% | 100.0% |
| goal_proximity | -90.806575 | -38.4% | 38.4% | 100.0% |
| attitude_constraint | -17.107631 | -7.2% | 7.3% | 100.0% |
| landing_contact | 3.100000 | 1.3% | 1.3% | 0.5% |
| fuel_efficiency | -0.135000 | -0.1% | 0.1% | 4.0% |

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
| -136.5 | -116.3 | -117.6 | -116.5 | -118.7 | -118.6 | -116.5 | -113.2 | -115.9 | -119.8 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -96.488 | -12.375 | -16.719 | -15.132 | -17.999 | -17.112 | -14.901 | -13.100 | -16.280 | -17.567 |
| fuel_efficiency | -1.347 | -1.500 | -0.165 | -0.158 | -0.152 | -0.140 | -0.128 | -0.175 | -0.160 | -0.138 |
| goal_proximity | -92.475 | -90.616 | -90.870 | -90.825 | -90.824 | -90.805 | -90.819 | -90.826 | -90.828 | -90.817 |
| landing_contact | 1.700 | 3.600 | 3.500 | 3.600 | 3.100 | 3.500 | 2.700 | 4.000 | 3.100 | 3.100 |
| settling_stability | -122.533 | -120.966 | -121.148 | -121.101 | -121.336 | -121.369 | -121.426 | -121.140 | -121.563 | -121.471 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 37.3% | 43.9% | 4.8% | 4.6% | 4.5% | 4.1% | 3.7% | 5.1% | 4.7% | 4.0% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 0.9% | 0.6% | 0.5% | 0.6% | 0.5% | 0.5% | 0.5% | 0.6% | 0.5% | 0.5% |
| settling_stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2。
    
    根本性改动（vs v1）：
    - 从纯 improvement-delta 切换为「delta + 直接状态惩罚」混合架构，
      确保每一步都有可优化的梯度信号。
    - 量级整体提升 5~10 倍，使我们的奖励不被环境原生惩罚淹没。
    - 姿态约束门控修复：远距离时完全关闭（gate=0），不再无故惩罚大角度机动。
    - landing_contact 条件放宽 + 奖励提升，使 agent 能感知到着陆信号。
    """
    
    # ========== 信号提取 ==========
    # 当前状态
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]
    
    # 距离与速率（L2 范数）
    current_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    current_speed = (vx**2 + vy**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5
    
    # ================================================================
    # Component A: goal_proximity — 混合架构（delta + 直接状态惩罚）
    # ================================================================
    # delta 部分：奖励每一步向目标靠近的进展
    dist_improvement = current_dist - next_dist
    
    # 直接状态惩罚：距离越远惩罚越重，提供逐步梯度
    # 使用平滑函数：dist=0→0，dist=1→−1.5，dist=2→−3.0
    dist_penalty = -1.5 * next_dist
    
    goal_proximity = 8.0 * dist_improvement + dist_penalty
    
    # ================================================================
    # Component B: settling_stability — 速度管理（delta + 直接惩罚）
    # ================================================================
    # 接近度门控：距离越近，速度控制越重要
    # dist=0 → gate≈1.0，dist=1 → gate≈0.33，dist=2 → gate≈0.2
    prox_gate = 1.0 / (1.0 + 2.0 * next_dist)
    
    # delta：奖励减速
    speed_improvement = current_speed - next_speed
    
    # 直接速度惩罚（仅在接近目标时显著）
    speed_penalty = -4.0 * next_speed * prox_gate
    
    settling_stability = 2.0 * speed_improvement * (0.5 + prox_gate) + speed_penalty
    
    # ================================================================
    # Component C: attitude_constraint — 姿态管理（修复门控）
    # ================================================================
    # 两级门控：
    #   dist >= 1.5  → gate=0（完全不干预，允许大角度机动）
    #   dist <= 0.5  → gate=1（全力约束姿态）
    #   中间线性过渡
    if next_dist >= 1.5:
        att_gate = 0.0
    elif next_dist <= 0.5:
        att_gate = 1.0
    else:
        att_gate = 1.0 - (next_dist - 0.5) / 1.0  # 从 1.0 线性降到 0.0
    
    # delta 部分
    angle_improvement = abs(angle) - abs(n_angle)
    angvel_improvement = abs(ang_vel) - abs(n_ang_vel)
    
    # 直接姿态惩罚
    angle_penalty = -att_gate * (abs(n_angle) * 3.0 + abs(n_ang_vel) * 1.5)
    
    # delta 奖励
    angle_delta_reward = att_gate * (2.0 * angle_improvement + 1.0 * angvel_improvement)
    
    attitude_constraint = angle_delta_reward + angle_penalty
    
    # ================================================================
    # Component D: landing_contact — 安全着陆奖励
    # ================================================================
    # 放宽条件，使用连续信号而非硬门控
    near_target = next_dist < 0.3
    low_speed = next_speed < 0.5
    both_contacts = (n_left > 0.5 and n_right > 0.5)
    any_contact = (n_left > 0.5 or n_right > 0.5)
    
    landing_contact = 0.0
    if near_target and low_speed:
        if both_contacts:
            landing_contact = 10.0  # 完美双支撑着陆，每步大奖励
        elif any_contact:
            landing_contact = 2.0   # 单边接触，小鼓励
        # 接近且低速但无接触：不给奖励，不惩罚（让 settling 自然引导）
    
    # ================================================================
    # Component E: fuel_efficiency — 燃料效率
    # ================================================================
    fuel_efficiency = -0.05 * float(action != 0)
    
    # ================================================================
    # 汇总
    # ================================================================
    total_reward = (
        goal_proximity +
        settling_stability +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )
    
    components = {
        'goal_proximity': goal_proximity,
        'settling_stability': settling_stability,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency,
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