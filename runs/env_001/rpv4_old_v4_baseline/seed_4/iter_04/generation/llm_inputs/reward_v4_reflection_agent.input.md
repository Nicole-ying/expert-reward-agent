下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (score=88.521717)
# Training Feedback

## Final-policy outcome
score=88.521717, len=892.550000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[14.625518, 267.349190]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 3090.000000 | 80.0% | 80.0% | 11.9% |
| goal_proximity | 675.197383 | 17.5% | 17.5% | 100.0% |
| settling_stability | -46.442001 | -1.2% | 1.2% | 100.0% |
| fuel_efficiency | -27.457500 | -0.7% | 0.7% | 61.5% |
| attitude_constraint | -21.251541 | -0.6% | 0.6% | 100.0% |

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
| -74.5 | -5.9 | 72.2 | 73.3 | 80.9 | -40.5 | 10.6 | 94.9 | 134.0 | 74.9 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -2.880 | -25.084 | -18.940 | -19.409 | -18.724 | -17.949 | -23.081 | -23.701 | -22.210 | -24.514 |
| fuel_efficiency | -1.068 | -31.323 | -28.805 | -28.778 | -16.900 | -14.093 | -28.658 | -28.290 | -24.288 | -30.045 |
| goal_proximity | 18.012 | 544.464 | 448.327 | 574.926 | 174.528 | 165.758 | 484.994 | 591.023 | 572.954 | 661.030 |
| landing_contact | 6.250 | 12.500 | 132.500 | 2356.250 | 522.500 | 490.000 | 2342.500 | 2992.500 | 2378.750 | 3037.500 |
| settling_stability | -17.377 | -42.198 | -42.095 | -39.274 | -31.897 | -31.141 | -43.145 | -43.811 | -45.612 | -45.542 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 30.3% | 62.6% | 71.3% | 65.5% | 73.0% | 64.1% | 68.9% | 67.2% | 62.5% | 65.1% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 0.4% | 0.1% | 0.6% | 9.5% | 4.1% | 3.8% | 9.7% | 12.7% | 11.2% | 11.3% |
| settling_stability | 100.0% | 100.0% | 99.9% | 100.0% | 99.9% | 99.9% | 100.0% | 100.0% | 99.9% | 100.0% |
# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v3。
    
    v2→v3 修改：landing_contact 从 per-step state reward 改为一次性 transition bonus。
    
    问题诊断（v2）：landing_contact 作为 per-step 状态奖励（2.0/腿/步），agent 双足着地后
    每步持续收割 ~4.0，导致 magnitude_share 高达 54%。checkpoint 趋势中 cp70% 出现 824.5 
    的极端 spike，随后剧烈振荡，是典型的"占据好状态持续获奖"exploit。
    
    v3 方案：
    - landing_contact: 改为 transition-based — 仅在每条腿从无接触→有接触的瞬间给予
      一次性 25.0/腿 奖励。消除持续累积 exploit，同时保留着陆引导信号。
    - 其余组件（goal_proximity, settling_stability, attitude_constraint, fuel_efficiency）不变。
    """
    
    # ========== 信号提取 ==========
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    # ========== 距离与速度 ==========
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    next_speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    
    # ========== Component A: goal_proximity ==========
    dist_improvement = current_dist - next_dist
    proximity_bonus = 1.0 * (2.718281828 ** (-next_dist / 0.5))
    goal_proximity = 3.0 * dist_improvement + proximity_bonus
    
    # ========== 邻近门控（多个组件共用）==========
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)
    
    # ========== Component B: settling_stability ==========
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier
    
    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)
    
    # ========== Component D: landing_contact (v3: transition-based) ==========
    # 仅在接近目标 + 低速时，对「从无接触到有接触」的转移瞬间给予一次性奖励。
    # 消除 v2 中 per-step 持续累积导致的 exploitation（cp70% spike=824.5）。
    near_target = next_dist < 0.5
    low_speed = next_speed < 1.0
    
    landing_contact = 0.0
    if near_target and low_speed:
        # 左腿：从无接触 → 有接触
        if left_contact < 0.5 and next_left_contact > 0.5:
            landing_contact += 25.0
        # 右腿：从无接触 → 有接触
        if right_contact < 0.5 and next_right_contact > 0.5:
            landing_contact += 25.0
    
    # ========== Component E: fuel_efficiency ==========
    fuel_efficiency = -0.05 * float(action != 0)
    
    # ========== 汇总 ==========
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
        'fuel_efficiency': fuel_efficiency
    }
    
    return float(total_reward), components
```

# 反思规则

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

分析完每个组件后再写代码。

### 输出要求

Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
