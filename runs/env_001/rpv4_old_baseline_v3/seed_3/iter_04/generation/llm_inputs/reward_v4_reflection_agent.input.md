# 1. Search objective
- target_score: 200.000000
- current_score: -121.569900
- gap_to_target: 321.569900

# 2. Current reward program (score: -121.569900)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: Remove proximity_bonus (local optimum trap), fix slam_cost (threshold-based),
    add descent_reward, boost progress/contact/landing, add step_cost.
    
    Core insight: v2 failed because proximity_bonus let agent hover forever near pad
    collecting ~0.25/step, while slam_cost punished ALL downward motion — so agent
    never descended to make leg contact. This version removes the trap completely.
    """
    # --- unpack ---
    x_c, y_c = obs[0], obs[1]
    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n = next_obs[4]
    angvel_n = next_obs[5]
    lc_n = next_obs[6]
    rc_n = next_obs[7]

    # --- 1. progress: potential-based, strong gradient toward (0,0) ---
    # Self-extinguishes at target — no local optimum
    dist_c = (x_c ** 2 + y_c ** 2) ** 0.5
    dist_n = (x_n ** 2 + y_n ** 2) ** 0.5
    progress = 10.0 * (dist_c - dist_n)

    # --- 2. x_centering: extra pressure to align horizontally ---
    x_improvement = abs(x_c) - abs(x_n)
    x_bonus = 2.0 * x_improvement

    # --- 3. descent_reward: reward controlled descent when x is near center ---
    # This replaces the old proximity_bonus — it pushes agent DOWN onto the pad
    x_near = max(0.0, 1.0 - abs(x_n) / 2.0)
    descending = max(0.0, y_c - y_n)
    speed_ok = max(0.0, 1.0 - (vx_n ** 2 + vy_n ** 2) ** 0.5 / 1.5)
    descent_reward = 3.0 * x_near * descending * speed_ok

    # --- 4. contact_reward: substantial per-leg reward ---
    # Only fires when legs actually touch — discoverable once agent descends
    contact_reward = 8.0 * (lc_n + rc_n)

    # --- 5. landing_bonus: big bonus for both legs on pad near origin ---
    near_origin = max(0.0, 1.0 - (x_n ** 2 + y_n ** 2) ** 0.5 / 0.8)
    landing_bonus = 20.0 * lc_n * rc_n * near_origin

    # --- 6. slam_cost: ONLY penalize EXCESSIVE downward speed near ground ---
    # Safe descent (< 0.5 units/step downward) is NOT penalized
    # This is the critical fix — v2 penalized ALL downward speed
    near_ground = max(0.0, 1.0 - abs(y_n) / 2.0)
    excess_downward = max(0.0, -vy_n - 0.5)
    slam_cost = -10.0 * near_ground * excess_downward

    # --- 7. orient_cost: mild regularizer (kept from v2) ---
    orient_cost = -0.15 * abs(angle_n) - 0.03 * (angvel_n ** 2)

    # --- 8. step_cost: tiny per-step penalty to discourage infinite hovering ---
    step_cost = -0.03

    # --- 9. fuel_cost: reduced from v2 to avoid discouraging necessary engine use ---
    if action == 2:
        fuel_cost = -0.04
    elif action in (1, 3):
        fuel_cost = -0.02
    else:
        fuel_cost = 0.0

    # --- assemble ---
    total_reward = (
        progress
        + x_bonus
        + descent_reward
        + contact_reward
        + landing_bonus
        + slam_cost
        + orient_cost
        + step_cost
        + fuel_cost
    )

    components = {
        'progress': progress,
        'x_bonus': x_bonus,
        'descent_reward': descent_reward,
        'contact_reward': contact_reward,
        'landing_bonus': landing_bonus,
        'slam_cost': slam_cost,
        'orient_cost': orient_cost,
        'step_cost': step_cost,
        'fuel_cost': fuel_cost,
    }

    return total_reward, components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-121.569900, len=546.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-234.415641, 99.438270]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 222.800000 | 69.5% | 69.5% | 3.1% |
| landing_bonus | 45.407138 | 14.2% | 14.2% | 1.9% |
| fuel_cost | -16.637000 | -5.2% | 5.2% | 96.6% |
| step_cost | -16.393500 | -5.1% | 5.1% | 100.0% |
| progress | 4.175703 | 1.3% | 3.5% | 100.0% |
| orient_cost | -3.777960 | -1.2% | 1.2% | 100.0% |
| descent_reward | 2.445011 | 0.8% | 0.8% | 95.6% |
| x_bonus | -1.887922 | -0.6% | 0.6% | 100.0% |
| slam_cost | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Checkpoint evaluations (every 10% of training)
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
| -667.8 | -231.9 | -158.6 | -130.0 | -68.1 | -112.6 | -95.0 | -50.4 | -143.2 | -136.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_reward | 0.000 | 0.000 | 0.000 | 125.600 | 272.000 | 164.400 | 232.000 | 364.400 | 0.000 | 88.800 |
| descent_reward | 0.056 | 0.186 | 1.617 | 2.437 | 2.704 | 2.651 | 2.694 | 2.827 | 2.473 | 2.482 |
| fuel_cost | -4.025 | -6.700 | -10.932 | -13.849 | -12.492 | -14.191 | -16.941 | -16.106 | -16.769 | -18.049 |
| landing_bonus | 0.000 | 0.000 | 0.000 | 66.617 | 149.810 | 2.747 | 33.141 | 129.514 | 0.000 | 37.293 |
| orient_cost | -12.305 | -4.356 | -3.342 | -3.391 | -3.522 | -3.544 | -4.411 | -3.963 | -2.954 | -3.591 |
| progress | -8.044 | -5.195 | 2.270 | 4.027 | 5.849 | 4.324 | 5.523 | 6.102 | 3.989 | 4.221 |
| slam_cost | -29.750 | -1.802 | 0.000 | 0.000 | -0.004 | -0.028 | 0.000 | -0.004 | 0.000 | 0.000 |
| step_cost | -3.225 | -6.281 | -10.399 | -13.953 | -12.763 | -14.251 | -16.784 | -16.165 | -16.733 | -17.600 |
| x_bonus | -2.028 | -2.000 | -1.998 | -1.926 | -1.611 | -1.890 | -1.669 | -1.568 | -1.929 | -1.880 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_reward | 0.0% | 0.0% | 0.0% | 1.8% | 5.3% | 2.8% | 3.5% | 5.2% | 0.0% | 1.2% |
| descent_reward | 6.4% | 19.5% | 98.6% | 97.5% | 96.7% | 95.8% | 94.0% | 95.0% | 95.4% | 95.4% |
| fuel_cost | 100.0% | 100.0% | 100.0% | 92.7% | 92.8% | 93.6% | 96.5% | 95.1% | 92.9% | 97.2% |
| landing_bonus | 0.0% | 0.0% | 0.0% | 1.5% | 2.4% | 0.2% | 1.6% | 2.9% | 0.0% | 0.6% |
| orient_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress | 100.0% | 100.0% | 100.0% | 100.0% | 99.5% | 100.0% | 100.0% | 99.4% | 100.0% | 100.0% |
| slam_cost | 6.6% | 1.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% |
| step_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| x_bonus | 100.0% | 100.0% | 100.0% | 100.0% | 99.5% | 100.0% | 100.0% | 99.4% | 100.0% | 100.0% |

# 4. Environment facts
## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 推测为 float32  
- obs[0]: x_position，水平坐标（相对于目标平台的 x 偏移），reward_usable: true  
- obs[1]: y_position，垂直坐标（相对于平台高度，即飞行器底部到平台表面的距离），reward_usable: true  
- obs[2]: x_velocity，水平线速度，reward_usable: true  
- obs[3]: y_velocity，垂直线速度，reward_usable: true  
- obs[4]: body_angle，身体倾斜角度（弧度 or 归一化角度），reward_usable: true  
- obs[5]: angular_velocity，角速度，reward_usable: true  
- obs[6]: left_support_contact，左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true  
- obs[7]: right_support_contact，右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true  

备注：所有维度均为连续数字，接触标志虽为 0/1 但以浮点形式给出。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine（不点火，靠惯性/重力运动）  
- action 1: left_orientation_engine（点火左向姿态调节引擎，产生向左旋转力矩）  
- action 2: main_engine（点火主引擎，产生向上的推力，通常用于减速/悬停）  
- action 3: right_orientation_engine（点火右向姿态调节引擎，产生向右旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 可能代表身体已静止，结合 y_position≈0 且左右支撑接触为 1 时可视为成功着陆。但 termination 本身是模糊的，因为 crash 后也可能导致静止。  
- failure-like termination: crash_or_body_contact 中“crash”部分，horizontal_position_outside_viewport（水平超出视口）  
- ambiguous termination: body_not_awake_or_settled 的具体触发条件未知，可能是身体进入休眠/静止状态；crash_or_body_contact 中的 body_contact 可能包含正常着陆接触，需依赖位置判断。  
- truncation: 未提供  

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 字段）  
- explicit_failure_flag_available: false（同样无 failure 字段）  
- allowed_info_fields: 无（info 返回空字典 {}）  
- forbidden_or_uncertain_info_fields: 所有 internal 字段均不可用，如 termination_reason 等

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: action 值（0~3），可用于燃料惩罚或推力鼓励
- other: 通过 next_obs 计算上述量的变化率（如速度变化、位置变化）