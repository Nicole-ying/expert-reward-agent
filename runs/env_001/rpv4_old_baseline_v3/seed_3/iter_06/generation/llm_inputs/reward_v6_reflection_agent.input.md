# 1. Search objective
- target_score: 200.000000
- current_score: 171.399798
- gap_to_target: 28.600202

# 2. Current reward program (score: 171.399798)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y = obs[0], obs[1]
    xv, yv = obs[2], obs[3]
    angle = obs[4]
    # angv = obs[5]  # not used directly, could be added later if needed
    lc, rc = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    # nangv = next_obs[5]
    nlc, nrc = next_obs[6], next_obs[7]

    # Euclidean distances
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # ================= A. Progress: delta distance =================
    # Dense reward for moving toward the target platform (0,0)
    w_progress = 10.0
    progress_reward = w_progress * (dist - next_dist)

    # ================= B. Velocity penalty near target =================
    # Apply only when close to the target (dist < d_threshold)
    d_threshold = 0.5  # meters, safe distance to start slowing down
    proximity_factor = max(0.0, 1.0 - dist / d_threshold)  # 1 at target, 0 beyond threshold

    safe_vx = 0.2
    safe_vy = 0.3
    excess_vx = max(0.0, abs(xv) - safe_vx)
    excess_vy = max(0.0, abs(yv) - safe_vy)
    w_vel_penalty = 2.0
    velocity_penalty = -w_vel_penalty * proximity_factor * (excess_vx + excess_vy)

    # ================= C. Touchdown event reward =================
    touchdown_reward = 0.0
    prev_double_contact = (lc > 0.5 and rc > 0.5)
    next_double_contact = (nlc > 0.5 and nrc > 0.5)
    if not prev_double_contact and next_double_contact:
        # Check soft landing requirements on the touchdown step
        speed_ok = abs(nxv) < safe_vx and abs(nyv) < safe_vy
        angle_ok = abs(nangle) < 0.2  # radians, near upright
        if speed_ok and angle_ok:
            touchdown_reward = 50.0

    total_reward = progress_reward + velocity_penalty + touchdown_reward

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "touchdown_reward": touchdown_reward
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=171.399798, len=536.650000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-1.659690, 264.867974]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| touchdown_reward | 1737.500000 | 99.2% | 99.2% | 6.5% |
| progress_reward | 11.954032 | 0.7% | 0.8% | 98.6% |
| velocity_penalty | -0.141551 | -0.0% | 0.0% | 3.2% |

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
| -451.6 | -237.3 | -206.6 | 20.0 | 98.5 | 128.7 | 138.2 | 148.3 | 129.9 | 157.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| progress_reward | -0.032 | -1.293 | -2.927 | 12.580 | 8.553 | 10.679 | 12.569 | 11.480 | 10.968 | 12.032 |
| touchdown_reward | 0.000 | 0.000 | 0.000 | 1765.000 | 492.500 | 1135.000 | 1722.500 | 1440.000 | 1727.500 | 2287.500 |
| velocity_penalty | 0.000 | 0.000 | 0.000 | 0.000 | -0.322 | -0.290 | -0.319 | -0.219 | -0.039 | -0.197 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.2% | 99.1% | 98.8% | 99.1% | 98.8% |
| touchdown_reward | 0.0% | 0.0% | 0.0% | 3.7% | 2.1% | 4.1% | 4.9% | 5.5% | 6.0% | 7.9% |
| velocity_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 4.0% | 2.3% | 0.7% | 2.0% | 1.9% | 3.3% |

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