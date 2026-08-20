# 1. Search objective
- target_score: 200.000000
- current_score: -37.583039
- gap_to_target: 237.583039
- target_achievement_ratio: -18.792%

# 2. 上一轮奖励函数代码（该轮得分: -37.583039）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # Distance to target pad (0,0)
    dist_prev = (prev_x**2 + prev_y**2) ** 0.5
    dist_next = (next_x**2 + next_y**2) ** 0.5

    # 1. Goal distance progress (improvement delta)
    w_progress = 5.0
    progress = w_progress * (dist_prev - dist_next)

    # 2. Soft landing (proximity-gated velocity penalty)
    w_vel = 1.0
    k_dist = 20.0
    proximity_gate = 1.0 / (1.0 + k_dist * dist_next**2)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # 3. Orientation penalty (quadratic on tilt)
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # 4. Landing contact success (continuous, speed-dependent)
    w_contact = 5.0
    v_thresh = 0.5
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    if legs_on_ground:
        speed = (next_vx**2 + next_vy**2) ** 0.5
        contact_bonus = w_contact * max(0.0, 1.0 - speed / v_thresh)
    else:
        contact_bonus = 0.0

    # Aggregate
    total_reward = progress + vel_penalty + angle_penalty + contact_bonus

    components = {
        "goal_distance_progress": progress,
        "soft_landing": vel_penalty,
        "orientation_penalty": angle_penalty,
        "landing_contact_success": contact_bonus
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.35 | -121.00 | ✅ |
| 2 | 骨架变化: angle_penalty + contact_quality + goal_proximity + | — | 947.35 | -183.32 | ❌ |
| 3 | 骨架变化: angle_penalty + contact_quality + goal_proximity + | — | 68.40 | -120.25 | ✅ |
| 4 | 骨架变化: angle_penalty + contact_quality + goal_proximity + | — | 435.40 | -1471.59 | ❓ |
| 5 | 骨架变化: goal_distance_progress + landing_contact_success + | — | 963.75 | -63.97 | ✅ |
| 6 | 骨架变化: goal_distance_progress + landing_contact_success + | — | 947.65 | -37.58 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. Subagent 行为诊断报告

## 1. 行为事实
- 得分：score=-37.58，range=[-107.99, 127.78]
- 终止方式：terminated 3/20，truncated 17/20
- 回合长度：len=947.65
- Checkpoint 轨迹：无数据

## 2. 骨架演进表
| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | angle_penalty_ep_sum | angle_penalty_active | contact_quality_ep_sum | contact_quality_active | success_proxy_ep_sum | success_proxy_active | thrust_penalty_ep_sum | thrust_penalty_active | soft_success_ep_sum | soft_success_active | goal_distance_progress_ep_sum | goal_distance_progress_active | soft_landing_ep_sum | soft_landing_active | orientation_penalty_ep_sum | orientation_penalty_active | landing_contact_success_ep_sum | landing_contact_success_active |
|------|-------|-----|------|------------------------|------------------------|--------------------------|--------------------------|------------------------|------------------------|--------------------------|--------------------------|------------------------|------------------------|-------------------------|-------------------------|---------------------|---------------------|--------------------------------|-------------------------------|---------------------|-----------------------|-----------------------------|------------------------------|--------------------------------|--------------------------------|
| 1    | -121.00 | 68.35 | 20/0 | -148.505275 | 100.0% | -93.225748 | 99.9% | -43.986293 | 100.0% | 0.766166 | 0.2% | 1.432261 | 0.1% | -0.215000 | 6.3% | - | - | - | - | - | - | - | - | - | - |
| 2    | -183.32 | 947.35 | 7/13 | -3768.392629 | 100.0% | -24.149682 | 100.0% | -9.949903 | 100.0% | 0.000000 | 0.0% | 5792.231178 | 100.0% | -47.367500 | 100.0% | - | - | - | - | - | - | - | - | - | - |
| 3    | -120.25 | 68.40 | 20/0 | -156.891027 | 100.0% | -93.439696 | 99.8% | -38.994126 | 100.0% | 0.768669 | 0.2% | 1.434087 | 0.1% | -0.182500 | 5.3% | - | - | - | - | - | - | - | - | - | - |
| 4    | -1471.59 | 435.40 | 20/0 | 163.929425 | 100.0% | -376.896537 | 100.0% | -3.785140 | 100.0% | 0.000000 | 0.0% | - | - | -21.770000 | 100.0% | 12716.349643 | 96.2% | - | - | - | - | - | - | - | - |
| 5    | -63.97 | 963.75 | 1/19 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 3.779845 | 100.0% | -4.930737 | 100.0% | -1.218466 | 100.0% | 0.000000 | 0.0% |
| 6    | -37.58 | 947.65 | 3/17 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 4.980184 | 100.0% | -2.922441 | 100.0% | -2.135488 | 100.0% | 2.312616 | 0.1% |

## 3. 数字标记
- **主导**：无（所有组件 magnitude_share < 80%）
- **死组件**：landing_contact_success（active_rate = 0.1%）
- **淹没**：无（无组件 active_rate > 90% 且 magnitude_share < 3%）
- **全覆盖惩罚**：soft_landing（active_rate = 100% > 95%，signed_share = -20.1%）
- **量级对比**（每步均值 = episode_sum_mean / len）：
  - 最大两组件：goal_distance_progress（4.980184 / 947.65 ≈ 0.00525）、landing_contact_success（2.312616 / 947.65 ≈ 0.00244）
  - 最小两组件：soft_landing（-2.922441 / 947.65 ≈ -0.00308）、orientation_penalty（-2.135488 / 947.65 ≈ -0.00226）
- **修改变化**（iter 6 vs iter 5）：
  - goal_distance_progress：ep_sum Δ = +1.200339，active_rate Δ = 0.0pp
  - soft_landing：ep_sum Δ = +2.008296，active_rate Δ = 0.0pp
  - orientation_penalty：ep_sum Δ = -0.917022，active_rate Δ = 0.0pp
  - landing_contact_success：ep_sum Δ = +2.312616，active_rate Δ = +0.1pp

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| goal_distance_progress | 0.013 | 0.013 | +0.000 | 99.9% | 99.8% | -0.1pp |
| landing_contact_success | 0.915 | 2.123 | +1.208 | 45.8% | 44.5% | -1.2pp |
| orientation_penalty | -0.014 | -0.016 | -0.002 | 100.0% | 100.0% | -0.0pp |
| soft_landing | -0.021 | -0.023 | -0.002 | 100.0% | 99.9% | -0.1pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -63.97 | -37.58 | +26.39 |
| 平均回合长度 | 963.8 | 947.6 | -16.1 |
| 终止回合数 | 1/20 | 3/20 | +2 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-37.583039, len=947.650000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-107.997234, 127.780807]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_distance_progress | 4.980184 | 34.3% | 49.2% | 100.0% |
| soft_landing | -2.922441 | -20.1% | 20.1% | 100.0% |
| landing_contact_success | 2.312616 | 15.9% | 15.9% | 0.1% |
| orientation_penalty | -2.135488 | -14.7% | 14.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主体从视野顶部中央附近以随机初始速度出发，目标是尽可能快速、稳定地降落在中央目标垫上并停稳（双腿接触、位置居中对齐、姿态水平、速度几乎为零）。次级目标是整个过程中尽量少使用发动机推力（省燃料）。不应混淆的目标：单纯的“存活”或“保持平衡”不是主目标，而是着陆成功的前提；单纯的“燃料为零”也不是主目标，只是在成功着陆前提下才受鼓励。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float64 或 float32（由环境决定）  
- obs[0]: name=x_position, meaning=相对于目标垫水平位置的偏差坐标，reward_usable: true  
- obs[1]: name=y_position, meaning=相对于目标垫高度的垂直偏差坐标，reward_usable: true  
- obs[2]: name=x_velocity, meaning=水平线速度，reward_usable: true  
- obs[3]: name=y_velocity, meaning=垂直线速度，reward_usable: true  
- obs[4]: name=body_angle, meaning=机体倾斜角度（弧度制），reward_usable: true  
- obs[5]: name=angular_velocity, meaning=角速度，reward_usable: true  
- obs[6]: name=left_support_contact, meaning=左支撑腿接触标志（1.0 表示接触），reward_usable: true  
- obs[7]: name=right_support_contact, meaning=右支撑腿接触标志（1.0 表示接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: name=no_engine, meaning=不点火，无推力  
- action 1: name=left_orientation_engine, meaning=点燃左姿态发动机（产生一个力矩和微小横向力）  
- action 2: name=main_engine, meaning=点燃主发动机（产生向上推力）  
- action 3: name=right_orientation_engine, meaning=点燃右姿态发动机（产生反向力矩和微小横向力）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  - body_not_awake_or_settled：机体进入睡眠或完全稳定不动。当双腿安全接触目标垫且速度、姿态均在阈值内时可能触发，属于成功信号（但无法直接与失败情形区分）。  
- failure-like termination:  
  - crash_or_body_contact：机体与地面或障碍猛烈碰撞（可能由于姿态过度倾斜或高速冲击非垫区域）→ 典型失败。  
  - horizontal_position_outside_viewport：水平位置超出有效视野范围 → 失败。  
- ambiguous termination:  
  - body_not_awake_or_settled 若发生在垫外（比如在倾斜坡道上稳定）也可能被终止，此时实际是失败，但终止模式相同。  
- truncation: 未在代码片段中明确给出最大步数，但通常存在隐含截断。本分析不依赖截断信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 字典为空，无 success 字段）  
- explicit_failure_flag_available: false （同上）  
- allowed_info_fields: 无。info 恒为空字典，不可用任何 info 键。  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用。

## 7. 可用于奖励函数的信号
- position: obs[0] (x 偏差), obs[1] (y 偏差) → 目标接近程度
- velocity: obs[2] (vx), obs[3] (vy) → 移动速度，用于鼓励减速
- orientation: obs[4] (body_angle) → 倾斜程度，应接近0
- angular_velocity: obs[5] → 角旋转速率，可辅助惩罚快速旋转
- contact: obs[6] (左腿接触), obs[7] (右腿接触) → 用于鼓励双腿着垫、避免单腿着陆
- action/engine: action 值本身 → 可惩罚非零动作（燃料使用）或针对性地惩罚姿态发动机频繁使用
- other: 无额外可用信号

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty | -121.00 | -121.00 | 0.00 | 68.35 | angle_penalty=-0.544 contact_quality=0.021 goal_proximity=-2.156 success_proxy=0.037 thrust_penalty=-0.024 | new_best |
| 2 | angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty | -183.32 | -121.00 | -62.32 | 947.35 | angle_penalty=-0.207 contact_quality=0.896 goal_proximity=-1.565 success_proxy=7.092 thrust_penalty=-0.040 | no_meaningful_improvement |
| 3 | angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty | -120.25 | -120.25 | 0.00 | 68.40 | angle_penalty=-0.563 contact_quality=0.022 goal_proximity=-2.285 success_proxy=0.037 thrust_penalty=-0.022 | new_best |
| 4 | angle_penalty + contact_quality + goal_proximity + soft_success + thrust_penalty + velocity_penalty | -1471.59 | -120.25 | -1351.33 | 435.40 | angle_penalty=-0.217 contact_quality=0.033 goal_proximity=0.946 soft_success=28.967 thrust_penalty=-0.044 | unsolved_stagnation_fresh_restart |
| 5 | goal_distance_progress + landing_contact_success + orientation_penalty + soft_landing | -63.97 | -63.97 | 0.00 | 963.75 | goal_distance_progress=0.013 landing_contact_success=0.915 orientation_penalty=-0.014 soft_landing=-0.021 | new_best |
| 6 | goal_distance_progress + landing_contact_success + orientation_penalty + soft_landing | -37.58 | -37.58 | 0.00 | 947.65 | goal_distance_progress=0.013 landing_contact_success=2.123 orientation_penalty=-0.016 soft_landing=-0.023 | new_best |
