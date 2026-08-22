# 1. Search objective
- target_score: 200.000000
- current_score: -183.316556
- gap_to_target: 383.316556
- target_achievement_ratio: -91.658%

# 2. 上一轮奖励函数代码（该轮得分: -183.316556）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, which represents the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad (height)
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact flag (0 or 1)
    right_c = next_obs[7]   # right leg contact flag (0 or 1)

    # ------------------------------------------------------------------
    # Component A – goal proximity + soft landing  (main learning signal)
    # ------------------------------------------------------------------
    w_prox  = 2.0
    w_vel   = 1.0
    w_ang   = 2.0

    comp_prox  = w_prox * ( -abs(x) - abs(y) )
    comp_vel   = w_vel  * ( -vx**2 - vy**2 )
    comp_angle = w_ang  * ( -angle**2 - angvel**2 )

    # ------------------------------------------------------------------
    # Component B – contact quality (stability constraint)
    # ------------------------------------------------------------------
    w_contact = 10.0
    pos_thresh = 0.3
    vel_thresh = 0.3
    gate_x   = max(0.0, 1.0 - abs(x) / (pos_thresh + 1e-8))
    gate_y   = max(0.0, 1.0 - abs(y) / (pos_thresh + 1e-8))
    gate_vx  = max(0.0, 1.0 - abs(vx) / (vel_thresh + 1e-8))
    gate_vy  = max(0.0, 1.0 - abs(vy) / (vel_thresh + 1e-8))
    leg_contact = left_c * right_c
    comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy

    # ------------------------------------------------------------------
    # Component C – success proxy (task‑completion continuous signal)
    # ------------------------------------------------------------------
    # Changed from geometric mean of near‑perfect conditions to an additive,
    # continuous reward that activates gradually as each condition improves.
    # This provides a smoother gradient for the agent to follow.
    w_proxy = 2.0   # reduced from 20.0 to avoid overwhelming penalties

    # Distance thresholds that produce a score from 1 (perfect) down to 0
    score_x     = max(0.0, 1.0 - abs(x) / 1.0)
    score_y     = max(0.0, 1.0 - abs(y) / 1.0)
    score_vx    = max(0.0, 1.0 - abs(vx) / 0.5)
    score_vy    = max(0.0, 1.0 - abs(vy) / 0.5)
    score_angle = max(0.0, 1.0 - abs(angle) / 0.3)
    score_contact = (left_c + right_c) * 0.5   # [0,1]

    # Sum of independent scores – provides partial credit
    comp_proxy = w_proxy * (score_x + score_y + score_vx + score_vy +
                            score_angle + score_contact)

    # ------------------------------------------------------------------
    # Component D – thrust penalty (efficiency cost, secondary)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)   # any engine use

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_proxy + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "success_proxy": comp_proxy,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.35 | -121.00 | ✅ |
| 2 | 骨架变化: angle_penalty + contact_quality + goal_proximity + | — | 947.35 | -183.32 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. Subagent 行为诊断报告

## 1. 行为事实
- 得分：score=-183.32，range=[-284.46, -99.25]
- 终止方式：terminated 7/20，truncated 13/20
- 回合长度：len=947.35
- Checkpoint 轨迹：未提供（无数据）

## 2. 骨架演进表
| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | angle_penalty_ep_sum | angle_penalty_active | contact_quality_ep_sum | contact_quality_active | success_proxy_ep_sum | success_proxy_active | thrust_penalty_ep_sum | thrust_penalty_active |
|------|--------|------|------|------------------------|-------------------------|--------------------------|---------------------------|-----------------------|------------------------|-------------------------|---------------------------|-----------------------|------------------------|-------------------------|--------------------------|
| 1    | -121.00 | 68.35 | 20/0 | -148.505 | 100.0% | -93.226 | 99.9% | -43.986 | 100.0% | 0.766 | 0.2% | 1.432 | 0.1% | -0.215 | 6.3% |
| 2    | -183.32 | 947.35 | 7/13 | -3768.393 | 100.0% | -24.150 | 100.0% | -9.950 | 100.0% | 0.000 | 0.0% | 5792.231 | 100.0% | -47.368 | 100.0% |

## 3. 数字标记
- **死组件**：contact_quality（active_rate=0.0% < 2%）
- **淹没**：thrust_penalty（active_rate=100.0% > 90%，magnitude_share=0.5% < 3%）、velocity_penalty（active_rate=100.0%，magnitude_share=0.3%）、angle_penalty（active_rate=100.0%，magnitude_share=0.1%）
- **全覆盖惩罚**：goal_proximity（active_rate=100.0% > 95%，signed_share=-39.1%）、thrust_penalty（active_rate=100.0%，signed_share=-0.5%）、velocity_penalty（active_rate=100.0%，signed_share=-0.3%）、angle_penalty（active_rate=100.0%，signed_share=-0.1%）
- **量级对比**（每步均值 = episode_sum_mean ÷ len）  
  - 最大值（绝对值）：success_proxy = 6.114，goal_proximity = 3.977  
  - 最小值（绝对值）：contact_quality = 0.000，angle_penalty = 0.0105
- **修改变化（iter1→iter2）**  
  - episode_sum_mean Δ：goal_proximity -3619.888，velocity_penalty +69.076，angle_penalty +34.036，contact_quality -0.766，success_proxy +5790.799，thrust_penalty -47.153  
  - active_rate Δ：goal_proximity 0.0，velocity_penalty +0.1，angle_penalty 0.0，contact_quality -0.2，success_proxy +99.9，thrust_penalty +93.7

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, which represents the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad (height)
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact flag (0 or 1)
    right_c = next_obs[7]   # right leg contact flag (0 or 1)

    # ------------------------------------------------------------------
    # Component A – goal proximity + soft landing  (main learning signal)
    # ------------------------------------------------------------------
    # Encourage small positional errors, low velocities, upright attitude.
    w_prox  = 2.0
    w_vel   = 1.0
    w_ang   = 2.0

    comp_prox  = w_prox * ( -abs(x) - abs(y) )
    comp_vel   = w_vel  * ( -vx**2 - vy**2 )
    comp_angle = w_ang  * ( -angle**2 - angvel**2 )

    # ------------------------------------------------------------------
    # Component B – contact quality (stability constraint)
    # ------------------------------------------------------------------
    # Reward simultaneous leg contact, but only when the body is already
    # near the pad and moving slowly.  This discourages single‑leg touch.
    w_contact = 10.0

    pos_thresh = 0.3            # meters
    vel_thresh = 0.3            # m/s

    gate_x   = max(0.0, 1.0 - abs(x) / (pos_thresh + 1e-8))
    gate_y   = max(0.0, 1.0 - abs(y) / (pos_thresh + 1e-8))
    gate_vx  = max(0.0, 1.0 - abs(vx) / (vel_thresh + 1e-8))
    gate_vy  = max(0.0, 1.0 - abs(vy) / (vel_thresh + 1e-8))

    # Both legs must touch (binary product)
    leg_contact = left_c * right_c
    comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy

    # ------------------------------------------------------------------
    # Component C – success proxy (task‑completion soft signal)
    # ------------------------------------------------------------------
    # Geometric mean of near‑perfect conditions.  Only active when all
    # signals are close to ideal, providing a final shaping pull.
    w_proxy = 20.0

    eps_x     = 0.1
    eps_y     = 0.1
    eps_vx    = 0.15
    eps_vy    = 0.15
    eps_angle = 0.15   # radians

    score_x     = max(0.0, 1.0 - abs(x) / eps_x)
    score_y     = max(0.0, 1.0 - abs(y) / eps_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / eps_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / eps_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / eps_angle)
    score_contact = (left_c + right_c) * 0.5   # [0,1]

    # Use geometric mean to avoid product collapse.
    # 1e-12 avoids zero division.
    prod = (score_x * score_y * score_vx * score_vy * score_angle * score_contact)
    if prod > 1e-12:
        comp_proxy = w_proxy * (prod ** (1.0 / 6.0))
    else:
        comp_proxy = 0.0

    # ------------------------------------------------------------------
    # Component D – thrust penalty (efficiency cost, secondary)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)   # any engine use

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_proxy + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "success_proxy": comp_proxy,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| angle_penalty | -0.544 | -0.207 | +0.337 | 100.0% | 100.0% | -0.0pp |
| contact_quality | 0.021 | 0.896 | +0.875 | 0.4% | 15.2% | +14.8pp |
| goal_proximity | -2.156 | -1.565 | +0.590 | 100.0% | 100.0% | 0.0pp |
| success_proxy | 0.037 | 7.092 | +7.055 | 0.2% | 100.0% | +99.8pp |
| thrust_penalty | -0.024 | -0.040 | -0.016 | 49.0% | 80.4% | +31.5pp |
| velocity_penalty | -1.313 | -0.168 | +1.145 | 99.8% | 100.0% | +0.2pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -121.00 | -183.32 | -62.32 |
| 平均回合长度 | 68.3 | 947.4 | +879.0 |
| 终止回合数 | 20/20 | 7/20 | -13 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-183.316556, len=947.350000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-284.463711, -99.250079]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_proxy | 5792.231178 | 60.1% | 60.1% | 100.0% |
| goal_proximity | -3768.392629 | -39.1% | 39.1% | 100.0% |
| thrust_penalty | -47.367500 | -0.5% | 0.5% | 100.0% |
| velocity_penalty | -24.149682 | -0.3% | 0.3% | 100.0% |
| angle_penalty | -9.949903 | -0.1% | 0.1% | 100.0% |
| contact_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

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
