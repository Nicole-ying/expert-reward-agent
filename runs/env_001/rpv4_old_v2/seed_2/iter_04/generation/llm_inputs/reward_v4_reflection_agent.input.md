# ⚠️ 上一版代码验证失败
错误信息：Reward v4 failed validation: 缺少准确函数签名; 代码无法解析 AST: invalid character '：' (U+FF1A) (<unknown>, line 1) (record: runs/env_001/rpv4_old_v2/seed_2/iter_04/generation/validations/reward_v4.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
```python
1. `evidence`：所有20个评估回合均在约68步因失败终止（crash/出界），得分-120完全由goal_proximity(-156.9)、velocity_penalty(-93.4)、angle_penalty(-39.0)这三个全覆盖惩罚构成；success_proxy和contact_quality激活率分别为0.1%与0.2%，属于完全死组件，无法提供正向引导。
2. `behavior_diagnosis`：策略学到的最优行为是尽快触发失败终止以截断持续惩罚的累积，完全没有进行着陆尝试的任何趋势。
3. `signal_completeness`：当前奖励函数缺少可达的、连续的进度正信号，唯一的正向奖励被严苛的乘积门控压至死状态，导致“接近并着陆”这一必要职责完全空缺。
4. `selected_level`：Level 3（重建主信号框架），触发条件是历史所有迭代均未取得可接受分数且Subagent诊断确认当前正向组件已死，整体方向错误。
5. `selected_intervention`：唯一目标为整体框架——将goal_proximity由负惩罚翻转为正向接近奖励，引入全新的加和型soft_success组件（取代死掉的几何平均proxy），移除contact_quality的乘积门控改为位置衰减的接触奖励，并降低速度/角度惩罚权重使正向引导不被淹没。
6. `falsifiable_hypothesis`：通过提供连续、可达的正向接近奖励和部分条件满足即可触发的soft_success，策略将转变为尝试停留并靠近目标垫，从而观察到回 合长度显著上升、soft_success激活率飙升、以及至少不再总是主动自杀。
7. `expected_next_round`：episode_length预计从68大幅提升至300+，soft_success active_rate从0.1%升至>60%，terminated比率可能仍高但应出现部分由于接触稳定而触发的body_not_awake终止，score有望收敛到-50以上甚至正数。
8. `main_risk`：若soft_success中的速度/位置阈值设置过宽，而速度/推力惩罚又不足以抑制hovering行为，策略可能学会在垫上方悬停而不真正着陆，导致len延长但着陆成功率不提升。
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact
    right_c = next_obs[7]   # right leg contact

    # ------------------------------------------------------------------
    # Component A – goal proximity (positive, bounded approach reward)
    # ------------------------------------------------------------------
    w_prox = 2.0
    dist = (x**2 + y**2) ** 0.5
    # bounded reward: 1.0 when dist=0, decays to 0 when dist large
    comp_prox = w_prox * (1.0 - dist / (dist + 1.0))

    # ------------------------------------------------------------------
    # Component B – velocity penalty (soft brake)
    # ------------------------------------------------------------------
    w_vel = 0.5
    comp_vel = w_vel * (-vx**2 - vy**2)

    # ------------------------------------------------------------------
    # Component C – angle penalty
    # ------------------------------------------------------------------
    w_ang = 1.0
    comp_angle = w_ang * (-angle**2 - 0.1 * angvel**2)

    # ------------------------------------------------------------------
    # Component D – contact quality (leg contact gated by distance)
    # ------------------------------------------------------------------
    w_contact = 2.0
    gate = 1.0 / (1.0 + dist)   # near pad -> ~1, far away -> small
    leg_contact = left_c * right_c   # bilateral contact only
    comp_contact = w_contact * leg_contact * gate

    # ------------------------------------------------------------------
    # Component E – soft success shaping (sum of linear scores)
    # ------------------------------------------------------------------
    w_soft = 30.0

    tol_x     = 0.1
    tol_y     = 0.1
    tol_vx    = 0.2
    tol_vy    = 0.2
    tol_angle = 0.2

    score_x     = max(0.0, 1.0 - abs(x) / tol_x)
    score_y     = max(0.0, 1.0 - abs(y) / tol_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / tol_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / tol_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / tol_angle)
    score_leg   = (left_c + right_c) * 0.5   # linear in [0,1]

    soft_sum = (score_x + score_y + score_vx + score_vy +
                score_angle + score_leg)
    comp_soft = w_soft * soft_sum

    # ------------------------------------------------------------------
    # Component F – thrust penalty (fuel efficiency)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_soft + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "soft_success": comp_soft,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```

# 1. Search objective
- target_score: 200.000000
- current_score: -120.253372
- gap_to_target: 320.253372
- target_achievement_ratio: -60.127%

# 2. 上一轮奖励函数代码（该轮得分: -120.253372）
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
    # Component A – goal proximity (bounded, soft guidance)
    # ------------------------------------------------------------------
    # Changed from unbounded -abs(x)-abs(y) to a bounded penalty that
    # saturates far from the pad.
    w_prox  = 5.0
    dist_sq = x**2 + y**2
    comp_prox = w_prox * ( - dist_sq / (1.0 + dist_sq) )

    # ------------------------------------------------------------------
    # Component B – velocity penalty
    # ------------------------------------------------------------------
    w_vel = 1.0
    comp_vel = w_vel * ( -vx**2 - vy**2 )

    # ------------------------------------------------------------------
    # Component C – angle penalty
    # ------------------------------------------------------------------
    w_ang = 2.0
    comp_angle = w_ang * ( -angle**2 - angvel**2 )

    # ------------------------------------------------------------------
    # Component D – contact quality (stability constraint)
    # ------------------------------------------------------------------
    # Reward simultaneous leg contact, gated by position and velocity.
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
    # Component E – success proxy (geometric mean of near‑perfect conditions)
    # ------------------------------------------------------------------
    w_proxy = 20.0

    eps_x     = 0.1
    eps_y     = 0.1
    eps_vx    = 0.15
    eps_vy    = 0.15
    eps_angle = 0.15

    score_x     = max(0.0, 1.0 - abs(x) / eps_x)
    score_y     = max(0.0, 1.0 - abs(y) / eps_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / eps_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / eps_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / eps_angle)
    score_contact = (left_c + right_c) * 0.5

    prod = (score_x * score_y * score_vx * score_vy * score_angle * score_contact)
    if prod > 1e-12:
        comp_proxy = w_proxy * (prod ** (1.0 / 6.0))
    else:
        comp_proxy = 0.0

    # ------------------------------------------------------------------
    # Component F – thrust penalty (efficiency)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)

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

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 组件逐项对比
（第一轮反思，无历史对比）

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-120.253372, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335058, -98.281266]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -156.891027 | -53.8% | 53.8% | 100.0% |
| velocity_penalty | -93.439696 | -32.0% | 32.0% | 99.8% |
| angle_penalty | -38.994126 | -13.4% | 13.4% | 100.0% |
| success_proxy | 1.434087 | 0.5% | 0.5% | 0.1% |
| contact_quality | 0.768669 | 0.3% | 0.3% | 0.2% |
| thrust_penalty | -0.182500 | -0.1% | 0.1% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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