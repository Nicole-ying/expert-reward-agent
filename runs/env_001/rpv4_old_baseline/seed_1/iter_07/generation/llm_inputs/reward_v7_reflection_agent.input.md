# 1. Search objective
- target_score: 200.000000
- current_score: -59.116198
- gap_to_target: 259.116198

# 2. Current reward program (score: -59.116198)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x, y, vx, vy, angle, ang_vel, left, right = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft, nright = next_obs

    # 常用量
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5
    next_dist = (nx**2 + ny**2)**0.5
    next_speed = (nvx**2 + nvy**2)**0.5

    # ---- 1. 主学习信号：增强的 potential shaping ----
    w_dist = 10.0
    w_speed = 5.0
    potential_cur = -(w_dist * dist + w_speed * speed)
    potential_next = -(w_dist * next_dist + w_speed * next_speed)
    shaping_reward = potential_next - potential_cur

    # ---- 2. 姿态惩罚（角度不要过大） ----
    w_angle = 0.5
    angle_penalty = -w_angle * (angle**2)

    # ---- 3. 角速度惩罚（避免快速旋转） ----
    w_angvel = 0.1
    ang_vel_penalty = -w_angvel * (ang_vel**2)

    # ---- 4. 燃料效率惩罚 ----
    if action == 2:           # 主发动机
        fuel_penalty = -0.15
    elif action in (1, 3):    # 左/右姿态引擎
        fuel_penalty = -0.02
    else:                     # 无推力
        fuel_penalty = 0.0

    # ---- 5. 着陆成功大奖励 ----
    # 条件：双腿接触，速度、位置、角度都十分接近零
    contact_success = nleft and nright and (next_speed < 0.5) and (abs(nx) < 0.3) and (abs(ny) < 0.3) and (abs(nangle) < 0.1)
    success_bonus = 50.0 if contact_success else 0.0

    # ---- 6. 猛烈着陆惩罚（双腿接触但不安全） ----
    # 如果双腿接触但速度过高或角度过大，视为 crash
    crash_condition = nleft and nright and ((next_speed > 1.5) or (abs(nangle) > 0.3))
    crash_penalty = -10.0 if crash_condition else 0.0

    # ---- 7. 边界危险惩罚（防止飞出场外） ----
    # 横向边界：避免水平飞出视口
    boundary_x_penalty = 0.0
    if abs(nx) > 2.0:
        boundary_x_penalty += -10.0
    if abs(nx) > 4.0:
        boundary_x_penalty += -40.0  # 累计 -50

    # 纵向下边界：避免掉出视图下方（撞击地面以外区域）
    boundary_y_penalty = 0.0
    if ny < -1.0:
        boundary_y_penalty += -10.0
    if ny < -3.0:
        boundary_y_penalty += -40.0

    total_reward = (
        shaping_reward
        + angle_penalty
        + ang_vel_penalty
        + fuel_penalty
        + success_bonus
        + crash_penalty
        + boundary_x_penalty
        + boundary_y_penalty
    )

    components = {
        "shaping_reward": shaping_reward,
        "angle_penalty": angle_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "fuel_penalty": fuel_penalty,
        "success_bonus": success_bonus,
        "crash_penalty": crash_penalty,
        "boundary_x_penalty": boundary_x_penalty,
        "boundary_y_penalty": boundary_y_penalty,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-59.116198, len=72.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-114.214464, -9.895323]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| shaping_reward | 11.253881 | 44.7% | 46.5% | 100.0% |
| success_bonus | 10.000000 | 39.7% | 39.7% | 0.3% |
| fuel_penalty | -2.640500 | -10.5% | 10.5% | 29.2% |
| crash_penalty | -0.500000 | -2.0% | 2.0% | 0.1% |
| angle_penalty | -0.301930 | -1.2% | 1.2% | 100.0% |
| ang_vel_penalty | -0.039459 | -0.2% | 0.2% | 100.0% |
| boundary_x_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| boundary_y_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 13/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 4. Environment facts
## 1. 任务目标
主体为一个2D飞行器（vehicle-like），初始位置在视口顶部中央附近，带有随机初始作用力。  
核心目标是 **尽快到达并稳定停靠在中央目标着陆平台上**，同时尽可能少使用引擎推力。  
智能体需要学习：趋近目标、降低线速度、保持姿态竖直、实现两腿同时安全接触（软着陆）。  
不应混淆的目标：单纯求快但忽略安全着陆，或一味省燃料而无法到达目标。

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: [8]  
- **dtype**: 默认为 float64 或 float32（取决于环境实现，但通常为 float64）  
- **obs[0]**: `x_position`，水平方向相对于目标着陆平台中心的偏移量，可用于奖励趋近目标，reward_usable: true  
- **obs[1]**: `y_position`，垂直方向相对于平台高度（接触面）的偏移量，reward_usable: true  
- **obs[2]**: `x_velocity`，水平线速度，reward_usable: true  
- **obs[3]**: `y_velocity`，垂直线速度，reward_usable: true  
- **obs[4]**: `body_angle`，机体倾角（如弧度），reward_usable: true  
- **obs[5]**: `angular_velocity`，角速度，reward_usable: true  
- **obs[6]**: `left_support_contact`，左支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true  
- **obs[7]**: `right_support_contact`，右支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  
- **动作/索引 0**: `no_engine` (不做任何事)，语义：无推力，用于滑行或停靠后保持  
- **动作/索引 1**: `left_orientation_engine` (左姿态引擎)，语义：产生逆时针或顺时针旋转力矩（具体方向取决于环境）  
- **动作/索引 2**: `main_engine` (主引擎)，语义：产生纵向（向上）推力  
- **动作/索引 3**: `right_orientation_engine` (右姿态引擎)，语义：产生与左引擎相反的旋转力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` 可能表示机体静止稳定，通常意味着已着陆并静止，可能视为成功；  
- **failure-like termination**: `crash_or_body_contact` （如机身碰撞地面或平台以外部分）、`horizontal_position_outside_viewport` （漂出水平边界）很可能表示失败；  
- **ambiguous termination**: `crash_or_body_contact` 若接触平台但判定为 crash 则为失败，但描述未区分成功接触与失败接触；我们需要从“到达并稳定停靠”推断理想行为是两腿接触且低速，但不能直接从终止信号中获知成功。  
- **truncation**: 源代码未显示截断（max_steps），但多数环境有步数限制，此处未给出，视为不存在或不可直接用于奖励。  

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 空字典 `{}`，无可用字段 。  
- **forbidden_or_uncertain_info_fields**: 任何 info 字段均不可用（因为提供的信息为空）。终止原因也不能从 info 获取。

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position`（可直接组成距离度量）  
- **velocity**: `x_velocity`, `y_velocity`（线速度）  
- **orientation**: `body_angle`, `angular_velocity`（角速度）  
- **contact**: `left_support_contact`, `right_support_contact`（双腿触地标志），可构造着陆条件  
- **action/engine**: 可以基于动作选择构建惩罚（如鼓励 no_engine，惩罚主引擎和姿态引擎）  
- **other**: 无