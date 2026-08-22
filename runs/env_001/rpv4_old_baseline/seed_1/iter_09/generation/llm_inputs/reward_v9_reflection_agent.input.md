# 1. Search objective
- target_score: 200.000000
- current_score: -113.486473
- gap_to_target: 313.486473

# 2. Current reward program (score: -113.486473)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包状态
    x, y, vx, vy, angle, ang_vel, left, right = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft, nright = next_obs

    # 常用量
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5
    next_dist = (nx**2 + ny**2)**0.5
    next_speed = (nvx**2 + nvy**2)**0.5

    # ---- 势能塑造：鼓励接近目标、保持低速 ----
    w_dist = 5.0
    w_speed = 5.0          # 适度惩罚速度，不阻碍必要移动
    potential_cur = -(w_dist * dist + w_speed * speed)
    potential_next = -(w_dist * next_dist + w_speed * next_speed)
    shaping = potential_next - potential_cur

    # ---- 姿态与角速度惩罚 ----
    angle_penalty = -0.5 * abs(nangle)       # 鼓励竖直
    angvel_penalty = -0.05 * abs(nang_vel)   # 抑制旋转

    # ---- 燃料效率惩罚 ----
    if action == 2:           # 主引擎
        fuel_penalty = -0.15
    elif action in (1, 3):    # 姿态引擎
        fuel_penalty = -0.02
    else:                     # 无推力
        fuel_penalty = 0.0

    # ---- 步数惩罚，推动尽快完成任务 ----
    step_penalty = -0.02

    # ---- 双腿接触持续奖励（鼓励稳定软着陆） ----
    contact_continuous = 0.0
    if nleft and nright:
        speed_factor = max(0.0, 1.0 - next_speed)          # 速度越慢越好，线性衰减到0
        angle_factor = max(0.0, 1.0 - abs(nangle) / 0.3)  # 倾角小于0.3 rad 时线性
        contact_continuous = 1.0 * speed_factor * angle_factor

    # ---- 成功软着陆大奖励 ----
    contact_success = (nleft and nright and
                       (next_speed < 0.5) and
                       (abs(nangle) < 0.2))
    success_bonus = 100.0 if contact_success else 0.0

    # ---- 猛烈着陆惩罚 ----
    crash_condition = nleft and nright and ((next_speed > 2.0) or (abs(nangle) > 0.5))
    crash_penalty = -20.0 if crash_condition else 0.0

    # 汇总
    total_reward = (shaping +
                    angle_penalty +
                    angvel_penalty +
                    fuel_penalty +
                    step_penalty +
                    contact_continuous +
                    success_bonus +
                    crash_penalty)

    components = {
        "shaping": shaping,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "fuel_penalty": fuel_penalty,
        "step_penalty": step_penalty,
        "contact_continuous": contact_continuous,
        "success_bonus": success_bonus,
        "crash_penalty": crash_penalty
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-113.486473, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582396, -95.638888]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 40.000000 | 77.7% | 77.7% | 0.6% |
| shaping | 5.144894 | 10.0% | 16.2% | 100.0% |
| step_penalty | -1.368000 | -2.7% | 2.7% | 100.0% |
| angle_penalty | -1.198616 | -2.3% | 2.3% | 100.0% |
| contact_continuous | 0.277654 | 0.5% | 0.5% | 0.9% |
| angvel_penalty | -0.227795 | -0.4% | 0.4% | 100.0% |
| fuel_penalty | -0.064000 | -0.1% | 0.1% | 4.7% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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