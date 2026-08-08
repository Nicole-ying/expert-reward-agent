# 1. Search objective
- target_score: 200.000000
- current_score: -91.705574
- gap_to_target: 291.705574

# 2. Current reward program (score: -91.705574)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs and next_obs fields: 0-x, 1-y, 2-vx, 3-vy, 4-angle, 5-angvel, 6-left_contact, 7-right_contact
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Shaping: encourage moving closer to the target (origin)
    prev_dist = (x_prev**2 + y_prev**2) ** 0.5
    dist = (x**2 + y**2) ** 0.5
    shaping_reward = 0.05 * (prev_dist - dist)

    # Reward each touching foot to promote ground contact
    contact_reward = (left_contact + right_contact) * 2.0

    # Large sparse bonus for a successful soft landing
    contact_ok = float(left_contact > 0.5 and right_contact > 0.5)
    speed_ok = float(abs(vx) < 0.2 and abs(vy) < 0.2 and abs(angvel) < 0.1)
    angle_ok = float(abs(body_angle) < 0.1)
    landing_bonus = 200.0 * contact_ok * speed_ok * angle_ok

    # Small per‑step penalty to discourage lingering
    time_penalty = -0.01

    total = shaping_reward + contact_reward + landing_bonus + time_penalty
    components = {
        "shaping_reward": shaping_reward,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "time_penalty": time_penalty
    }
    return total, components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-91.705574, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-118.270844, -55.650460]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| time_penalty | -10.000000 | -99.6% | 99.6% | 100.0% |
| shaping_reward | 0.019371 | 0.2% | 0.4% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 4. Environment facts
## 1. 任务目标
主目标：控制一个从画面顶部中央附近出发的飞行器，安全、稳定地降落在画面中央的目标平台上。要求着陆时速度接近于零、姿态接近竖直，且所有支脚平稳接触平台。

次要目标：在确保主目标达成的前提下，尽量缩短飞行时间，并尽量减少主引擎和姿态引擎的使用（即节省燃料）。

不可混淆的目标：不应将“快速到达”或“节省燃料”凌驾于“安全着陆”之上；也不能将“悬停”或“保持在目标上方”当作成功条件。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测）
- obs[0]: x_position (相对目标平台的水平坐标), 可直接用于距离/接近奖励，reward_usable: true
- obs[1]: y_position (相对目标平台高度的垂直坐标), 同上，reward_usable: true
- obs[2]: x_velocity (水平线速度), 可用于着陆软度控制，reward_usable: true
- obs[3]: y_velocity (垂直线速度), 同上，reward_usable: true
- obs[4]: body_angle (机体朝向角), 可用于姿态奖励，reward_usable: true
- obs[5]: angular_velocity (角速度), 可用于姿态稳定性惩罚，reward_usable: true
- obs[6]: left_support_contact (左侧支脚接触标志，1.0 表示接触), 可用于着陆状态判断，reward_usable: true
- obs[7]: right_support_contact (右侧支脚接触标志，1.0 表示接触), 同上，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine (无推力，仅受重力/物理影响)
- action 1: left_orientation_engine (点燃左侧姿态推进器，产生旋转力矩)
- action 2: main_engine (点燃主推进器，提供向上推力并可能产生力矩)
- action 3: right_orientation_engine (点燃右侧姿态推进器，产生反方向旋转力矩)

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: body_not_awake_or_settled 如果发生在飞行器已接触地面且速度/角速度极低时，极可能意味着成功着陆；但如果发生在半空中或刚碰撞后，则可能是早期终止。
- failure‑like termination: crash_or_body_contact（与地面或障碍的异常碰撞）、horizontal_position_outside_viewport（水平飞出边界）明确为失败。
- ambiguous termination: body_not_awake_or_settled 本身不区分成功/失败，需要结合观察判断。
- truncation: 代码中未出现 episode length 截断，但实际部署时可能通过外部 wrapper 实现，当前源中未见。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {} 或未返回任何键）
- forbidden_or_uncertain_info_fields: 任何未在以上列出的字段均不可用

## 7. 可用于奖励函数的信号
- position: x_position, y_position（可直接计算到目标的距离、高度差）
- velocity: x_velocity, y_velocity（可衡量接近速度、着陆软度）
- orientation: body_angle（偏离竖直的角度），angular_velocity（旋转速度）
- contact: left_support_contact, right_support_contact（着陆脚是否触地，可判断着陆状态）
- action/engine: action 索引可映射到是否使用主引擎、姿态引擎，用于推力/燃料惩罚
- other: 可通过 (obs, next_obs) 的组合构造微分信号，如速度变化、角速度变化等