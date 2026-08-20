# ⚠️ 上一版代码验证失败
错误信息：Reward v6 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs\env_001\lander_v2_20260704\seed_3\iter_06\generation\validations\reward_v6.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="get_reward_transformation">
<｜｜DSML｜｜parameter name="query" string="true">persistent_to_transition_event continuous state reward to one-time event reward on state entry</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>

# 上一轮奖励函数代码（该轮得分: -236.148516）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Potential‑based shaping hyper‑parameters
    w_vel = 0.1            # weight for absolute linear velocity sum
    w_angle = 0.5          # weight for absolute body angle
    shaping_scale = 50.0   # scales the shaping signal
    gamma = 0.99           # discount factor for shaping

    # Landing continuous reward hyper‑parameters
    alpha = 5.0            # x‑position penalty (0 = target)
    beta = 5.0             # y‑position penalty
    delta = 1.0            # speed penalty exponent coefficient
    eta = 10.0             # angle penalty exponent coefficient

    # ---------- potential functions ----------
    dist_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    vel_curr = abs(obs[2]) + abs(obs[3])
    vel_next = abs(next_obs[2]) + abs(next_obs[3])

    angle_curr = abs(obs[4])
    angle_next = abs(next_obs[4])

    phi_curr = -(dist_curr + w_vel * vel_curr + w_angle * angle_curr)
    phi_next = -(dist_next + w_vel * vel_next + w_angle * angle_next)

    shaping_reward = shaping_scale * (gamma * phi_next - phi_curr)

    # ---------- landing continuous reward ----------
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    landing_reward = 0.0

    if left_contact > 0.5 and right_contact > 0.5:
        x = next_obs[0]
        y = next_obs[1]
        vx = next_obs[2]
        vy = next_obs[3]
        angle = next_obs[4]

        score = (2.718281828 ** (-alpha * x ** 2)) * \
                (2.718281828 ** (-beta * y ** 2)) * \
                (2.718281828 ** (-delta * (vx ** 2 + vy ** 2))) * \
                (2.718281828 ** (-eta * angle ** 2))
        landing_reward = score

    total_reward = shaping_reward + landing_reward

    components = {
        'shaping_reward': shaping_reward,
        'landing_reward': landing_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs indices:
    # 0: x_position, 1: y_position, 2: x_velocity, 3: y_velocity,
    # 4: body_angle, 5: angular_velocity, 6: left_support_contact, 7: right_support_contact
    # goal is at (0,0).

    # 1. Main learning signal: progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraint: light penalty on large speeds, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    w_speed = 0.001
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = -w_speed * speed - w_angle * angle - w_angvel * ang_vel

    # 3. Soft landing proxy: boost when almost on target, slow, upright, and both feet contact.
    dist = d_next
    contact_both = next_obs[6] * next_obs[7]
    k_dist = 2.0
    k_speed = 1.0
    k_angle = 5.0
    w_proxy = 5.0
    soft_landing_proxy = (w_proxy * contact_both *
                          (2.718281828 ** (-k_dist * dist)) *
                          (2.718281828 ** (-k_speed * speed)) *
                          (2.718281828 ** (-k_angle * angle)))

    total_reward = approach_reward + stability_penalty + soft_landing_proxy

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-236.148516, len=835.500000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-297.616417, -144.606198]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| shaping_reward | 894.241736 | 100.0% | 100.0% | 100.0% |
| landing_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从起点（视口顶部中心附近，有随机初始受力）移动并平稳降落到画面中央的目标平台。  
要求尽可能**快**地到达目标，同时**尽量少使用引擎推力**，并在接触时保持**姿态稳定**和**安全接触**。  
核心目标：到达目标平台并稳定停驻。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64（默认连续值）
- obs[0]：x_position —— 相对于目标平台的水平坐标  
- obs[1]：y_position —— 相对于平台高度的垂直坐标  
- obs[2]：x_velocity —— 水平线速度  
- obs[3]：y_velocity —— 垂直线速度  
- obs[4]：body_angle —— 机体倾角  
- obs[5]：angular_velocity —— 角速度  
- obs[6]：left_support_contact —— 左脚接触标志（1.0/0.0）  
- obs[7]：right_support_contact —— 右脚接触标志（1.0/0.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0：no_engine —— 无推力
- action 1：left_orientation_engine —— 启动左侧姿态引擎（产生角速度/微小推力）
- action 2：main_engine —— 启动主引擎（产生主要上升/减速推力）
- action 3：right_orientation_engine —— 启动右侧姿态引擎（与左边方向相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` —— 机体停止运动并稳定（即成功着陆并停驻）
- failure-like termination: 
  - `crash_or_body_contact` —— 任何非法的碰撞或身体接触（如撞到非平台区域、过猛撞击）
  - `horizontal_position_outside_viewport` —— 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 未出现在 step 中（`truncated` 返回 `False`），说明无时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（`info` 为空字典 `{}`）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用

## 7. 可用于奖励函数的信号
- position: next_obs[0] (水平位置误差), next_obs[1] (垂直位置/接近平台高度)  
- velocity: next_obs[2] (水平速度), next_obs[3] (垂直速度) —— 可用于惩罚硬着陆  
- orientation: next_obs[4] (倾角) —— 可用于要求平稳姿态  
- contact: next_obs[6], next_obs[7] (左右接触标志) —— 可奖励双脚稳定着地  
- action/engine: action id —— 可惩罚使用引擎推力以促进节能