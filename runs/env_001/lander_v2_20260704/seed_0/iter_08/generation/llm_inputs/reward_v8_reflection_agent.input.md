# 上一轮奖励函数代码（该轮得分: 167.992192）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取 next_obs 的各个分量
    x_pos = next_obs[0]          # 水平位置
    y_pos = next_obs[1]          # 垂直位置
    x_vel = next_obs[2]          # 水平速度
    y_vel = next_obs[3]          # 垂直速度
    body_angle = next_obs[4]     # 身体角度
    left_contact = next_obs[6]   # 左支撑腿接触
    right_contact = next_obs[7]  # 右支撑腿接触

    # 主学习信号：非线性接近奖励，鼓励向目标靠近
    dist_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    proximity_reward = 1.0 / (1.0 + dist_to_target)

    # 稳定约束：抑制高速，防止剧烈运动
    velocity_penalty = -0.1 * (abs(x_vel) + abs(y_vel))

    # 姿态约束：抑制大幅倾斜，保证着陆姿态
    angle_penalty = -0.05 * abs(body_angle)

    # 任务完成近似信号：鼓励双腿接触着陆垫
    contact_reward = 0.3 * (left_contact + right_contact)

    total_reward = proximity_reward + velocity_penalty + angle_penalty + contact_reward

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'angle_penalty': angle_penalty,
        'contact_reward': contact_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=167.992192, len=647.400000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-41.151395, 248.794180]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 491.129187 | 88.1% | 88.1% | 100.0% |
| contact_reward | 51.210000 | 9.2% | 9.2% | 14.9% |
| velocity_penalty | -13.459726 | -2.4% | 2.4% | 99.9% |
| angle_penalty | -1.521084 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器，使其从起点（靠近视野顶部中央）尽快且尽可能省燃料地降落到中心目标平台上，并保持稳定姿态与安全接触。重点要求：快速到达、稳定着陆、合理使用引擎（主引擎和姿态引擎）。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x_position —— 相对目标平台中心的水平坐标
- obs[1]: y_position —— 相对目标平台高度的垂直坐标
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 本体倾角（方向角）
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左支撑腿触地标志（0.0/1.0）
- obs[7]: right_support_contact —— 右支撑腿触地标志（0.0/1.0）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine —— 不点火，无推力
- action 1: left_orientation_engine —— 点燃左侧姿态调整引擎
- action 2: main_engine —— 点燃主引擎（提供向上推力）
- action 3: right_orientation_engine —— 点燃右侧姿态调整引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 当 left_support_contact == 1.0 且 right_support_contact == 1.0，同时 body_not_awake_or_settled 被触发，可认为成功着陆（推测）。
- failure-like termination: crash_or_body_contact 或 horizontal_position_outside_viewport 触发，表示撞击或飞出视野，显然是失败。
- ambiguous termination: body_not_awake_or_settled 但接触点并非双足同时触地（例如在空中稳定或单侧触地），该终止状态较模糊，不应直接视作成功或失败。
- truncation: 未提及时间截断，可能由环境外部控制，此处无可靠信息。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，没有 success 字段）
- explicit_failure_flag_available: false （不存在 failure 或 termination_reason）
- allowed_info_fields: {} （info 字典不含任何可用字段）
- forbidden_or_uncertain_info_fields: 所有未在观测中出现的终止原因信息均不可用，只能从 obs 和终止事实反向推断。

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) —— 用于评估与目标点的距离。
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) —— 可用于惩罚快速移动，尤其在接近目标时。
- orientation: obs[4] (body_angle) —— 理想值应为 0（垂直向上），可用于惩罚倾斜。
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) —— 指示着陆腿是否触地，可用于推动双腿稳定着陆。
- action/engine: 动作编号 1, 2, 3 分别代表不同引擎点火，可设计燃料惩罚或使用频率惩罚。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_proxy + stability_penalty | -518.69 | -518.69 | 0.00 | 62.35 | distance_reward=-0.981 soft_landing_proxy=0.000 stability_penalty=-0.035 | new_best |
| 2 | distance_reward + soft_landing_proxy + stability_penalty | -85.75 | -85.75 | 0.00 | 88.95 | distance_reward=0.071 soft_landing_proxy=0.005 stability_penalty=-0.014 | new_best |
| 3 | distance_reward + landing_quality + stability_penalty | -17.84 | -17.84 | 0.00 | 1000.00 | distance_reward=0.016 landing_quality=1.007 stability_penalty=-0.004 | new_best |
| 4 | distance_reward + landing_quality + stability_penalty | -110.72 | -17.84 | -92.88 | 68.50 | distance_reward=0.080 landing_quality=0.092 stability_penalty=-0.015 | no_meaningful_improvement |
| 5 | distance_reward + landing_quality + stability_penalty | -23.32 | -17.84 | -5.47 | 1000.00 | distance_reward=0.017 landing_quality=2.090 stability_penalty=-0.004 | no_meaningful_improvement |
| 6 | distance_reward + landing_quality + stability_penalty | -73.82 | -17.84 | -55.98 | 87.30 | distance_reward=0.073 landing_quality=0.025 stability_penalty=-0.014 | unsolved_stagnation_fresh_restart |
| 7 | angle_penalty + contact_reward + proximity_reward + velocity_penalty | 167.99 | 167.99 | 0.00 | 647.40 | angle_penalty=-0.006 contact_reward=0.267 proximity_reward=0.723 velocity_penalty=-0.024 | new_best |
