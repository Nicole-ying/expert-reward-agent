`evidence`：landing_bonus 活跃率仅1.2%且占signed_share的97.3%，shaped_progress 占1.6%但 episode_sum_mean 仅0.83，说明全过程引导信号极端微弱；40% episode 成功着陆（terminated=8/20），60% 超时失败，得分方差巨大（-182~+222），根因是稠密引导太弱无法可靠驱动接近行为。
`behavior_diagnosis`：agent 在约40%的episode中偶然学会着陆并获得高额bonus，其余60%的episode徘徊至超时，全过程缺乏足够强度的接近引导使行为高度不一致。
`signal_completeness`：任务所需的主引导（距离改善）、稳定约束（角速度惩罚）和完成信号（着陆质量bonus）职责均已存在且可达，但主引导信号尺度严重不足。
`selected_level`：Level 1 — shaped_progress 的职责、符号（距离差分，正值=靠近）和数学形态（势能差×姿态门控）合理，证据指向该组件过弱而非结构错误，且历史未对此做过尺度调整。
`selected_intervention`：将 shaped_progress 的系数从隐式1.0提升至25.0，其他组件不变。
`falsifiable_hypothesis`：提升25倍后，每episode的shaped_progress累计值应从~0.83增至~20，使接近行为获得与landing_bonus（~50）可比的过程反馈，从而将terminated比例从40%显著提升，得分均值转正。
`expected_next_round`：shaped_progress episode_sum_mean ≈ 15–25，terminated > 10/20，score 均值 > 0 且方差缩小，landing_bonus active_rate 可能随更多成功着陆而上升。
`main_risk`：失败episode中远离目标的负progress同样被放大25倍，可能导致极负score；若姿态门控在必要侧倾机动时过度抑制progress，即使放大后仍无法有效引导水平位移。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 匿名环境观测假设（经典2D着陆器）：
    # obs[0]: x (水平偏移)    obs[1]: y (高度)
    # obs[2]: vx             obs[3]: vy
    # obs[4]: angle          obs[5]: ang_vel
    # obs[6]: left_contact   obs[7]: right_contact

    # 1. 主学习信号：距离改善 × 姿态门控（尺度修复：系数 25.0）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next          # 正值表示靠近目标

    angle_abs = abs(next_obs[4])                     # 机体角度（弧度）
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)        # soft health gate
    shaped_progress = 25.0 * progress_raw * gate     # 系数从1.0→25.0

    # 2. 安全约束：角速度轻量惩罚
    ang_vel         = next_obs[5]
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 3. 着陆质量奖励：低速+姿态良好才给显著奖励
    left  = next_obs[6]
    right = next_obs[7]
    both_contact = left * right                      # 仅当两腿同时接触时非零

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    velocity_gate = 1.0 / (1.0 + 2.0 * speed)        # 低速→1，高速→0

    angle_gate = 1.0 / (1.0 + 5.0 * angle_abs)       # 直立→1，倾斜→0

    landing_bonus = 10.0 * both_contact * velocity_gate * angle_gate

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return float(total_reward), components
```