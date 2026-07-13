# Response Record

`evidence`：landing_bonus episode_sum_mean=107（magnitude_share=99%）但 active_rate 仅 1.1%，说明每 episode 约 10 次短接触而非持续着陆；episode 长 975 步、18/20 被截断，terminated 仅 2/20；score=-42.74 而 training reward 中 landing_bonus 贡献巨大——典型 proxy farming：agent 通过反复弹跳刷接触奖励。

`behavior_diagnosis`：agent 已能到达目标区域附近，但无法执行平稳着陆——以较高速度反复触碰着陆垫后弹开，靠连续接触奖励维持高 training reward，实际外部任务因油耗/撞击惩罚得分仍为负。

`signal_completeness`：进度引导（shaped_progress）存在但信号极弱（mean=0.548）；角速度约束存在但可忽略；着陆信号存在但数学形态错误——奖励"处于接触状态"而非"高质量着陆"，缺失触地时的速度节制。

`selected_level`：Level 2 —— `global_to_local_gate` 变换。证据表明 landing_bonus 在无关阶段（高速弹跳）被大量领取，需用速度和姿态门控使其仅在合理着陆条件下生效。

`selected_intervention`：唯一修改 landing_bonus 组件，从 `10.0 * both_contact` 变为 `10.0 * both_contact * velocity_gate * angle_gate`，其中 velocity_gate = `1/(1+2*speed)` 压制高速触地奖励，angle_gate 复用现有姿态门控。

`falsifiable_hypothesis`：速度门控使高速弹跳触地获得的奖励大幅降低（speed=3 时 gate≈0.14），而低速轻柔着陆获得接近全额奖励。这应产生梯度压力迫使 agent 在接近垫面时减速，减少弹跳、提高真正着陆成功率。

`expected_next_round`：landing_bonus 的 episode_sum_mean 应下降（弹跳收益被压制），active_rate 可能暂时下降或持平；若减速行为开始出现，terminated 比例应上升；score 中期可能因探索减速而波动，长期应改善。

`main_risk`：双门控乘积可能使着陆奖励过于稀疏——若 agent 当前从未经历过同时低速+小角度的接触状态，landing_bonus 可能接近零，信用分配困难可能导致 agent 放弃接近目标区域。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 匿名环境观测假设（经典2D着陆器）：
    # obs[0]: x (水平偏移)    obs[1]: y (高度)
    # obs[2]: vx             obs[3]: vy
    # obs[4]: angle          obs[5]: ang_vel
    # obs[6]: left_contact   obs[7]: right_contact

    # 1. 主学习信号：距离改善 × 姿态门控
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next          # 正值表示靠近目标

    angle_abs = abs(next_obs[4])                     # 机体角度（弧度）
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)        # soft health gate
    shaped_progress = progress_raw * gate

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
