# Response Record

# 设计理由

**第0步信号覆盖审计**：所有8维观测已被使用，信号齐全。`failure_penalty` active_rate=0%（僵尸组件），`landing_proxy` active_rate=100%且占99.7% signed share，`progress` 为负值且极弱。19/20 episode truncated（超时），说明 agent 并未完成着陆目标，但无失败终止——它学会了利用 `landing_proxy` 的每步奖励在安全区悬停徘徊，始终不触发 `body_not_awake_or_settled`（着陆终止），从而累积大量奖励（~865/eps）。行为与最终环境得分 -167 严重背离。

**干预对象**：`landing_proxy` — 该组件是 exploit 根源：它由距离、速度、姿态、接触四因子平均而成，每当 agent 处于垫附近且低速小倾角时就给出高额持续奖励，而无需真正着陆。必须将其从“每步生存奖励”重构为“成功事件奖励 + 弱接近引导”。

**层级选择**：Level 2 结构变换。将 `landing_proxy` 从 `joint_condition_proxy`（多因子平均）改为 **`terminal_event` 成功奖励 + `dense_state_signal` 有界接近奖励**。保留 `progress`、`failure_penalty`、`stability_penalty` 不动。

**数学形式与系数校准**：
- *接近奖励*：`max(0, 1 - dist / CLOSE_THRESHOLD) * PROXIMITY_REWARD_FACTOR`，`CLOSE_THRESHOLD=0.2`，`PROXIMITY_REWARD_FACTOR=0.5`。最大每步 +0.5，仅当飞行器进入原点0.2范围内才非零。远小于原 exploit 信号的 ~0.87/步，防止悬停价值过高。
- *成功奖励*：当满足严苛着陆条件（|x|<0.1, |y|<0.1, |vx|<0.2, |vy|<0.2, |angle|<0.1，且至少一脚接触）时，给予 +100。远远超过全 episode 可能累积的接近奖励（即使 1000 步全在阈值内也只有 500），引导 agent 尽快完成着陆并终止 episode。
- 其他组件未修改：`progress`（每步 -0.0007）和 `stability_penalty`（每步 -0.001）远小于新主信号，`failure_penalty` 保持原有检测逻辑（等待后续激活）。

**预期效果**：agent 去除“悬停赚钱”的激励后，仅能通过接近原点获得少量奖励，且必须触发成功着陆才能获得巨额回报，从而学习快速、稳定着陆。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_cur, y_cur = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 超参数
    PROGRESS_WEIGHT = 2.0
    FAIL_PENALTY = -10.0
    LANDING_PROXY_WEIGHT = 1.0   # 已重新定义组件，权重内化
    ANGLE_PENALTY = 0.5
    ANG_VEL_PENALTY = 0.1

    X_BOUNDARY = 1.0
    ANGLE_CRASH = 0.8
    GROUND_Y_CLOSE = 0.2
    VEL_CRASH = 1.5
    CLOSE_THRESHOLD = 0.2         # 接近原点奖励的边界
    PROXIMITY_REWARD_FACTOR = 0.5 # 最大每步接近奖励
    SUCCESS_X_THRESH = 0.1
    SUCCESS_Y_THRESH = 0.1
    SUCCESS_VEL_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1
    SUCCESS_REWARD = 100.0

    # 1. 进展信号（保持不变）
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # 2. 失败惩罚（保持不变）
    out_of_bounds = abs(x_next) > X_BOUNDARY
    crash = False
    if (left_contact == 1.0 or right_contact == 1.0):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > ANGLE_CRASH
        high_impact = abs(y_vel_next) > VEL_CRASH
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True
    failure = out_of_bounds or crash
    failure_penalty = FAIL_PENALTY if failure else 0.0

    # 3. 着陆引导（重构后的 landing_proxy）
    dist = (x_next**2 + y_next**2) ** 0.5
    # 弱接近奖励：在垫附近提供渐进引导
    proximity_reward = max(0.0, 1.0 - dist / CLOSE_THRESHOLD) * PROXIMITY_REWARD_FACTOR
    # 成功着陆检测
    landed = (abs(x_next) < SUCCESS_X_THRESH and
              abs(y_next) < SUCCESS_Y_THRESH and
              abs(x_vel_next) < SUCCESS_VEL_THRESH and
              abs(y_vel_next) < SUCCESS_VEL_THRESH and
              abs(body_angle_next) < SUCCESS_ANGLE_THRESH and
              (left_contact == 1.0 or right_contact == 1.0))
    success_reward = SUCCESS_REWARD if landed else 0.0
    landing_proxy_reward = LANDING_PROXY_WEIGHT * (proximity_reward + success_reward)

    # 4. 姿态稳定惩罚（保持不变）
    stability_penalty = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    total_reward = progress + failure_penalty + landing_proxy_reward + stability_penalty

    components = {
        'progress': progress,
        'failure_penalty': failure_penalty,
        'landing_proxy': landing_proxy_reward,  # 名称不变，便于追踪
        'stability_penalty': stability_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测维度已被使用，信号齐全；`landing_proxy` 每步持续高回报导致 agent 悬停 exploit，是得分与任务目标偏离的根因。
- **behavior**: agent 学会了在垫附近保持低速小倾角来持续获取 `landing_proxy` 奖励，几乎全 episode truncated 而不着陆。
- **signal**: 缺失任务完成（成功着陆）的强正激励，`landing_proxy` 作为常驻奖励强度过高，覆盖了进展和姿态信号。
- **level**: Level 2
- **hypothesis**: 将常驻多因子代理奖励重构为“弱接近引导 + 大额成功事件奖励”后，agent 不再有悬停激励，必须快速完成安全着陆才能获得主要回报，从而学习真正完成任务。
- **risk**: 接近奖励可能引起在阈值边缘小幅振荡，但成功奖励远大于振荡累积，风险可控；着陆检测阈值可能需根据实际可达精度微调。
