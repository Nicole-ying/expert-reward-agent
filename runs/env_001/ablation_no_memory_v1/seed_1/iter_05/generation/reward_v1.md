# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Helper: 计算当前和目标距离
    dx_obs, dy_obs = obs[0], obs[1]
    dist_obs = (dx_obs**2 + dy_obs**2)**0.5

    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_next = (dx_next**2 + dy_next**2)**0.5

    # 1. 主学习信号：每一步更接近目标
    progress_delta = dist_obs - dist_next

    # 2. 稳定/安全约束：抑制高速、大角度、高角速度
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (-0.01 * (abs(vx) + abs(vy))
                         - 0.05 * abs(angle)
                         - 0.01 * abs(ang_vel))

    # 3. 任务完成代理信号：软着陆的多条件组合
    near_target = abs(next_obs[0]) < 0.2 and abs(next_obs[1]) < 0.2
    low_speed = abs(vx) < 0.5 and abs(vy) < 0.5
    stable_angle = abs(angle) < 0.3
    both_contact = next_obs[6] > 0.5 and next_obs[7] > 0.5

    landing_bonus = 10.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 总奖励
    total_reward = progress_delta + stability_penalty + landing_bonus

    # 组件字典（只包含加到 total_reward 的项）
    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

- **progress_delta（主学习信号）**：奖励每一步飞行器到目标（平台中心）距离的减小量。提供每步稠密梯度，直接驱动“靠近目标”这一核心目标。
- **stability_penalty（轻量稳定约束）**：惩罚线速度、机体角度和角速度的绝对值，鼓励飞行器以低速、平稳姿态飞行，为安全着陆创造条件。权重较小，不会压制探索。
- **landing_bonus（任务完成代理信号）**：当且仅当飞行器同时满足“紧贴目标、低速、姿态平稳、双腿着地”四个条件时给予一次性正向奖励。作为无显式成功标志时的任务完成近似，引导策略正确地完成着陆而不仅是接近目标。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境中 **不存在显式的 success/failure 标志**（info 为空，且 terminate 原因未传入）。因此不能也不应该伪造这些信号。我们用 `landing_bonus` 这个多条件组合的 soft proxy 替代成功奖励，用 `stability_penalty` 和距离信号的结合间接避免危险行为。

## 哪些组件留到后续迭代
- **energy_penalty / time_penalty**：v1 阶段先让飞行器学会安全着陆，燃料优化留到后续再引入，避免 agent 因害怕耗能而不敢使用主发动机。
- **gated_reward / dynamic_curriculum**：在完成基础行为后，如果需要更精细的阶段控制或渐进课程，可以在 v2 加入。
- **终端失败惩罚**：只有当 wrapper 明确提供可靠的失败原因（如 crash 类型）时才考虑加入。

## 训练后应观察的 failure mode
- **目标附近震荡（oscillation without landing）**：`progress_delta` 可能让飞行器在平台上方反复调整位置而不降落。应检查 `landing_bonus` 的触发频率和每幕最后几步的奖励分量。
- **高速俯冲 crash**：如果 `stability_penalty` 过弱，agent 可能选择以较高速度冲向目标试图快速获得 `progress_delta` 奖励，导致触地 crash。此时可适当上调稳定性权重或收紧 `landing_bonus` 的速度条件。
- **软着陆条件过宽导致的 contact reward hacking**：如果 agent 在平台附近低速悬停并间歇性触发单腿或双腿接触而获得奖励，却不完成最终稳定停稳（例如反复轻触地面），需在后续版本中收紧 `landing_bonus` 的位置、速度及接触持续要求。