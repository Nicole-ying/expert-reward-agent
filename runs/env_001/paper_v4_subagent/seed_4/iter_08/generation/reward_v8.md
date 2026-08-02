# 设计理由
将 `landing_bonus` 从“全局状态品质奖励”改为“接触门控品质奖励”：仅当支撑脚接触时，奖励才激活（乘以 `contact_gate`）。无接触时该组件为零，消除 agent 在目标附近悬停刷分的动机，迫使它依赖 `progress` 驱动快速接近并在接触后获得奖励。同时保留 `angle_penalty` 和 `angvel_penalty`，不做改动。

数学形式不变，只是在原有乘积中乘入 `float(nl_contact or nr_contact)`，使其在接触前为 0。系数和阈值保持一致，不引入新参数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, l_contact, r_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nl_contact, nr_contact = next_obs

    # 1. Main progress signal: distance reduction to target pad
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_obs - dist_next

    # 2. Stability constraints (quadratic penalties on next state)
    angle_penalty = -0.1 * (nangle ** 2)
    angvel_penalty = -0.05 * (nangvel ** 2)

    # 3. Contact-gated soft landing attractor
    #    Only active when support legs are in contact (gate), otherwise zero.
    proximity = 2.718281828 ** (-dist_next / 0.8)
    speed_factor = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 1.0)
    angle_factor = max(0.0, 1.0 - abs(nangle) / 0.5)
    contact_gate = float(nl_contact or nr_contact)
    landing_attractor = proximity * speed_factor * angle_factor * contact_gate

    w_progress = 10.0
    w_attractor = 1.0

    total = (w_progress * progress +
             angle_penalty + angvel_penalty +
             w_attractor * landing_attractor)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": w_attractor * landing_attractor
    }

    return float(total), components
```

# 诊断摘要
- **audit**: 所有观测维度已使用；终止模式以超时和终止各半，无缺失观测信号；核心问题是 `landing_bonus` 每步持续奖励导致 origin 附近悬停刷分。改为接触门控后，未接触时该组件为零，消除悬停激励。
- **behavior**: agent 被 `landing_bonus` 驱动在目标附近徘徊，偶尔接触；`progress` 仅占 1.2% 份额，未能驱动快速完成任务。
- **signal**: `landing_bonus` 过强（98.6% episode_sum_mean 份额），`progress` 过弱。需取消接触前的吸引力，让 `progress` 重新主导接近阶段。
- **level**: Level 2（结构变换：全局品质奖励 → 局部门控奖励）。
- **hypothesis**: 接触门控使接触前仅有 `progress` 提供梯度，驱使 agent 快速抵达目标区域；接触后品质奖励激活，鼓励稳定保持并完成软着陆。这将缩短 episode 长度、提高成功终止比例，从而提升总得分。
- **risk**: 若 agent 在接触前未能减速/调平，可能无法触发接触，导致完全失去 `landing_bonus`；但当前 agent 已具备接近能力，风险可控。`progress` 将在接近后变小，接触后品质奖励成为主要回报，整体仍可提供充分完成信号。