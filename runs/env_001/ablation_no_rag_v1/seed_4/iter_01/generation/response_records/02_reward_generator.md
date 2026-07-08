# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 下一状态的关键信号
    x = next_obs[0]    # 相对目标平台的水平坐标
    y = next_obs[1]    # 相对于平台高度的垂直坐标
    vx = next_obs[2]   # 水平线速度
    vy = next_obs[3]   # 垂直线速度
    angle = next_obs[4] # 机身朝向角度
    omega = next_obs[5] # 角速度

    # 权重设置（可在后续迭代中调整）
    w_x = 0.2
    w_y = 0.2
    w_vx = 0.05
    w_vy = 0.05
    w_angle = 0.05
    w_omega = 0.05

    # 主学习信号：靠近目标（位置绝对值越小，惩罚越小）
    target_proximity = -w_x * abs(x) - w_y * abs(y)

    # 稳定/安全约束：抑制高速运动与机身摆动
    velocity_penalty = -w_vx * (vx ** 2) - w_vy * (vy ** 2)
    orientation_penalty = -w_angle * (angle ** 2)
    angvel_penalty = -w_omega * (omega ** 2)

    total_reward = target_proximity + velocity_penalty + orientation_penalty + angvel_penalty

    components = {
        "target_proximity": target_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "angvel_penalty": angvel_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **主学习信号**：`target_proximity`，基于下一状态的位置绝对值惩罚，驱动飞行器向目标中心（水平0，垂直0）靠近。每一步都提供稠密的梯度，与任务目标直接对齐。
- **稳定约束**：`velocity_penalty` 对水平与垂直线速度平方惩罚，防止过快接近平台或水平漂移；`orientation_penalty` 抑制机身倾斜；`angvel_penalty` 抑制旋转晃动。三个约束共同推动飞行器在接近目标时减速并保持稳定姿态。
- **未使用 terminal_success_reward / terminal_failure_penalty**：环境没有提供显式的成功/失败标志，`info` 固定为空，因此无法安全引入终止型奖励。
- **未使用接触标志**：虽然 `obs[6]` 和 `obs[7]` 提供左右支撑接触信号，但在 v1 暂不作为奖励成分；接触后的稳定由速度与姿态惩罚自然实现，避免因单独奖励接触而引发的轻触奖励黑客。
- **留待后续迭代**：可考虑加入极轻的动作代价以鼓励节能，或利用接触+低速度+小姿态角组合构造 soft landing proxy，进一步提升稳定着陆质量。
- **训练后可观察的 failure mode**：① 飞行器在目标正上方悬停而不着落（若垂直速度惩罚过强可能抑制下降）；② 水平方向反复振荡，通过左右摆动持续获得进度奖励（需监测速度惩罚是否足够抑制）；③ 接触平台后机身未放平或仍有角速度，长期难以达到完全稳定。根据实际训练曲线调整各惩罚项的相对权重。
