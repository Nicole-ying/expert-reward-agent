# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Unpack previous obs
    x_prev = obs[0]
    y_prev = obs[1]

    # Distances to target (origin)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_next = (x ** 2 + y ** 2) ** 0.5

    # 1. Progress delta – main learning signal (only reward forward progress)
    delta = dist_prev - dist_next
    progress_reward = max(0.0, delta) * 1.0

    # 2. Proximity attraction – smooth attraction to the target area
    sigma = 0.5
    proximity_reward = (2.718281828 ** (-dist_next / sigma)) * 0.5

    # 3. Stability penalty – gentle suppression of high speed, tilt, and spin
    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = -(
        w_vel * (abs(vx) + abs(vy)) +
        w_angle * abs(angle) +
        w_angvel * abs(angvel)
    )

    # 4. Contact bonus – only rewarded when already near the landing pad
    contact_bonus = 0.0
    if dist_next < 0.3:
        contact_bonus = 0.2 * (left_contact + right_contact)

    total_reward = progress_reward + proximity_reward + stability_penalty + contact_bonus

    components = {
        'progress_reward': progress_reward,
        'proximity_reward': proximity_reward,
        'stability_penalty': stability_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components
```

# reward_v1 设计说明

## 组件概览
本版奖励由四个组件构成，均只依赖环境声明的观测信号，不使用 `original_reward`、`info` 或任何未声明的字段。

- **progress_reward (主学习信号)**：每一步距离目标更近时获得正向奖励（`max(0, Δd)`）。只奖励前进，不惩罚后退，避免过度消极。这是驱动飞行器向目标移动的核心梯度。
- **proximity_reward (持续吸引力)**：使用指数衰减形式 `exp(-dist / σ)` 提供平滑的靠近奖励。在目标区域附近给出持续的正值，帮助 agent 保持位置、为最终的稳定悬停提供基础梯度。
- **stability_penalty (轻量约束)**：惩罚线速度、机体角度和角速度的绝对值之和。权重极小，作为“方向盘”而非“刹车”，主要防止高速撞击、大幅翻滚等极端行为，而不扼杀正常机动。
- **contact_bonus (任务完成近似信号)**：仅当飞行器距离目标平台小于 0.3 时，按左右腿接触状态给予小奖励。连续且位于限定区域内，避免了单独的接触奖励被利用（如在远处故意触地）。

## 角色划分
- **主学习信号**：`progress_reward`
- **稳定约束**：`stability_penalty`（0~1 个，已选用）
- **任务完成近似信号**：`contact_bonus`（0~1 个，已选用）
- 效率/动作代价未加入，留待后续迭代。

## 为什么没有 terminal_success / terminal_failure？
环境卡明确 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，且 `info` 为空，因此无法使用基于显式标志的奖励。本版通过 `proximity` + `contact_bonus` 提供密集的间接完成引导。

## 与之前失败尝试的区别
此前尝试普遍包含 `distance_anchor`（负距离惩罚）与 `progress_reward` 共存，容易造成信号冲突和大量负奖励累积，且接触信号使用门控/乘积形式导致激活率极低。本设计：
- 放弃 `distance_anchor`，用正值的 `proximity_reward` 提供吸引力，整体奖励为正；
- 接触信号简化为距离门控内的直接加性奖励，没有乘积塌缩风险；
- 稳定性惩罚保持轻量，不因过度压制而让 agent 不敢行动。

## 后续迭代方向
- 若 agent 能在目标附近稳定但双腿接触率仍然低，可适当增加 `contact_bonus` 权重或调紧距离门控阈值。
- 如果出现目标附近震荡而不完成着陆，可考虑加入速度衰减项（如对低速度给予额外奖励）或动态课程。
- 效率和能耗惩罚（`energy_penalty`、`time_penalty`）在任务完成稳定之后加入。

## 应观察的 failure mode
- `high progress without landing`：agent 反复飞越目标区域赚取 progress 奖励却不着陆 → 检查 progress 权重是否过大，或需要加强 proximity 与稳定性。
- `early crash`：飞行器快速失控，progress 很少而 stability penalty 主导 → 考虑降低 stability 权重或增加初始 survival 引导（如轻度高度奖励）。
- `contact exploiting`：agent 学会在目标区域附近反复轻触地面却不完成最终稳定 → 可能需要更严格的着陆条件组合（如同时要求低速）。