# Response Record

## 组件分析

### 1. 整体信号结构的问题
最终评估分数 `score=-80.35`，远低于目标 200，且 40%/80% 检查点分数均为 0.00，表明智能体在训练过程中几乎没有学到有效策略。当前奖励函数由大量持续性惩罚（`distance_penalty`, `vel_penalty`, `angle_penalty`, `angvel_penalty`）主导，正向奖励（`dist_reward`, `contact_reward`, `success_bonus`）贡献极小。智能体在这样的信号下陷入“少动即少罚”的局部最优，无法探索接触与着陆行为。**必须重写整个奖励，将主信号切换为正向的“向目标移动 + 接近目标”框架。**

### 2. 现有组件逐一诊断

| 组件 | 诊断 |
|---|---|
| `distance_penalty` | `magnitude_share=37.2%`，每一步持续扣分，是总负分的主要来源。虽然它理论上会驱动靠近中心以减小惩罚，但智能体由于其他惩罚（速度/角度）不敢做大幅移动，导致停滞。**结论：丢弃该组件，改用正向的“接近奖励”。** |
| `dist_reward` | `magnitude_share=0.7%`，系数 0.5 太小，且因智能体移动方向随机（或无移动），正负抵消后近乎 0。**结论：大幅提高系数，作为新主信号“进步奖励”。** |
| `vel_penalty` | 系数 0.2，叠加其他惩罚后进一步抑制探索。**结论：保留但降低系数**，让智能体敢于加速向目标移动。 |
| `angle_penalty` | 对保持垂直有作用，但当前幅度合理，**暂不修改**。 |
| `angvel_penalty` | 已很小，**保留**。 |
| `contact_reward` | `magnitude_share=52%` 但 `active_rate=1.8%`，说明一旦接触，每步给予大量奖励，但触发概率极低，对学习指引微弱。**结论：降低每步值，避免落地后奖励淹没其他信号；同时提高稀疏的 success_bonus 来凸显着陆成功。** |
| `success_bonus` | 稀疏，只有双触地且距离<0.5 才给 2.0，几乎不出现。**结论：大幅提高值（例如 10.0），激励智能体努力完成稳定着陆。** |

### 3. 修改方案

- 删除 `distance_penalty`，加入 `proximity = 0.5 / (1.0 + curr_dist)` 作为正向接近奖励，不产生负分。
- 将 `dist_reward` 改为 `progress = 5.0 * (prev_dist - curr_dist)`，作为主驱动信号。
- 降低 `vel_penalty` 系数到 0.1。
- 降低 `contact_reward` 系数，并将 `success_bonus` 提升至 10.0。
- 其他组件暂时不动，构成一个新的、以正向进步为核心的奖励框架。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    px, py = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_px, prev_py = obs[0], obs[1]

    curr_dist = (px**2 + py**2)**0.5
    prev_dist = (prev_px**2 + prev_py**2)**0.5

    # 主正向驱动：向中心移动的进步奖励
    progress = 5.0 * (prev_dist - curr_dist)

    # 接近目标的正奖励，替代原来的距离惩罚
    proximity = 0.5 / (1.0 + curr_dist)

    # 速度惩罚（降低系数，允许探索加速）
    vel_penalty = -0.1 * (vx**2 + vy**2)

    # 保持直立
    angle_penalty = -0.5 * (angle**2)

    # 抑制过度旋转
    angvel_penalty = -0.02 * (ang_vel**2)

    # 接触奖励（降低幅度，避免落地后数值占主导）
    contact_reward = 0.5 * left_contact + 0.5 * right_contact + 1.0 * (left_contact * right_contact)

    # 稀疏成功奖励，大幅提高以凸显着陆目标
    success_bonus = 10.0 if (left_contact and right_contact and curr_dist < 0.5) else 0.0

    total = (progress + proximity + vel_penalty + angle_penalty +
             angvel_penalty + contact_reward + success_bonus)

    components = {
        "progress": progress,
        "proximity": proximity,
        "vel_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus
    }
    return float(total), components
```
