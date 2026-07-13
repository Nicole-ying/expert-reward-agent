# Response Record

# 分析
- **行为证据**：上一轮得分 –205，terminated 15/20，说明大部分轨迹以坠毁或越界终止。组件中 `height_near` (494) 和 `alignment` (231) 提供了很大的正奖励，但 `contact_reward` 的 active_rate 为 0%，意味着双腿从未同时接触。这表明 agent 学会了下降到低空（约 y≈1.3）并保持对准，却无法完成最终接触着陆，常因超出边界或碰撞而终止。过强的 `height_near` 让 agent 滞留在 2 米高度附近“刷分”，阻碍了进一步下降。
- **最值得干预的组件**：`height_near` 是主要正奖励来源，但其形态（`max(0, 2-y_new)`）对下降激励不足，反而在 1~2 米高度产生稳定奖励吸引 agent 停留。应将其替换为更陡峭、聚焦于极低高度的奖励（如反比高度），同时大幅强化接触导向信号与低空速度/姿态约束，将 agent 推出盘旋区。
- **历史修改回顾**：iter9 在 best (iter3) 基础上添加了 `down_penalty`、`descend_vel`、`still_bonus` 等，但 `height_near` 和 `alignment` 结构未变。这些增补因触发极少或无贡献而无效，且未解决核心盘旋问题。因此本轮从 best 骨架出发，用一种新的高度形态替代 `height_near`，并取消无用的添加项，增大关键惩罚，彻底改变奖励地貌。

# 修改理由
把 `height_near` 从线性吸引改为反比激励，让奖励随高度降低急剧上升，驱散盘旋行为；同时引入 `any_contact` 降低接触门槛，并用低空垂直速度惩罚强制软着陆，移除未激活的 `down_penalty`、`descend_vel`、`still_bonus`，使奖励更聚焦于最终触地阶段。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_new = next_obs[0]
    y_new = next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 高度反比奖励（替代 height_near，急剧引导最终下降）----------
    height_reward = 8.0 / (1.0 + y_new ** 2)

    # ---------- 低空强化对准（高度越低，对准越重要）----------
    alignment_gate = 1.0 / (1.0 + max(0.0, y_new))
    alignment = 2.5 * alignment_gate / (1.0 + 12.0 * (x_new ** 2))

    # ---------- 下降进度奖励 ----------
    delta_y = obs[1] - y_new
    progress_y = 0.5 * max(0.0, delta_y)

    # ---------- 接触奖励（单腿即奖，双腿加磅）----------
    any_contact = max(left_contact, right_contact)
    both_contact = left_contact * right_contact
    contact_reward = 2.0 * any_contact + 12.0 * both_contact

    # ---------- 低空垂直速度惩罚（软着陆约束）----------
    low_alt = max(0.0, 1.5 - y_new)
    soft_landing_penalty = -8.0 * low_alt * (max(0.0, -vy_new) ** 2)

    # ---------- 水平速度惩罚 ----------
    lat_penalty = -8.0 * (max(0.0, abs(vx_new) - 0.25) ** 2)

    # ---------- 姿态角惩罚 ----------
    att_penalty = -6.0 * (abs(angle_new) ** 2)

    # ---------- 角速度惩罚 ----------
    angvel_penalty = -3.0 * (abs(angvel_new) ** 2)

    total_reward = (
        height_reward +
        alignment +
        progress_y +
        contact_reward +
        soft_landing_penalty +
        lat_penalty +
        att_penalty +
        angvel_penalty
    )

    components = {
        'height_reward': height_reward,
        'alignment': alignment,
        'progress_y': progress_y,
        'contact_reward': contact_reward,
        'soft_landing_penalty': soft_landing_penalty,
        'lat_penalty': lat_penalty,
        'att_penalty': att_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components
```
