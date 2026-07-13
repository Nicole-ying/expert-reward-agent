# Response Record

根据反馈，当前 agent 全部 episode 以 crash/failure 终止（terminated=20/20），episode length 约 536，得分 -139，远低于 best 的 71 分。奖励分解显示 height_near_reward 和 alignment 占比高但 active_rate 仅 69.6%，且 stable_contact_reward 始终为 0，any_contact 触发极低（0.7%），表明 agent 几乎没有经历安全着陆过程。上一轮修改添加了 `descent_factor` 和 `y_progress`，试图用下降速度门控来引导，结果反而导致 agent 急坠撞毁或飘出边界——这是因为 `descent_factor` 要求向下速度达标才给予高度/对准奖励，同时高权重 `y_progress` 鼓励快速下降，而速度安全约束（vel_penalty）过弱，使得 agent 在接近地面时无法稳定减速。

**分析结论**：
- 崩溃原因：下降速度失控和水平速度过大，现有惩罚不足以阻止危险行为，且高度引导（descent_factor）存在断层。
- 核心干预：移除具有误导性的下降速度门控和 `y_progress`，扩大高度引导范围（从 0.8 提升到 1.5），采用平方形态强化近地引力，同时引入较强的 hinged 惩罚（垂直下坠速度和水平速度），迫使 agent 学会减速和软着陆。

**修改方案**：基于 best 时期的基础结构（progress + height_near + alignment + contact + velocity penalties），但做出以下有证据的改动：
1. 将 `height_near` 阈值扩展到 1.5，形态改为 `(max(0, threshold - y))²`，确保从较高位置就有降落激励且越近奖励越大；
2. 取消 `descent_factor` 和 `y_progress`，避免强制下降速度约束；
3. 加入较强的 hinged 下坠惩罚（`-vy` 超过 0.5 时）和水平速度惩罚（`|vx|` 超过 0.2 时）；
4. 提升 `att_penalty` 和 `angvel_penalty` 的权重，使姿态在着陆区保持稳定；
5. 保留 `progress`（距离缩减）作为全局引导，略微提高权重。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 读取状态 ----------
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 全局前进引导（向原点缩短距离）----------
    dist_old = (x_old**2 + y_old**2)**0.5
    dist_new = (x_new**2 + y_new**2)**0.5
    progress = 3.0 * (dist_old - dist_new)

    # ---------- 高度近地奖励（平方形态，宽范围）----------
    height_threshold = 1.5
    height_near_raw = max(0.0, height_threshold - y_new)
    w_height = 5.0
    height_near = w_height * (height_near_raw ** 2)

    # ---------- 水平对准（依赖高度的 gate）----------
    height_gate = max(0.0, 1.0 - y_new / height_threshold)
    alignment_raw = 1.0 / (1.0 + x_new**2)
    w_align = 0.5
    alignment = w_align * alignment_raw * height_gate

    # ---------- 接触奖励 ----------
    any_contact = max(left_contact, right_contact)
    w_contact = 2.0
    contact_reward = w_contact * any_contact

    # ---------- 水平速度惩罚（hinge）----------
    vx_limit = 0.2
    lat_penalty = -2.0 * (max(0.0, abs(vx_new) - vx_limit) ** 2)

    # ---------- 下坠速度惩罚（hinge，防止过快下降）----------
    vy_down_limit = 0.5
    down_penalty = -2.0 * (max(0.0, -vy_new - vy_down_limit) ** 2)

    # ---------- 姿态与角速度惩罚（hinge）----------
    angle_limit = 0.1
    att_penalty = -2.0 * (max(0.0, abs(angle_new) - angle_limit) ** 2)
    angvel_limit = 0.1
    angvel_penalty = -2.0 * (max(0.0, abs(angvel_new) - angvel_limit) ** 2)

    # ---------- 总奖励 ----------
    total_reward = (progress + height_near + alignment + contact_reward +
                    lat_penalty + down_penalty + att_penalty + angvel_penalty)

    components = {
        'progress': progress,
        'height_near': height_near,
        'alignment': alignment,
        'contact_reward': contact_reward,
        'lat_penalty': lat_penalty,
        'down_penalty': down_penalty,
        'att_penalty': att_penalty,
        'angvel_penalty': angvel_penalty
    }
    return float(total_reward), components
```
