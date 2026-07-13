当前agent行为：全部episode超时（truncated=20/20），terminated=0，没有坠毁也没有成功着陆。agent学会了水平对准（alignment贡献98.7%），但在高度上完全停滞在2.0以上（height_near、contact_reward激活率为0），没有下降意愿。奖励结构导致“悬停在高空且水平对准”即可获得大量正奖励，无需冒险下降。因此需要将水平对准奖励与高度绑定（高度越低奖励越高），同时增加下降进度激励，引导agent向下运动。

修改依据：
- 加入高度门控（soft gate）：将alignment乘以`1/(1+y)`，使高处的水平奖励被抑制，低处被放大，提供下降梯度。
- 引入下降进度奖励：`max(0, y_old - y_new)`和`max(0, -vy_new)`，直接鼓励降低高度和向下速度。
- 简化近地奖励：去掉landing_quality调制，改用线性`max(0, 2.0 - y)`，确保agent进入低空时获得正向引导。
- 保留并适当提高接触奖励、保留各项速度/姿态惩罚和静止奖励，维持安全约束。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 高度门控的水平对准（高空抑制，低空增强）----------
    height_gate = 1.0 / (1.0 + y_new)
    alignment = 1.5 * height_gate / (1.0 + 10.0 * (x_new ** 2))

    # ---------- 下降进度奖励（高度降低）----------
    delta_y = y_old - y_new
    progress_y = 0.2 * max(0.0, delta_y)

    # ---------- 下降速度奖励 ----------
    descend_vel = 0.1 * max(0.0, -vy_new)

    # ---------- 近地奖励（无接触要求，引导进入低空）----------
    height_near = 1.0 * max(0.0, 2.0 - y_new)

    # ---------- 接触奖励（双腿同时接触）----------
    both_contact = left_contact * right_contact
    contact_reward = 10.0 * both_contact

    # ---------- 垂直速度惩罚（容忍度略放宽）----------
    down_penalty = -5.0 * (max(0.0, -vy_new - 0.5) ** 2)

    # ---------- 水平速度惩罚 ----------
    vx_limit = 0.2
    lat_penalty = -2.0 * (max(0.0, abs(vx_new) - vx_limit) ** 2)

    # ---------- 姿态角惩罚 ----------
    angle_limit = 0.05
    att_penalty = -2.0 * (max(0.0, abs(angle_new) - angle_limit) ** 2)

    # ---------- 角速度惩罚 ----------
    angvel_limit = 0.05
    angvel_penalty = -2.0 * (max(0.0, abs(angvel_new) - angvel_limit) ** 2)

    # ---------- 趋近静止奖励 ----------
    speed_norm = (vx_new ** 2 + vy_new ** 2) ** 0.5
    still_bonus = 0.5 * max(0.0, 0.1 - speed_norm)

    total_reward = (
        alignment +
        progress_y +
        descend_vel +
        height_near +
        contact_reward +
        down_penalty +
        lat_penalty +
        att_penalty +
        angvel_penalty +
        still_bonus
    )

    components = {
        'alignment': alignment,
        'progress_y': progress_y,
        'descend_vel': descend_vel,
        'height_near': height_near,
        'contact_reward': contact_reward,
        'down_penalty': down_penalty,
        'lat_penalty': lat_penalty,
        'att_penalty': att_penalty,
        'angvel_penalty': angvel_penalty,
        'still_bonus': still_bonus
    }

    return float(total_reward), components
```