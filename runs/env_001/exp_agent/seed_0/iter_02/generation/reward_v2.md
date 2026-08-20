## 诊断

**症状链：**
1. `stability_penalty` 的 `ratio_to_progress = -8.8`，远超 0.5 警戒线 → 惩罚信号强度是主学习信号的 8.8 倍
2. 每步 `total_reward ≈ -0.057`（净负）→ agent 学到"不动比动好"
3. `episode_length = 74`，100% 提前终止 → agent 快速 crash
4. `soft_landing_proxy` nonzero_rate = 0.46% → 几乎不触发，无梯度引导

**根因：** 稳定性惩罚对所有位置一视同仁，agent 在远处靠近目标时就被速度/姿态惩罚压垮，无法探索。

**修改方案（层次 2 — 改数学形态）：**
- **距离门控 stability_penalty**：乘 `max(0, 1 - dist/D)`，远处不惩罚，只有靠近目标时才要求稳定
- **soft_landing_proxy 二值→连续乘积**：5 个 bounded 因子相乘，提供密集梯度（修复 0.46% 触发率）
- **移除 components 中的 `"total_reward"`**（这是 validation 失败的直接原因：components 只能包含公式中直接出现的变量，不能包含结果本身）

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前和下一时刻的位置
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    # 速度、姿态、接触信息（使用 next_obs 反映动作后果）
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 主学习信号：进度增量奖励 ----------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5
    progress_delta = dist - next_dist
    # 裁剪防止目标附近震荡主导信号
    progress_delta = max(-0.5, min(0.5, progress_delta))

    # ---------- 距离门控因子 ----------
    # agent 距离目标 > D_threshold 时，gate≈0，不需要稳定
    # agent 靠近目标时 gate→1，稳定性要求逐渐生效
    D_threshold = 4.0
    proximity_gate = max(0.0, 1.0 - next_dist / D_threshold)

    # ---------- 稳定/安全惩罚（距离门控版） ----------
    # 仅在靠近目标时生效：远处自由移动，近处精细控制
    stability_penalty = -proximity_gate * (
        0.05 * abs(vx) +
        0.05 * abs(vy) +
        0.10 * abs(angle) +
        0.05 * abs(ang_vel)
    )

    # ---------- 软着陆 proxy（连续乘积版，取代二值条件） ----------
    # 每个因子用 bounded max(0, 1 - x/threshold) 提供连续梯度
    prox_factor   = max(0.0, 1.0 - next_dist / 0.5)               # 距离 < 0.5 有信号
    vel_factor    = max(0.0, 1.0 - (abs(vx) + abs(vy)) / 0.4)     # 总速度 < 0.4
    angle_factor  = max(0.0, 1.0 - abs(angle) / 0.2)              # 角度 < 0.2
    ang_vel_factor = max(0.0, 1.0 - abs(ang_vel) / 0.2)           # 角速度 < 0.2
    contact_factor = min(left_contact, right_contact)              # 双脚均接触

    soft_landing_proxy = (
        prox_factor * vel_factor * angle_factor * ang_vel_factor * contact_factor
    )

    # ---------- 总奖励 ----------
    w_progress = 5.0
    w_stab     = 0.5    # 大幅降低（原 1.0）+ 距离门控，确保不压制 progress
    w_soft     = 2.0    # 连续乘积整体偏弱，需略大权重补偿

    total_reward = (
        w_progress * progress_delta +
        w_stab * stability_penalty +
        w_soft * soft_landing_proxy
    )

    # 注意：components 只放公式中直接出现的变量，不放 total_reward
    components = {
        "progress_delta_reward": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
    }

    return float(total_reward), components
```