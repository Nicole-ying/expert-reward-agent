# 设计理由
将姿态稳定性惩罚从全时二次型改为基于 hull_angle 的门控惩罚（hinge），以减少对小角度摆动的无差别压制，让 agent 在安全区域内自由探索更高速步态；只在倾角越过安全阈值（0.4 rad，约 23°，约为摔倒边界的 80%）时施加较强惩罚。同时保留对角速度和垂直速度的微小二次惩罚，抑制剧烈旋转和跳跃。这一变换旨在解除对正常行走小幅摇摆的持续惩罚，从而可能释放出额外的速度提升空间，突破 300 分瓶颈。

系数校准：
- forward_reward 保持 1.0 线性速度奖励，每步均值约 0.576。
- energy_penalty 恒有，每步均值约 0.021，负担合理。
- 新稳定性惩罚中，hull_angle hinge 系数设为 1.5，当倾角刚超过阈值时惩罚轻微（如 0.5 rad 时仅 -0.015），在危险区（如 0.7 rad）惩罚约 -0.135，与前 v3 在相同角度下的惩罚量级相当，但对安全区行为无压制。
- angvel 和 vertical 惩罚系数维持 0.01，负担极小。
- 所有惩罚总 per‑step 负担远小于主信号的 0.5 倍，满足设计校准。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ====== 主学习信号：线性前进速度奖励 ======
    forward_vel = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_vel

    # ====== 姿态稳定性惩罚（hinge 门控） ======
    hull_angle = obs[0]
    hull_angvel = obs[1]
    vertical_vel = obs[3]

    # 倾角安全阈值：设为 0.4 rad（约 23°），在该范围内不施加惩罚
    safe_angle = 0.4
    angle_violation = max(0.0, abs(hull_angle) - safe_angle)

    w_angle = 1.5
    w_angvel = 0.01
    w_vertical = 0.01

    stability_penalty = -(
        w_angle * angle_violation ** 2 +
        w_angvel * hull_angvel ** 2 +
        w_vertical * vertical_vel ** 2
    )

    # ====== 能量效率惩罚 ======
    w_energy = 0.01
    energy_penalty = -w_energy * (
        action[0] ** 2 + action[1] ** 2 +
        action[2] ** 2 + action[3] ** 2
    )

    total_reward = forward_reward + stability_penalty + energy_penalty
    components = {
        'forward_reward': forward_reward,
        'stability_penalty': stability_penalty,
        'energy_penalty': energy_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前 reward 缺少摔倒前兆的显式门控，全时二次姿态惩罚可能抑制高速探索。
- **behavior**: agent 能够稳定行走至接近满分，len 缩短而 score 微增表明速度提升有效，但需要释放更多速度潜力。
- **signal**: 缺失对危险倾角的软保护，同时存在对小偏差的持续惩罚。
- **level**: Level 2
- **hypothesis**: 将姿态惩罚改为倾角 hing 形态，解放安全区内小幅度摆动的惩罚，同时保留对危险倾角的强斥力，agent 将更敢于探索高速步态，从而提升最终得分。
- **risk**: 若安全阈值设置偏大，agent 可能偶尔在危险边缘试探从而增加摔倒概率，但 hinge 惩罚在越界后呈平方增长，应能有效制止。若倾角频繁越界，则需要适当降低阈值或提高系数。