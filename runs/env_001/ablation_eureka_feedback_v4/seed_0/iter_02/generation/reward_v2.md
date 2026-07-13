# 设计理由
stability 组件当前对速度/角度/角速度的惩罚过弱（平均每步仅占主靠近信号的约 7%），而 agent 已能接近目标但全部以失败终止，大概率是高速或大角度碰撞所致。因此将 stability 整体系数放大 2 倍，在不改变数学形态下强化减速与姿态平稳的引导力，放大后 per‑step 惩罚约为主信号的 15%，不会压倒接近动机。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    distance = (x**2 + y**2) ** 0.5
    proximity = -1.0 * distance

    vel_penalty = -0.05 * (vx**2 + vy**2)
    ang_penalty = -0.02 * (ang_vel**2)
    angle_penalty = -0.5 * (angle**2)
    stability = 2.0 * (vel_penalty + ang_penalty + angle_penalty)

    contact = 0.5 * (left_contact + right_contact)

    total = proximity + stability + contact

    components = {
        "proximity": proximity,
        "stability": stability,
        "contact": contact
    }
    return float(total), components
```

# 诊断摘要
- **evidence**: terminated=20/20 全部失败，平均每步距中心 ~0.97，contact 步占比 ~2.5%
- **behavior**: 着陆器已能靠近目标区域，但无法稳定停靠，在接近时因速度/姿态控制不良而坠毁
- **signal**: stability 惩罚过于微弱（per‑step 仅 -0.07），不足以在接近段强制减速
- **level**: Level 1（尺度修复），stability 组件系数放大 2 倍
- **hypothesis**: 更强的稳定性惩罚将促使 agent 在接近目标时主动降低线速度、角速度和角度偏差，避免碰撞失败
- **risk**: 惩罚过强可能导致 agent 在远离目标时过早减速，造成徘徊或延迟到达，但预期仍能保持接近能力