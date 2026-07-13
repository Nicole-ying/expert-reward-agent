# 设计理由
接触奖励的转移形态（`next_lc*next_rc` 与上一步的差值）导致 agent 通过反复弹跳（接触→脱离→再接触）刷取大量 bonus，平均值高达 341，占主导地位，但实际外部得分仅 170，说明刷分未转化为真实稳定着陆。因此将 **转移事件** 改为 **密集状态奖励**：`contact_reward = 1.0 * (next_lc * next_rc)`，每步只要双接触即给予常量奖励，不再有一次性大额跳变，且配合已有的速度/角度惩罚自然抑制弹跳。系数 1.0 在接触后与归零的 proximity 相比不会破坏主导权，也不会造成拖延（proximity 惩罚在远离目标时远大于 1.0）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    next_lc = next_obs[6]
    next_rc = next_obs[7]

    dist = (x**2 + y**2) ** 0.5

    proximity_reward = -0.1 * (x**2 + y**2)

    vel_gate = 1.0 / (1.0 + 5.0 * dist)
    velocity_penalty = -0.1 * (vx**2 + vy**2) * vel_gate

    orientation_penalty = -0.5 * (angle**2) - 0.1 * (av**2)

    # 密集状态奖励，每步双接触时给予固定奖励，取代转移事件
    contact_reward = 1.0 * (next_lc * next_rc)

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + contact_reward

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_reward': contact_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: contact_bonus 均值 341，远大于其他组件总和，且 score 170 远低于该奖励水平，表明存在刷分 exploit。
- **behavior**: agent 学会了通过反复弹跳（接触-脱离-再接触）频繁触发转移事件，获取大量奖励，但未能稳定停靠，导致部分 episode 得分很高、部分崩溃。
- **signal**: 转移事件奖励被重复领取，缺少对接触持续性的稳定奖励，弹跳行为未被抑制。
- **level**: Level 2，数学结构变换：从转移事件 → 密集状态奖励。
- **hypothesis**: 每步双接触固定奖励 + 现有速度/角度惩罚会将行为引导至“尽快稳定双接触”，不再通过弹跳来获取奖励，从而提升真实着陆成功率。
- **risk**: 双接触期间的持续奖励可能使 agent 在接触后略微延迟终止（累积少量步数），但 proximity 惩罚驱动快速接近，且终止条件会自然停止。