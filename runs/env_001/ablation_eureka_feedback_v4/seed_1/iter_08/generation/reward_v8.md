# 设计理由
上一轮的线性距离惩罚（-0.5*dist）导致 agent 为减少累积负奖励而快速 crash，success_reward 因接触稀疏无法提供有效引导。本轮 Level 3 重建主骨架，将距离惩罚替换为每步 progress 奖励（奖励靠近目标的距离变化），同时用强化的速度门控惩罚（-5.0*speed²/(1+3*dist)）强制减速，再保留小权重姿态惩罚和接触奖励。progress 奖励消除了“自杀以停止扣分”的动机，速度惩罚在近处足够强（当 dist=1、speed=1 时惩罚约 -1.25）以阻止快速冲撞，接触奖励提供着陆后的小额持续激励，促使稳定停靠。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    lc = next_obs[6]
    rc = next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5

    # 进步奖励：向目标靠近
    progress = prev_dist - dist
    progress_reward = 2.0 * progress

    # 速度惩罚：高速时惩罚，接近目标时放大
    vel_gate = 1.0 / (1.0 + 3.0 * dist)
    velocity_penalty = -5.0 * (speed**2) * vel_gate

    # 姿态惩罚
    orientation_penalty = -0.1 * (angle**2) - 0.05 * (av**2)

    # 接触奖励：双接触 + 低速 + 竖直
    contact_bonus = 10.0 * lc * rc * max(0.0, 1.0 - speed) * max(0.0, 1.0 - abs(angle))

    total = progress_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'progress_reward': progress_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_bonus': contact_bonus
    }

    return float(total), components
```

# 诊断摘要
- **evidence**: terminated=20/20，len=68.45，proximity_reward 每步 -0.486 占主导，全部 episode 快速 crash
- **behavior**: agent 学会了为减少每步高额距离惩罚而高速冲向原点，导致失控碰撞
- **signal**: 缺少正向接近激励，负奖励累积迫使 agent 主动寻求终止；速度惩罚过弱，无法在近处强制减速
- **level**: Level 3（同一骨架家族已迭代≥3轮且最佳得分未超目标 5%，连续两轮未刷新 best，上一轮修改导致 crash）
- **hypothesis**: progress 奖励消除自杀动机，强化后的速度门控惩罚会在 target 附近强制减速，从而恢复稳定着陆并提高得分
- **risk**: progress 奖励可能导致 agent 在目标附近徘徊但不触发接触着陆，需要 contact_bonus 提供最终激励；速度惩罚若仍过强可能抑制有效移动，需观察后续训练。