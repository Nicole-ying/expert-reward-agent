# Response Record

# 设计理由
将 `landing_improvement`（基于角度/速度的变化量）替换为 `landing_quality`（基于即时状态的有界连续 proxy），原因是变化量无法区分 crash 前瞬间的速度降低与真正的姿态改善，而即时状态奖励能直接鼓励接触垫子、低空减速和平稳姿态。新组件在近地面（`ny < 0.3`）时施加角度惩罚、垂直速度惩罚和双足接触奖励，高空仅保留轻微姿态提示，所有子项系数经过手工校准，使单步预期绝对值不超过 progress 的 2 倍，且接触奖励在成功着陆时能提供强正向信号，引导 agent 完成停车。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nleft, nright = next_obs[6], next_obs[7]

    # 1. Proximity: 向目标移动的距离减少量
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_next = (nx ** 2 + ny ** 2) ** 0.5
    progress = max(0.0, dist_curr - dist_next)

    # 2. Landing quality: 基于即时状态的着陆姿态、速度与接触奖励
    if ny < 0.3:
        angle_penalty = -0.2 * abs(nangle)
        vel_penalty = -0.1 * max(0.0, -nvy - 0.2)
        contact_reward = 1.0 * (nleft + nright)
        landing_quality = angle_penalty + vel_penalty + contact_reward
    else:
        landing_quality = -0.05 * abs(nangle)

    # 3. Energy penalty: 惩罚不必要的引擎使用
    energy_penalty = -0.01 if action != 0 else 0.0

    total_reward = progress + landing_quality + energy_penalty

    components = {
        "proximity": progress,
        "landing_quality": landing_quality,
        "energy_penalty": energy_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: score=-2471（全 terminated，len≈303），proximity 均值 0.037，landing_improvement 均值 1.414，energy_penalty -3.03，terminal_velocity_penalty 始终为 0
- **behavior**: agent 几乎全部 episode 快速终止（可能 crash 或 settle），但外部分数极低，说明当前行为（硬着陆或碰撞）对任务无益
- **signal**: 原有的 landing_improvement 虽然提供正向奖励，但未能提升外部分数，缺失与任务成功直接挂钩的即时状态信号（姿态、速度、接触）
- **level**: Level 2——替换组件，将稀疏/间接的改善信号替换为密集的即时状态 proxy
- **hypothesis**: 新 landing_quality 在接近地面时直接奖励小角度、低下降速度和双足接触，能引导 agent 在着陆阶段保持平稳并完成安全停车，从而改善外部任务得分
- **risk**: 若接触奖励系数过大，agent 可能在未完全稳定时重复触发接触以获取奖励，但通过捆绑姿态与速度惩罚已限制此类 exploit；早期下降阶段的小幅负奖励可能导致探索缓慢，但被 progress 正向信号补偿
