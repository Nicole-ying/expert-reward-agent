# 设计理由
contact_proximity 将着陆奖励与距离紧密耦合，导致双腿接触时奖励受限（均值仅8.03），多数episode因超时截断，agent缺乏足够动力完成着陆。改为 contact_bonus = 20.0 * lc * rc，移除距离门控，大幅增强“双腿接触”这一关键行为信号。系数20基于每episode约20接触步的估计，使接触奖励成为主导，驱动agent快速着陆，而 proximity 和 velocity 惩罚仍负责稳定定位与减速。

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

    contact_bonus = 20.0 * next_lc * next_rc

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_bonus': contact_bonus
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: score=33.6, terminated=5/20, truncated=15/20, contact_proximity均值仅8.03，且受 `1/(1+5*dist)` 抑制。
- **behavior**: agent大部分episode超时，未能稳定着陆，徘徊且接触信号过弱。
- **signal**: 着陆接触奖励稀疏且条件严格（需同时接触且距离近），不足以引导快速着陆。
- **level**: Level 2——稀疏二值proxy去距离门控并放大为固定bonus，强化关键行为信号。
- **hypothesis**: 移除距离抑制并大幅提高接触奖励，将促使agent优先触发双腿接触，结合现有位置与速度惩罚，实现稳定着陆。
- **risk**: 若agent能在非平台表面触发接触，可能错误刷分，但在该任务中接触仅发生在着陆平台；接触奖励过强可能导致忽略速度控制，但 velocity_penalty 仍保持制衡。