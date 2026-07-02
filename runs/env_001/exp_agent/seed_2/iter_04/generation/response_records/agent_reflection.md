# Response Record

## 诊断

### Agent 发生了什么？
- Episode 完整跑完（len=1000），没有 crash，得分 143.84（较 iter2 的 187.93 下降）
- `soft_landing_continuous` mean=0.237，是 `progress`（0.003）的 **~83 倍**，完全主导总奖励
- nonzero_rate=61.7%：连续形式比之前的二值形式（~51%）更宽松，agent 在大部分时间都能蹭到部分 landing 奖励
- 连续乘积形式虽然提供了梯度，但阈值太宽、系数太大，导致 agent 在目标附近徘徊即可获得可观奖励，不需要高效完成任务

### 哪个组件是主因？
`soft_landing_continuous`。83x progress 的比率说明 landing 信号淹没了主学习信号。

### 上一轮改了什么？
Iter2→Iter3：把 `soft_landing_proxy`（二值）改为 `soft_landing_continuous`（连续乘积），系数保持 0.5。结果得分从 187.93 降到 143.84——连续化反而恶化了 exploitation。

### 本轮策略
**层次 1：改系数**。保持连续乘积的数学形态（梯度好），但大幅削弱系数（0.5→0.08）并收紧 prox 阈值（0.3→0.2），把 landing 信号压到合理范围（目标 ~5-10x progress），让 progress 重新成为主导信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 1. 提取观察量 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    vel_x, vel_y = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 2. 主学习信号：进度差奖励（不动）──
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_old - dist_new

    # ── 3. 轻量稳定约束（不动）──
    stability_penalty = -0.002 * (abs(vel_x) + abs(vel_y)) \
                        -0.002 * abs(angle) \
                        -0.002 * abs(angular_vel)

    # ── 4. 连续软着陆引导信号 ──
    # 上一轮：coefficient=0.5, prox_threshold=0.3 → mean=0.237, ratio=83x progress
    # 本轮：coefficient→0.08, prox_threshold→0.2，目标 ratio≈5-10x
    speed = (vel_x ** 2 + vel_y ** 2) ** 0.5

    prox_factor = max(0.0, 1.0 - dist_new / 0.2)       # dist=0→1, dist≥0.2→0（收紧）
    speed_factor = max(0.0, 1.0 - speed / 0.5)          # speed=0→1, speed≥0.5→0
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.3)    # angle=0→1, |angle|≥0.3→0
    leg_factor = 0.5 * (left_contact + right_contact)   # 两腿→1, 单腿→0.5, 无→0

    # 系数从 0.5 降至 0.08，大幅削弱 landing 信号的主导地位
    soft_landing_continuous = 0.08 * prox_factor * speed_factor * angle_factor * leg_factor

    # ── 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_continuous

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_continuous": soft_landing_continuous,
        "total_reward": total_reward
    }

    return float(total_reward), components
```
