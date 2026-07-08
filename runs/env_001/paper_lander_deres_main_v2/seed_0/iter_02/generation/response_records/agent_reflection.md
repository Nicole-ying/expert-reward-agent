# Response Record

## 诊断

### 1. Evidence

| 证据 | 数值 | 含义 |
|---|---|---|
| score | -108.90，全 20 轮 early terminal (<150步, score<-50) | 策略快速失败，无法到达目标 |
| stability_penalty episode_sum_mean | -9.36，magnitude_share 87.7% | 稳定性惩罚完全主导总奖励 |
| progress_delta episode_sum_mean | +1.12，magnitude_share 10.9% | 进度信号被淹没，无法引导学习 |
| soft_landing_proxy active_rate | 0.4% | 着陆奖励几乎不触发，信用分配断裂 |
| episode_length | 68.45 | 远早于正常着陆截断，agent 无法存活足够久 |

agent 每走一步都被速度/角度/角速度惩罚，累计惩罚远超进度奖励。agent 为避免惩罚而抑制所有动作，却因此无法导航到目标——整个奖励函数变成了"刹车踩死"。

### 2. Diagnosis dimension

**global_to_local_gate**：稳定性约束是全局的（每步生效），但它在远离目标的探索阶段毫无必要且有害。agent 需要先飞到目标附近，然后才需要减速稳定。当前全局惩罚让 agent 不敢动。

### 3. Selected transformation

`global_to_local_gate`：用距离门控把稳定性惩罚限制在目标附近区域。远离目标时惩罚归零，agent 可自由导航；靠近时惩罚逐渐激活，迫使减速和姿态修正。

### 4. Falsifiable hypothesis

取消远距离的速度惩罚后，agent 将能有效利用 progress_delta 信号学会接近目标，episode 长度和得分应显著改善；靠近目标时门控激活的局部惩罚会引导着陆减速。

### 5. Expected next round

- `stability_penalty` 的 `episode_sum_mean` 绝对值应显著下降（远距离部分被门控归零）
- `progress_delta` 应上升（agent 敢动了）
- `episode_length` 应增长（存活更久）
- `score` 应改善（不再被全局惩罚拖累）
- `soft_landing_proxy` active_rate 可能仍然很低，但那是下一步的问题

### 6. Main risk

agent 可能在远距离高速冲向地面导致 crash（没有速度约束），但 progress_delta 提供朝向目标的梯度，且 crash 本身会截断 episode 限制负收益——agent 有机会从失败中学习减速时机。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测与下一观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 主学习信号：progress delta ----------
    def dist(px, py):
        return (px**2 + py**2)**0.5

    prev_dist = dist(x, y)
    next_dist = dist(next_x, next_y)
    progress_delta = prev_dist - next_dist    # 正值 = 靠近目标

    # ---------- 距离门控：稳定性惩罚仅在靠近目标时激活 ----------
    speed_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    angvel_penalty = abs(angular_vel)

    # 门控因子：dist=0 时为 1.0（全效），dist>=1.5 时为 0.0（关闭）
    gate_threshold = 1.5
    stability_gate = max(0.0, 1.0 - next_dist / gate_threshold)

    stability_penalty = stability_gate * (
        -0.1 * speed_penalty - 0.1 * angle_penalty - 0.05 * angvel_penalty
    )

    # ---------- 任务完成近似信号：soft landing proxy ----------
    both_contact = left_contact + right_contact   # 两个都接触时 = 2.0
    near_target = abs(next_x) < 0.2 and abs(next_y) < 0.2
    low_speed = (abs(vx) + abs(vy)) < 0.15
    upright = abs(angle) < 0.2
    if both_contact == 2.0 and near_target and low_speed and upright:
        soft_landing_proxy = 0.5
    else:
        soft_landing_proxy = 0.0

    # ---------- 合成 ----------
    total_reward = progress_delta + stability_penalty + soft_landing_proxy

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_proxy': soft_landing_proxy,
    }
    return float(total_reward), components
```
