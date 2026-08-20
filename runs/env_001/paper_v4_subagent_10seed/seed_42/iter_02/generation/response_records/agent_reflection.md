# Response Record

# 设计理由

## 第 0 步审计
- **终止模式**：所有 episode 均以 `terminated` 结束（无 truncation），长度 436 步，说明 agent 已学会到达目标并触发 `body_not_awake_or_settled` 稳定着陆。不存在过早失败。
- **观测使用**：全部 8 维观测均被代码引用，无缺失信号。
- **信号缺口**：无。问题在于 **信号校准**——`progress_delta` 激励过于温和（线性，per‑step 仅 0.0027），导致 agent 缺乏快速接近目标的动力，大量时间消耗在缓慢漂移和终端软着陆区域徘徊。
- **僵尸组件**：无（所有组件 active_rate > 2%）。

## 行为诊断
Agent 学会了软着陆（`soft_landing` 贡献 93.7% 分数），但接近过程极度缓慢，整个 episode 步数偏高。`progress_delta` 是推动 agent 向目标前进的唯一正向信号，但其线性形态无法区分大步快速接近与小步慢移——agent 自然选择最“安全”的最低消耗策略，牺牲速度以降低风险。

## 干预层级：Level 2 — 结构变换
基于 `Formula switching guide` 中 **“线性正奖励 w * signal，score 停滞在低水平”** 对应的 **dense_state_signal（凸化）**，将 `progress_delta` 从线性改为 **线性 + 二次项**，使得每步较大的位置改善获得超线性奖励，激励 agent 加快靠近目标。

## 系数校准
- 原 `progress_delta` 平均 per‑step ≈ 0.0027，新公式对典型步长 0.1 给出 reward ≈ 0.1 + 2*0.01 = 0.12，与主信号 `soft_landing` per‑step (0.064) 量级相当，不会淹没终端软着陆激励。
- `soft_landing` 的 proximity 门控（dist<0.2）会自然迫使 agent 在终端段减速，防止因前期过快导致的失控。
- `orientation_penalty` 保持不变（极小惩罚），不额外增加负担，遵守总惩罚 ≤ 主信号 0.5x 的要求。

## 风险
agent 可能在前半段变得过于激进，速度过高，导致进入 proximity 区域时难以平稳减速，引发 crash 或增加调整步数。但 `soft_landing` 本身提供的减速奖励（speed_factor）应能提供反向信号，引导它在最后阶段减速。若下一轮出现大量 crash，需考虑引入速度软约束（如 gate）。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Convexified progress reward to incentivize faster approach.
    """
    # ---- Unpack observations ----
    px0, py0 = obs[0], obs[1]
    px1, py1 = next_obs[0], next_obs[1]
    vx1, vy1 = next_obs[2], next_obs[3]
    angle1  = next_obs[4]
    angvel1 = next_obs[5]
    left_leg  = next_obs[6]
    right_leg = next_obs[7]

    # ---- 1. Progress to target: convex combination of linear + quadratic ----
    dist_prev = (px0**2 + py0**2) ** 0.5
    dist_next = (px1**2 + py1**2) ** 0.5
    raw_progress = dist_prev - dist_next       # positive when approaching
    progress = max(0.0, raw_progress)           # only reward net progress
    progress_reward = progress + 2.0 * progress**2   # convex -> bigger steps get more

    # ---- 2. Orientation / stability soft constraints (unchanged) ----
    angle_penalty  = -0.01 * (angle1 ** 2)
    angvel_penalty = -0.005 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ---- 3. Soft landing guidance (unchanged) ----
    speed1 = (vx1**2 + vy1**2) ** 0.5
    proximity_threshold = 0.2
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor = 1.0 / (1.0 + 10.0 * speed1)
        soft_landing = contact_factor * speed_factor
    else:
        soft_landing = 0.0

    # ---- Combine components ----
    total_reward = (
        1.0 * progress_reward
        + 1.0 * orientation_penalty
        + 1.0 * soft_landing
    )

    components = {
        "progress_delta": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_landing": soft_landing
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号齐全，无缺失维度，问题在于进度信号的激励强度不足。
- **behavior**: agent 缓慢漂移并成功软着陆，但 episode 长度偏长，得分未充分利用快速接近的潜力。
- **signal**: `progress_delta` 线性奖励过弱，无法区分大步快速接近与小步慢移。
- **level**: Level 2 — 结构变换（凸化 progress 奖励）。
- **hypothesis**: 二次项使大步前进的奖励显著更高，agent 将更早到达 proximity 区域并仍由 `soft_landing` 引导稳定着陆，从而在更短 episode 内获得相等或更高的总分。
- **risk**: 前期过于激进可能导致速度过大，增大终端减速压力或 crash 风险；但 `soft_landing` 的 speed 惩罚应提供必要的反向信号。
