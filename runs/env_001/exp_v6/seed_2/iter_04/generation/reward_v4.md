## 诊断

### 1. Agent 发生了什么？
Episode 长度 1000（满）→ agent **没有 crash**，但得分 -12.26 仍为负。agent 学会了在原点附近"悬停"——保持低速、直立、接近原点，持续收取 `approach_bonus`，但没有完成实际 landing。这是典型的 **proxy reward hacking**：agent 优化了奖金函数而非真实目标。

### 2. 哪个组件是主要原因？
`approach_bonus` 的 `ratio_to_progress = 1584x`，均值 2.71 vs progress_delta 的 0.0017。agent 每步从"靠近原点 + 低速 + 直立"拿到的奖励是 progress 的 1584 倍。悬停是最优策略——完全不需要真正着陆。

### 3. 我上一轮改了什么？
Iter 3 把 `both_legs_contact` 二值门控去掉了，改成纯连续乘积。这成功让 nonzero_rate 从极低升到 98.3%，但也造成了信号过度密集——agent 在任何靠近原点的地方都能持续获得高分。**问题不是系数大小，是数学形态**：一个不依赖 progress 的 per-step 状态奖励天然会诱导悬停。

### 修复方案（层次 2：改数学形式）
将 `approach_bonus` 乘以 `max(0, progress_delta)`，使奖金只在 agent **实际向原点靠近** 时才发放。悬停（progress_delta ≈ 0）拿不到奖金。这是一个关键的结构性修正：奖金从"处于好状态"变为"以好姿态靠近"。

同时，由于乘了 progress_delta（~0.002），scale 需要相应调整以保持信号存在但不压倒 progress。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal) ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty ----
    distance_penalty = -0.005 * d_next

    # ---- stability penalty ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.005 * abs(vx)
        - 0.005 * abs(vy)
        - 0.02 * abs(angle)
        - 0.02 * abs(ang_vel)
    )

    # ---- approach quality bonus (KEY FIX: gated by actual progress) ----
    # Previous version paid agent for "being in a good state" every step,
    # creating a hover-and-collect local optimum.  Now the bonus only fires
    # when the agent is actually approaching (progress_delta > 0), converting
    # it from a "state reward" to a "quality-of-progress amplifier".
    APPROACH_DIST = 2.0
    SPEED_THRESH = 2.0
    ANGLE_THRESH = 0.5

    proximity = max(0.0, 1.0 - d_next / APPROACH_DIST)
    speed_ok = max(0.0, 1.0 - (abs(vx) + abs(vy)) / SPEED_THRESH)
    angle_ok = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)

    # progress_delta ~0.002 on average when positive; quality factors ~0.5-0.8.
    # scale=2.0 gives bonus ~0.002-0.005 per approaching step, comparable to
    # progress_delta itself — a meaningful amplifier, not a dominating signal.
    approach_bonus = proximity * max(0.0, progress_delta) * speed_ok * angle_ok * 2.0

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + approach_bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "approach_bonus": approach_bonus,
        "total_reward": total,
    }
    return float(total), components
```