## 诊断报告

### Evidence

| 信号 | 数据 |
|---|---|
| **Score** | -117.24（目标 200），全部 10 个 checkpoint 在 [-126, -117] 窄区间，**完全平坦** |
| **Episode length** | 68.45，20/20 early termination（<150 步且 score<-50） |
| **proximity_reward** | ep_sum≈-20，100% active，magnitude_share 41.8%，**10 个 checkpoint 全部平坦（-19.96 ~ -20.19）** |
| **vel_damping** | ep_sum≈-21，100% active，magnitude_share 44.1%，**10 个 checkpoint 全部平坦（-21.07 ~ -21.30）** |
| **proximity + vel_damping** | 合计 magnitude_share **85.9%**，且均为**每步负值** |
| **landing_bonus** | ep_sum=2.5，active_rate **0.4%**（≈0.27步/集），条件过严 |
| **contact_reward** | ep_sum=1.625，active_rate **3.1%**（≈2步/集，仅坠毁前瞬间触发） |
| **orientation_penalty** | ep_sum=-0.54，1.1% share，权重太轻 |

### Behavior Diagnosis

**核心问题：奖励函数催生了"自杀最优"策略。**

proximity_reward（`-0.3*dist`）和 vel_damping（`-0.4*gate*speed`）始终为负，合计约 -0.6/步。agent 发现：越早终止（crash），累积负奖励越少。episode 长度从 iter_1 的 303 骤降至 68，正是 agent 学会了"快速 crash = 更高回报"。所有 checkpoint 分数完全平坦，证明此奖励函数**没有任何学习梯度**——agent 从一开始就收敛到 crash 策略。

正向信号（landing_bonus 0.4%、contact_reward 3.1%）过于稀疏，无法在探索中被发现。landing 条件（dist<0.3, angle<0.15, speed<0.3, both feet）对从未成功着陆的 agent 来说是不可能触发的。

**策略必须从根本上重写**：将主导的负信号转为正信号或使用 excess-penalty 结构，同时大幅放宽成功条件。

---

## 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: Radical rewrite — eliminate suicide incentive.
    
    Core principles:
    - Proximity is ALWAYS POSITIVE (bounded, 1/(1+dist)) — no suicide gradient
    - Progress reward (delta-distance) for efficient navigation
    - Velocity damping only penalizes EXCESS speed near pad (safe_speed threshold)
    - Relaxed landing conditions for discoverability
    - Contact as progress signal, not sparse rarity
    - Modest action/orientation shaping
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    x_prev = obs[0]
    y_prev = obs[1]

    # ── 1. Proximity bonus: always positive, bounded ──
    # Maximum +2.0 at dist=0, decays smoothly, never negative
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    proximity_bonus = 2.0 / (1.0 + dist_curr)

    # ── 2. Progress reward: delta-distance shaping ──
    # Positive when approaching pad, negative when retreating
    # Bounded total: roughly 2 * (initial_dist - final_dist)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    progress_reward = 2.0 * (dist_prev - dist_curr)

    # ── 3. Orientation penalty (moderate) ──
    orientation_penalty = -0.15 * (angle ** 2) - 0.06 * (angvel ** 2)

    # ── 4. Velocity damping: only EXCESS speed near pad ──
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # Gaussian gate: ~1.0 at pad, ~0.37 at dist=1.0, ~0.02 at dist=2.0
    proximity_gate = 2.718281828 ** (- (dist_curr ** 2) / 1.0)
    safe_speed = 0.35          # speed below this is free near pad
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -1.0 * proximity_gate * excess_speed

    # ── 5. Contact reward: any foot on ground is progress ──
    contact_sum = left_contact + right_contact
    contact_reward = 1.0 * contact_sum

    # ── 6. Landing success bonus (relaxed thresholds) ──
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist_curr < 0.8)
    upright = (abs(angle) < 0.4)
    slow = (speed < 0.8)
    is_good_landing = both_on_ground and near_pad and upright and slow
    landing_bonus = 10.0 if is_good_landing else 0.0

    # ── 7. Action efficiency ──
    if action == 0:            # glide — small reward for not wasting fuel
        action_reward = 0.02
    elif action == 2:          # main engine — highest cost
        action_reward = -0.06
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.03

    # ── Aggregate ──
    total_reward = (
        proximity_bonus
        + progress_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "proximity_bonus": proximity_bonus,
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
    }
    return float(total_reward), components
```