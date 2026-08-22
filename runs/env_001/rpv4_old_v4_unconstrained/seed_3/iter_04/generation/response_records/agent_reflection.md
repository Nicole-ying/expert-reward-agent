# Response Record

## 诊断报告

### evidence

**全局演变（iter_1 → iter_3）：**

| 指标 | iter_1 | iter_2 | iter_3 |
|---|---|---|---|
| score | -32.4 | 79.8 | 129.7 |
| len | 1000 | 316 | **725** |
| terminated | 0/20 | 20/20 | **9/20** |
| contact_bonus ep_sum | 0 | 442 (92%) | **483 (60%)** |
| landing_approach ep_sum | — | 17 (3.6%) | **305 (38%)** |

**Checkpoint 趋势（iter_3）关键信号：**
- score 在 cp50% 达到峰值 **157.0**，随后持续下跌至 cp100% 的 **111.0**——经典 exploitation 模式：agent 学会刷分但外部表现退化。
- contact_bonus 从 cp40% 的 145 猛增至 cp50% 的 **511**，active_rate 仅 8.4% 却贡献 60% 量级。
- landing_approach 从 cp70% 的 196 跳至 cp80% 的 **344**，随 episode 变长持续累积。

**核心发现：v3 的 "one-shot" contact_bonus 并非真正 one-shot。** `new_contact = has_contact * (1 - had_contact)` 在 agent 弹跳（离地→再触地）时反复触发。ep_sum=483 说明平均每 episode 触发约 60 次（483/8≈60），即 agent 在反复弹跳。这解释了 len 翻倍（316→725）和 terminated 暴跌（20/20→9/20）：agent 学会"弹跳刷 contact_bonus + 悬停刷 landing_approach"。

velocity_penalty 从 -5.0 降至 -2.8（尽管 w_v 增大），因为 agent 学会了减速以规避惩罚——这本身是好的，但速度惩罚太弱，不足以阻止弹跳。

### behavior_diagnosis

当前策略已偏离任务目标：agent 不是在"尽快稳定着陆"，而是在目标附近**悬停+弹跳**。具体行为模式：
1. 接近 (x≈0, y≈0) 后不果断着陆，而是在低空悬停，累积 landing_approach（0.6×4个指数因子，每步 ≈0.4）
2. 间歇性触地→弹起→再触地，反复触发 contact_bonus（每次 ≈8.0 × spatial_gate × soft_gate）
3. 725 步长 episode，仅 9/20 自然终止（body settled），11/20 被截断——agent 无法/不愿稳定着陆
4. cp50% 后 score 下跌说明 exploitation 在训练后期加剧

**根因**：contact_bonus 的 transition-detection 机制可被弹跳反复触发，landing_approach 的持续累积激励悬停。两者形成"悬停刷分循环"。

---

### 干预策略 (v4)

1. **删除 transition-based contact_bonus** → 替换为 per-step `settled_reward`：只有双腿持续接触 + 紧邻原点 + 极低速 + 直立才给分。弹跳时接触断开，奖励立即中断。
2. **新增 `contact_speed_penalty`**：接触状态下速度越大惩罚越重，直接惩罚弹跳行为。
3. **`landing_approach` 权重从 0.6 降至 0.20**：降低悬停累积收益。
4. **`velocity_penalty` 新增下坠速度分量**：接近地面时向下速度被强力惩罚（反重击）。
5. **`goal_proximity` 微调**：降低 w_dist、w_approach 以免与 landing_approach 重复，让两个信号分工明确（goal_proximity 负责"靠近"，landing_approach 负责"调整姿态准备着陆"）。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: eliminate bounce-exploitable contact_bonus; replace with per-step settled_reward.
    
    Core changes:
    - REMOVED transition-based contact_bonus (bounce-exploitable).
    - ADDED settled_reward: per-step, both legs must STAY in contact, tight gates.
    - ADDED contact_speed_penalty: speed while touching ground is punished (anti-bounce).
    - REDUCED landing_approach weight 0.6 → 0.20 (anti-hover).
    - ADDED downward-velocity penalty near ground (anti-slam).
    - REBALANCED goal_proximity for cleaner signal separation.
    """

    # --- unpack next_obs ---
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_cont = next_obs[6]
    right_cont = next_obs[7]

    # --- unpack obs (for delta computations) ---
    x_cur = obs[0]
    y_cur = obs[1]

    # --- derived quantities ---
    dist = (x ** 2 + y ** 2) ** 0.5
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    speed_sq = vx ** 2 + vy ** 2
    both_legs = left_cont * right_cont          # 0.0 or 1.0

    # ================================================================
    # 1. goal_proximity — distance penalty + approach incentive
    #    Moderate weights. landing_approach handles fine positioning.
    # ================================================================
    w_dist = 0.06
    proximity_penalty = -w_dist * dist

    w_approach = 0.30
    delta_dist = dist_cur - dist                 # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ================================================================
    # 2. velocity_damping — general speed penalty + anti-slam near ground
    #    w_v increased, plus dedicated downward-velocity penalty.
    # ================================================================
    w_v = 0.28
    k_gate = 5.0
    gate = 1.0 / (1.0 + k_gate * dist)           # ~1 near pad, ~0 far away

    # anti-slam: heavy penalty for downward velocity when very near ground
    w_slam = 1.0
    slam_gate = 1.0 / (1.0 + 20.0 * dist)        # ultra-tight: only |dist|<0.1 matters
    downward_v = max(0.0, -vy)                   # positive = moving downward

    velocity_penalty = (
        -w_v * gate * speed_sq
        - w_slam * slam_gate * (downward_v ** 2)
    )

    # ================================================================
    # 3. orientation_penalty — unchanged structure, slightly higher weights
    # ================================================================
    w_ang = 0.05
    w_angvel = 0.01
    orientation_penalty = -w_ang * abs(angle) - w_angvel * (angvel_nxt := angvel) ** 2

    # ================================================================
    # 4. landing_approach — continuous shaping for pre-contact guidance
    #    Weight reduced 0.6 → 0.20. Temperature tightened slightly.
    #    Provides a broad basin of attraction toward landing configuration.
    # ================================================================
    near_y   = 2.718281828 ** (-abs(y) / 0.22)
    near_x   = 2.718281828 ** (-abs(x) / 0.30)
    slow     = 2.718281828 ** (-speed_sq / 0.45)
    upright  = 2.718281828 ** (-abs(angle) / 0.28)
    landing_approach = 0.20 * near_y * near_x * slow * upright

    # ================================================================
    # 5. settled_reward — per-step reward for STAYING landed properly
    #    Replaces old contact_bonus entirely. Only pays when:
    #    - both legs are ON the ground (sustained, not transition)
    #    - VERY close to (0,0) in position
    #    - VERY low speed
    #    - near-upright orientation
    #    Tight gates mean bouncing → near-zero reward (no re-trigger exploit).
    # ================================================================
    w_settled = 3.5
    s_spatial  = 2.718281828 ** (-(x ** 2 + y ** 2) / 0.04)
    s_slow     = 2.718281828 ** (-speed_sq / 0.03)
    s_upright  = 2.718281828 ** (-abs(angle) / 0.10)
    settled_reward = w_settled * both_legs * s_spatial * s_slow * s_upright

    # ================================================================
    # 6. contact_speed_penalty — anti-bounce: speed while in contact is BAD
    #    Directly punishes the bouncing exploit. Strong enough to matter.
    # ================================================================
    w_contact_spd = 2.0
    contact_speed_penalty = -w_contact_spd * both_legs * speed_sq

    # ================================================================
    # 7. fuel_efficiency — moderate regularisation
    # ================================================================
    w_fuel = 0.006
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ================================================================
    # assemble
    # ================================================================
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + settled_reward
        + contact_speed_penalty
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'settled_reward': settled_reward,
        'contact_speed_penalty': contact_speed_penalty,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components
```
