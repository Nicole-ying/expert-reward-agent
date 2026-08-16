# Response Record

Now I have a comprehensive picture. Let me synthesize all evidence.

---

## 诊断报告

### evidence

| 来源 | 关键数据 | 解读 |
|---|---|---|
| **iter_3 goal_proximity** | ep_sum=**+1204**, magnitude_share=86%, len=1000 | 指数 proximity bonus 被 exploit：agent 悬停在中等距离收租，永不降落 |
| **iter_4 goal_proximity** | ep_sum=**+8.16**（正值！）, settling_stability=-28.49（59% magnitude） | potential-based shaping 让 agent **确实在靠近目标**，但稳定性惩罚太重拖垮总分 |
| **iter_4 landing_contact** | ep_sum=**+10.5**, active=0.8% | 着陆奖励**已经偶尔激活**，信号可触达，证明门控条件合理 |
| **iter_5 approach_reward** | ep_sum=**-38.58**，10 个 checkpoint 间完全平坦（±0.01 波动） | 纯状态惩罚 `-0.5*dist²` 是死信号——agent 不学习靠近，距离从未改善 |
| **iter_5 landing_bonus / boundary_penalty** | active=**0%**，全程为 0 | 两个组件完全死亡；landing 因 never reach target, boundary 因 |x|<1.0 未触发 |
| **iter_5 terminated** | **20/20** early termination, len=68.85 | agent 学到的策略是尽快 crash 终止，因为所有奖励都是负数 |

### behavior_diagnosis

**当前策略（iter_5）是不健康的「尽快自杀」行为**：所有 20 个 episode 在 ~69 步 crash 终止（terminated=20/20），无一存活。根因是奖励函数**全为负值**——approach_reward 是死板的 `-0.5*dist²`，全程约 -38.6 且不随训练改善；velocity_penalty 和 angle_penalty 也都是惩罚项；landing_bonus 和 boundary_penalty 从未激活。agent 的最优策略就是尽快触发终止以停止累积负奖励。

**对比 iter_4**：potential-based shaping 让 goal_proximity 为 **+8.16**（agent 确实在靠近），landing_contact 有 **0.8% 激活率且 +10.5**（着陆信号可触达）。问题只是 settling_stability（-28.49）和 attitude_constraint（-9.18）惩罚过重。iter_5 的错误方向是**完全抛弃了 iter_4 的有效信号**（delta 靠近 + 着陆引导），转向纯惩罚。

### 干预策略

**核心思路**：融合 iter_4 的有效组件（delta-based approach + progressive landing）与 iter_5 的优点（轻量惩罚），修复 iter_4 的缺陷（稳定性惩罚过重）。

1. **approach_reward**：恢复 delta 驱动 —— `w * (current_dist - next_dist)`，高权重（15.0）提供强梯度。纯 delta 不可 exploit（悬停收益=0，振荡净收益=0）。
2. **proximity_warmth**：微小状态奖励（max 0.05/step，dist≥3 时归零），解决冷启动问题——给 agent 活下来的理由，远小于着陆收益，不可 exploit。
3. **velocity/angle 惩罚**：近距门控 + 大幅降低系数（相比 iter_4 的 -28.49），确保远距离不干预机动。
4. **landing_bonus**：延续 iter_4 的渐进乘积设计（prox × speed × contact × angle），但改为连续因子（非二元门控），创造从「靠近」到「完美着陆」的平滑梯度。iter_4 证明 0.8% active 已经可触达，连续化会进一步提高。
5. **boundary_penalty**：保留但降低阈值到 1.2，更早预警。
6. **fuel_penalty**：保持轻量。

---

## 奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化 v6 — 融合 iter_4 有效信号 + iter_5 轻量惩罚。

    核心设计：
    - approach_reward：强 delta 驱动（15.0 * dist_improvement），不可 exploit
    - proximity_warmth：微小状态奖励，仅解决冷启动，远小于着陆收益
    - velocity/angle 惩罚：门控 + 轻量，远距离几乎为零
    - landing_bonus：渐进乘积式，从靠近到完美着陆的连续梯度
    - boundary_penalty：预防飞出视口
    - fuel_penalty：轻量引擎使用惩罚
    """

    # --- 当前状态 ---
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    # --- 下一状态 ---
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # --- 派生量 ---
    current_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5

    # ================================================================
    # 1. approach_reward — 强 delta 驱动（核心学习信号）
    #
    #    纯势能差：reward = 15 * (current_dist - next_dist)
    #    - 靠近目标 → 正奖励（每 0.1 单位 ≈ +1.5）
    #    - 远离目标 → 对称惩罚
    #    - 悬停/振荡 → 净收益 = 0（不可 exploit）
    #    总 episode 贡献 = 15 * (initial_dist - final_dist)，路径无关
    # ================================================================
    dist_improvement = current_dist - next_dist
    approach_reward = 15.0 * dist_improvement

    # ================================================================
    # 2. proximity_warmth — 微小状态奖励，仅防冷启动
    #
    #    线性衰减：dist=0 → 0.05, dist=1 → 0.033, dist=3 → 0
    #    即使 agent 在 dist=1 悬停 1000 步也只获得 ~33 点——远小于
    #    着陆后的 landing_bonus（可达 5+/步），不可 exploit。
    #    唯一作用：给早期随机探索的 agent 一个活下来的微弱理由。
    # ================================================================
    proximity_warmth = 0.05 * max(0.0, 1.0 - next_dist / 3.0)

    # ================================================================
    # 3. velocity_penalty — 近距门控速度管理
    #
    #    gate = 1/(1+5*dist)：dist=0→1.0, dist=1→0.17, dist=5→0.038
    #    远处几乎不干预，近处温和引导减速。
    #    系数 0.3 相比 iter_4 的 -1.0*next_speed*prox_gate 大幅降低，
    #    防止 iter_4 中 settling_stability 以 59% magnitude 碾压一切的问题。
    # ================================================================
    vel_gate = 1.0 / (1.0 + 5.0 * next_dist)
    velocity_penalty = -0.3 * vel_gate * (nvx**2 + nvy**2)

    # ================================================================
    # 4. angle_penalty — 近距门控姿态约束
    #
    #    比 iter_4/iter_5 都更温和，避免 attitude 惩罚阻碍中距离机动。
    #    gate 在远处趋近 0，近处才生效。
    # ================================================================
    angle_gate = 1.0 / (1.0 + 3.0 * next_dist)
    angle_penalty = -angle_gate * (1.0 * n_angle**2 + 0.5 * n_ang_vel**2)

    # ================================================================
    # 5. boundary_penalty — 水平越界预警
    #
    #    阈值 1.2（比 iter_5 的 1.0 稍宽）用二次惩罚在越界时急剧增大。
    # ================================================================
    safe_x_bound = 1.2
    if abs(nx) > safe_x_bound:
        boundary_penalty = -2.0 * (abs(nx) - safe_x_bound) ** 2
    else:
        boundary_penalty = 0.0

    # ================================================================
    # 6. landing_bonus — 渐进着陆质量奖励
    #
    #    四个连续因子（0~1）的乘积 × 12.0：
    #    - prox_factor:   dist=0→1.0,  dist≥0.4→0（靠近目标垫板）
    #    - speed_factor:  speed=0→1.0, speed≥0.4→0（低速着陆）
    #    - contact_factor: 0(无接触), 0.5(单腿), 1.0(双腿)
    #    - angle_factor:  |angle|=0→1.0, |angle|≥0.25→0（姿态平稳）
    #
    #    与 iter_4 的关键区别：全部使用连续因子而非二元门控，
    #    即使只满足部分条件也有部分奖励，创造平滑学习梯度。
    #    iter_4 证明 landing 信号可触达（active 0.8%, ep_sum +10.5）；
    #    连续化后 accessibility 应显著提升。
    # ================================================================
    prox_factor = max(0.0, 1.0 - next_dist / 0.4)
    speed_factor = max(0.0, 1.0 - next_speed / 0.4)
    contact_factor = 0.5 * (n_left + n_right)  # 0.0 / 0.5 / 1.0
    angle_factor = max(0.0, 1.0 - abs(n_angle) / 0.25)

    landing_bonus = 12.0 * prox_factor * speed_factor * contact_factor * angle_factor

    # ================================================================
    # 7. fuel_penalty — 轻量引擎使用惩罚
    # ================================================================
    fuel_penalty = -0.03 * float(action != 0)

    # ================================================================
    # 汇总
    # ================================================================
    total_reward = (
        approach_reward +
        proximity_warmth +
        velocity_penalty +
        angle_penalty +
        boundary_penalty +
        landing_bonus +
        fuel_penalty
    )

    components = {
        "approach_reward": approach_reward,
        "proximity_warmth": proximity_warmth,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "boundary_penalty": boundary_penalty,
        "landing_bonus": landing_bonus,
        "fuel_penalty": fuel_penalty,
    }

    return float(total_reward), components
```
