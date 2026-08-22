## 诊断报告

### evidence

| 来源 | 关键数据 | 解读 |
|---|---|---|
| **iter_6 score/len** | score=-5.35, len=1000, truncated=20/20 | agent 活满整集但从未自然终止——从未成功着陆 |
| **iter_6 cp trend** | cp10%: landing_bonus=282.9 (active 11.1%), approach_reward=+8.87 → cp100%: landing_bonus=0 (0%), approach_reward=+19.87 | agent 早期随机探索时误入着陆区获得大量奖励，但训练后主动抛弃了着陆，选择了更"划算"的策略 |
| **iter_6 proximity_warmth** | ep_sum=45.13, magnitude_share=**45.7%** | 这是租金来源：0.05×(1−dist/3) 在 dist=1 处每步约 0.033，1000步≈33，配合燃料节省构成局部最优 |
| **iter_6 fuel_penalty** | ep_sum=-27.12 (27.5%), active=90.4%, cp趋势从-13→-27 | agent 学会大量减少引擎使用来避免惩罚，加剧了"悬停收租"策略 |
| **iter_3 exploitation** | goal_proximity=+1204 (86% share), len=1000 | iter_3 的指数 proximity bonus 在 dist≈1 时每步 +1.1，1000步=1100，相同模式：agent 在中等距离悬停收租，从不降落 |
| **iter_6 landing_bonus** | ep_sum=0.0, active=0%, 全程为0 | 乘法形式的着陆奖励（四个连续因子乘积×12）条件过严，agent 无法通过探索发现——即使靠近目标，任一因子不满足就完全为零 |
| **iter_4 landing_contact** | ep_sum=10.5, active=0.8% (len=68.5) | 二元门控着陆奖励在短 episode 中可访问，但乘法连续形式（iter_6）变成了死组件 |

### behavior_diagnosis

**核心问题："生存收租"取代"着陆完成任务"。**

iter_6 的 agent 学到了一套精致的生存策略：保持在距目标垫板约 1–2 单位的位置（处于 proximity_warmth 的收租区内，dist<3），尽量减少引擎使用以节省燃料惩罚，同时通过微小的周期性靠近获得少量 approach_reward。这套策略每 episode 净赚约 +35 奖励（45 proximity + 20 approach − 27 fuel），远高于尝试着陆的风险收益（着陆信号始终为零，无法被发现）。

这是一个典型的 **reward exploitation cycle**：iter_3 被指数 proximity 租金 exploit → 切换到 delta 为主的 approach（iter_4-5）→ 添加 warmth 解决冷启动 → warmth 又被 exploit（iter_6）。**只要存在"不依赖着陆就能稳定收入"的正奖励源，agent 就会找到并 exploit 它。**

**干预方向**：
1. **彻底移除 proximity_warmth**——它是租金的根源。用仅在极近距离（<0.4）才激活的微小梯度替代。
2. **着陆信号必须可发现**——放弃乘法形式，改用加法独立组件，每个腿部接触都独立给予奖励，降低发现门槛。
3. **approach_delta 是唯一正确的 backbone**——路径无关的势能差，不可 exploit，应成为最大信号源。
4. **确保"什么都不做"也不划算**——移除 warmth 后，不靠近目标就不再有正奖励，逼迫 agent 主动接近。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化 v7 — 根治租金 exploit，重构着陆信号。

    核心设计哲学：
    - 移除所有可在中途"收租"的状态奖励（proximity_warmth 是 iter_6 的罪魁祸首）
    - approach_reward（路径无关势能差）是唯一 backbone 信号，不可 exploit
    - 着陆奖励改为加法独立组件：每条腿独立奖励 + 双腿额外奖励 + 完美着陆大奖
      降低发现门槛，确保 agent 在探索中能偶然触及
    - 极近距离的微弱梯度（<0.4）仅用于在目标附近提供额外方向指引
    - velocity/angle 惩罚在远处几乎为零，近处温和引导减速和姿态
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
    # 1. approach_reward — 路径无关势能差（backbone，不可 exploit）
    #
    #    reward = 20.0 * (current_dist - next_dist)
    #    靠近目标 → 正，远离 → 负，悬停/振荡 → 净≈0
    #    总 episode 贡献 = 20 * (initial_dist - final_dist)，完全路径无关
    #    比 iter_6 的 15 提升到 20，使接近目标的收益更显著
    # ================================================================
    dist_improvement = current_dist - next_dist
    approach_reward = 20.0 * dist_improvement

    # ================================================================
    # 2. close_proximity — 极近距离微弱梯度（仅 < 0.4 单位）
    #
    #    二次衰减：dist=0 → 0.15, dist=0.2 → 0.0375, dist=0.4 → 0
    #    即使 agent 在 dist=0.2 悬停 1000 步也只有 ~37 点——远小于
    #    着陆后 contact_reward 可达的 16-46 每步。
    #    唯一作用：在目标垫板附近提供额外的细粒度梯度。
    #    关键区别 vs iter_6 proximity_warmth：阈值为 0.4 而非 3.0，不会创建租金区。
    # ================================================================
    if next_dist < 0.4:
        close_proximity = 0.15 * (1.0 - next_dist / 0.4) ** 2
    else:
        close_proximity = 0.0

    # ================================================================
    # 3. velocity_penalty — 距离门控速度抑制
    #
    #    gate = 1/(1+2*dist)：dist=0→1.0, dist=1→0.33, dist=3→0.14
    #    远处几乎不干预，近处温和引导减速。
    #    系数 0.8 比 iter_6 的 0.3 略强，但仍在合理范围。
    # ================================================================
    vel_gate = 1.0 / (1.0 + 2.0 * next_dist)
    velocity_penalty = -0.8 * vel_gate * (nvx**2 + nvy**2)

    # ================================================================
    # 4. angle_penalty — 距离门控姿态约束
    #
    #    与 velocity_penalty 使用相同的门控逻辑。
    # ================================================================
    angle_gate = 1.0 / (1.0 + 2.0 * next_dist)
    angle_penalty = -angle_gate * (1.5 * n_angle**2 + 0.8 * n_ang_vel**2)

    # ================================================================
    # 5. contact_reward — 加法独立着陆奖励（核心创新）
    #
    #    关键设计决策：
    #    - 不再是 iter_6 的乘法形式（四个因子乘积，任一为零则全为零）
    #    - 每条支撑腿独立给予奖励，降低发现门槛
    #    - 条件适度宽松（dist<0.3, speed<0.6），确保 agent 在接近目标时
    #      有较高概率偶然获得首次接触奖励
    #    - 双腿同时接触 + 完美着陆有额外叠加奖励
    #
    #    奖励结构：
    #    - 左腿接触: +8
    #    - 右腿接触: +8
    #    - 双腿同时: 额外 +15  (总计 8+8+15=31)
    #    - 完美着陆: 额外 +30  (双腿+极低速+极正姿态, 总计 8+8+15+30=61)
    #
    #    iter_4 证明二元门控着陆信号可达（active 0.8%, ep_sum=10.5），
    #    本设计的加法独立性应进一步提升可发现性。
    # ================================================================
    contact_reward = 0.0
    near_pad = next_dist < 0.3
    slow_approach = next_speed < 0.6

    if near_pad and slow_approach:
        # 每条腿独立奖励
        if n_left > 0.5:
            contact_reward += 8.0
        if n_right > 0.5:
            contact_reward += 8.0
        # 双腿稳定接触额外奖励
        if n_left > 0.5 and n_right > 0.5:
            contact_reward += 15.0
        # 完美着陆大奖：双腿 + 极低速 + 姿态端正
        if (n_left > 0.5 and n_right > 0.5 and
                next_speed < 0.15 and abs(n_angle) < 0.15):
            contact_reward += 30.0

    # ================================================================
    # 6. fuel_penalty — 轻量引擎使用惩罚
    # ================================================================
    fuel_penalty = -0.03 * float(action != 0)

    # ================================================================
    # 7. boundary_penalty — 水平越界安全网
    #
    #    阈值 1.3，二次惩罚在越界时急剧增大。
    #    iter_6 中 boundary_penalty 全程为 0（agent 学会了不越界），
    #    保留作为安全网但不调整。
    # ================================================================
    safe_x_bound = 1.3
    if abs(nx) > safe_x_bound:
        boundary_penalty = -3.0 * (abs(nx) - safe_x_bound) ** 2
    else:
        boundary_penalty = 0.0

    # ================================================================
    # 汇总
    # ================================================================
    total_reward = (
        approach_reward +
        close_proximity +
        velocity_penalty +
        angle_penalty +
        contact_reward +
        fuel_penalty +
        boundary_penalty
    )

    components = {
        "approach_reward": approach_reward,
        "close_proximity": close_proximity,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "contact_reward": contact_reward,
        "fuel_penalty": fuel_penalty,
        "boundary_penalty": boundary_penalty,
    }

    return float(total_reward), components
```