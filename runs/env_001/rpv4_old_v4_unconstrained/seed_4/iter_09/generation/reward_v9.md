Now I have a comprehensive picture. Let me compile the full diagnosis.

---

## 诊断报告

### evidence

| 来源 | 关键数据 | 解读 |
|---|---|---|
| **iter_6 cp trend** | cp10%: landing_bonus=282.9 (active 11.1%), cp20%→cp100%: landing_bonus=0 | agent 早期随机探索时误入着陆区获得巨量奖励，但训练后**主动抛弃着陆**，因为 proximity_warmth 从 22 涨到 45 — agent 发现悬停收租更划算 |
| **iter_6 proximity_warmth** | ep_sum=45.1 (45.7% mag), 100% active | 这是纯粹的「租金源」— agent 只需靠近目标即可稳定获得正奖励，无需着陆 |
| **iter_7 contact_reward** | ep_sum=0, active=0% | 门控过紧（dist<0.3 AND speed<0.6），agent 从未触发 |
| **iter_8 checkpoint** | cp40%: landing_readiness=3.5 (active 26.9%), cp80%: 4.6 (37.5%), cp100%: **0 (0%)** | agent 在训练中段探索了着陆准备姿态，但**最终完全抛弃** — 因为 approach_reward(16.7) + proximity_warmth(10.6) = +27.3 稳定收入 > 着陆探索的风险 |
| **iter_8 contact_reward** | ep_sum=0, active=0% | 连续两轮 contact_reward 完全死亡 — 着陆信号不可达 |
| **iter_8 final score** | -45.74, len=1000, truncated=20/20 | 所有 episode 跑满 1000 步未终止 — agent 从未成功着陆 |
| **全部 8 轮** | 最佳 score=-5.35 (iter_6)，但该轮也是 20/20 truncated | 没有任何一轮实现过真正的软着陆 |
| **iter_8 fuel_penalty** | -19.6, active 65.4% | agent 使用引擎频率高（维持悬停位置），但从未用于着陆 |

### behavior_diagnosis

当前策略学到的是一个**「悬停寄生」行为**：

1. agent 通过 approach_reward 被驱动靠近目标
2. 到达中等距离（dist≈0.5-1.0）后，proximity_warmth 提供 ~0.01/step 的稳定正向收入
3. approach_reward 在振荡中净贡献 ≈ 0（来回摆动）
4. 着陆组件（contact_reward、landing_readiness、perfect_landing）全部为 0 — agent **从未着陆**
5. 燃料消耗适中（65% active），主要用于维持悬停位置
6. 所有 episode 跑满 1000 步截断 — 机体从未自然稳定

**核心矛盾**：proximity_warmth（以及之前 iter_6 更强的版本）创建了一个稳定的局部最优——在目标附近悬停比冒着 crash 风险着陆更"划算"。agent 在 iter_8 cp40% 和 cp80% 曾短暂探索过着陆条件（landing_readiness 活跃），但每次都**主动退回悬停策略**。这不是信号不可达的问题，而是**悬停的性价比高于着陆**。

**根本原因**：只要存在任何形式的「仅依赖距离」的正向状态奖励（proximity_warmth），agent 就能在中距离找到租金均衡点。着陆奖励虽然量级更大但门控条件使其难以稳定触发，agent 理性选择了更容易的悬停策略。

### 干预策略

三个关键改变，缺一不可：

1. **彻底消灭租金源**：删除 proximity_warmth。替换为 `landing_potential`——一个同时要求「近 + 慢 + 正姿态」的连续乘积奖励。单纯靠近目标不再有奖励，必须同时减速和调姿。

2. **拆除着陆接触的速度门控**：contact_reward 仅需 `dist<0.6`，不要求低速。任何距离内的腿接触都获得奖励。让 agent 先学会「接触是好事」，再通过 landing_potential 引导它慢下来。

3. **着陆激励必须压倒悬停**：contact_reward 的每步量级（~5-16/step）必须远超 landing_potential 的最佳悬停值（~0.15/step），比例 > 30:1。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化 v9 — 消灭租金源，着陆优先。

    核心诊断（iter_6/7/8）：
    - proximity_warmth 是租金源：agent 在中距离悬停收租，永不降落
    - 着陆奖励门控过紧（速度门）：agent 无法稳定触发，探索后退回悬停
    - iter_8 cp80% landing_readiness 活跃但 cp100% 归零 → agent 主动抛弃着陆

    v9 三管齐下：
    1. 删除 proximity_warmth，替换为 landing_potential（近×慢×正=三位一体）
       → 单纯靠近不再有奖励，必须同时减速调姿
    2. 拆除 contact_reward 的速度门控 → dist<0.6 即有接触奖励
       → agent 只需被 approach_reward 带到近处，随机探索即可触发接触
    3. 接触奖励 magnitude 压倒悬停 → 比例 > 30:1
       → 一旦发现接触，agent 有强烈动机复制

    梯度链：
      远距 → approach_reward（靠近）
      近距 → landing_potential（靠近+减速+调姿，连续梯度）
      接触 → contact_reward（巨大跳跃，无速度门槛）
      完美 → perfect_landing（精炼）
    """

    # ================================================================
    # 信号提取
    # ================================================================
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # 派生量
    current_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5

    # ================================================================
    # 1. approach_reward — 路径无关 backbone（不可 exploit）
    #
    #    总 episode 贡献 = 20 * (initial_dist - final_dist)
    #    靠近→正，远离→负，悬停振荡→净≈0
    # ================================================================
    dist_improvement = current_dist - next_dist
    approach_reward = 20.0 * dist_improvement

    # ================================================================
    # 2. landing_potential — 「近×慢×正」三位一体梯度（替代 proximity_warmth）
    #
    #    关键区别 vs proximity_warmth：
    #    - proximity_warmth：仅依赖距离 → 悬停即可收租
    #    - landing_potential：距离×速度×角度三因子乘积 → 必须三者同时优化
    #
    #    在 dist=0.5, speed=0.3, angle=0.1 悬停：
    #      landing_potential = 0.25 * 0.667 * 0.7 * 0.8 = 0.093/step
    #    在 dist=0.2, speed=0.1, angle=0.05 准备着陆：
    #      landing_potential = 0.25 * 0.867 * 0.9 * 0.9 = 0.176/step
    #    梯度清晰：越接近着陆条件，奖励越高。但量级受控（<0.25/step），
    #    远小于 contact_reward，不会成为新的租金源。
    # ================================================================
    landing_potential = 0.0
    if next_dist < 1.5:
        d_factor = max(0.0, 1.0 - next_dist / 1.5)       # dist=0→1, dist=1.5→0
        s_factor = max(0.0, 1.0 - next_speed / 1.0)      # speed=0→1, speed=1→0
        a_factor = max(0.0, 1.0 - abs(n_angle) / 0.5)    # angle=0→1, |angle|=0.5→0
        landing_potential = 0.25 * d_factor * s_factor * a_factor

    # ================================================================
    # 3. contact_reward — 拆除速度门控！仅需距离门控
    #
    #    iter_7/8 的核心教训：速度门控（speed<0.6 或 quality_mult 含速度因子）
    #    使 agent 在快速接近时即使腿触地也得不到奖励，无法建立「接触=好」的关联。
    #
    #    v9：只要 dist<0.6 且有腿接触，就给奖励。不要求低速。
    #    高速俯冲单腿擦碰 → 也有信号（虽然小），agent 能感知到。
    #    然后 landing_potential 会引导它下次更慢地接近。
    #
    #    典型值：
    #    - dist=0.4, 单腿: 1*5*(1-0.4/0.6) = 1.67/step
    #    - dist=0.2, 双腿: 2*5*0.667 + 12*0.667 = 6.67 + 8.0 = 14.67/step
    #    - dist=0.05, 双腿: 2*5*0.917 + 12*0.917 = 9.17 + 11.0 = 20.17/step
    #
    #    vs landing_potential 最佳悬停 ~0.1/step → 比例 17-200x，压倒性优势。
    # ================================================================
    contact_reward = 0.0
    if next_dist < 0.6:
        d_quality = 1.0 - next_dist / 0.6   # 1 at dist=0
        contact_count = n_left + n_right      # 0.0, 1.0, or 2.0

        if contact_count > 0:
            # 每腿基础奖励 × 距离质量
            contact_reward = contact_count * 5.0 * d_quality

            # 双腿同时接触 → 大幅叠加（发现「正确姿势」的关键信号）
            if contact_count >= 2.0:
                contact_reward += 12.0 * d_quality

    # ================================================================
    # 4. perfect_landing — 完美着陆精炼信号
    #
    #    只有当 agent 已经学会着陆后，这个组件才开始生效，
    #    引导从「粗糙着陆」→「完美软着陆」的精炼。
    #    条件严格但非门控式——乘积形式保留部分梯度。
    # ================================================================
    perfect_landing = 0.0
    if n_left > 0.5 and n_right > 0.5:
        # 双腿已接触，根据速度和姿态质量给予额外奖励
        speed_quality = max(0.0, 1.0 - next_speed / 0.15)
        angle_quality = max(0.0, 1.0 - abs(n_angle) / 0.15)
        dist_quality = max(0.0, 1.0 - next_dist / 0.12)
        perfect_landing = 15.0 * speed_quality * angle_quality * dist_quality

    # ================================================================
    # 5. velocity_penalty — 轻量距离门控（辅助信号）
    #
    #    保持温和，不喧宾夺主。landing_potential 已承担主要的速度引导。
    # ================================================================
    vel_gate = 1.0 / (1.0 + 3.0 * next_dist)
    velocity_penalty = -0.3 * vel_gate * (nvx**2 + nvy**2)

    # ================================================================
    # 6. angle_penalty — 轻量距离门控（辅助信号）
    # ================================================================
    angle_gate = 1.0 / (1.0 + 3.0 * next_dist)
    angle_penalty = -angle_gate * (0.5 * n_angle**2 + 0.2 * n_ang_vel**2)

    # ================================================================
    # 7. fuel_penalty — 轻量引擎使用惩罚
    #
    #    保持 -0.03。着陆后 agent 自然不需要引擎，省油自动达成。
    # ================================================================
    fuel_penalty = -0.03 * float(action != 0)

    # ================================================================
    # 8. boundary_penalty — 水平越界安全网
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
        landing_potential +
        contact_reward +
        perfect_landing +
        velocity_penalty +
        angle_penalty +
        fuel_penalty +
        boundary_penalty
    )

    components = {
        "approach_reward": approach_reward,
        "landing_potential": landing_potential,
        "contact_reward": contact_reward,
        "perfect_landing": perfect_landing,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "fuel_penalty": fuel_penalty,
        "boundary_penalty": boundary_penalty,
    }

    return float(total_reward), components
```