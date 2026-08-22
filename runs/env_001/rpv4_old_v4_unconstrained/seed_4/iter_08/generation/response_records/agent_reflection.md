# Response Record

现在我对全局有了全面了解。让我进行完整的诊断分析。

---

## 诊断报告

### evidence

| 来源 | 关键数据 | 解读 |
|---|---|---|
| **iter_3** | `goal_proximity` ep_sum=**+1204** (86% mag), len=1000, score=-65 | `3.0*e^(-dist)` 指数 proximity bonus 被 exploit：agent 悬停在中等距离收租，永不降落 |
| **iter_4** | score=-109, len=68.5 | 试图用 potential-based shaping 修复，agent 直接崩溃——修复过度 |
| **iter_5** | `approach_reward`=-38.6 (95% mag), len=68.8, terminated=20/20 | 纯 delta 奖励：agent 学到的策略是**远离目标**（dist 增大→负 approach_reward），全部 crash |
| **iter_6** | score=**-5.35**, len=1000, truncated=20/20, `proximity_warmth`=+45.1 (45.7%), `landing_bonus`=**0.0** | 最佳得分但 agent 从未着陆——微小 proximity (0.05*(1-dist/3)) 提供了足够"租金"让 agent 活满 1000 步，但 landing_bonus（乘法四因子乘积）根本无法发现 |
| **iter_6 cp trend** | cp10%: landing_bonus=**282.9** (11.1% active) → cp100%: **0.0** | agent 早期随机探索时误入着陆区获得大量奖励，训练后期**主动抛弃了着陆**——因为 proximity_warmth 提供稳定"租金"，着陆风险高且信号太窄 |
| **iter_7** | score=**-115.61**, len=948, terminated=8/20, `contact_reward`=0.0, `close_proximity`=+4.31 | 移除 proximity_warmth 后 agent 失去引导，大量 crash。contact_reward 虽改为加法但门控条件（dist<0.3 AND speed<0.6 AND contact）仍然太窄，最终 policy 完全无法触及 |
| **iter_7 cp trend** | cp10%: contact_reward=**543.7** (6.5% active) → cp20-100%: 大多为 0 | 与 iter_6 相同的模式：早期随机撞入，后期策略收敛后反而丢失 |
| **fuel_penalty 趋势** | iter_6→iter_7: -27.1→-28.5, active 90%→100% | fuel_penalty 在整个训练中稳步增长且始终很高，agent 被强制大量使用引擎却无对应的正向引导 |

### behavior_diagnosis

**核心病理：「租金陷阱 + 着陆信号不可达」循环。**

整个 7 轮迭代暴露了一个结构性矛盾：

1. **没有 proximity 引导 → agent 崩溃**（iter_4, iter_5, iter_7）。纯 approach_reward（delta-based）不足以引导 agent 在稀疏奖励空间中学会靠近目标垫板。
2. **有 proximity 引导 → agent 悬停收租**（iter_3, iter_6）。任何正的状态奖励都会被 exploit：agent 学会在中等距离保持低速悬停，收集稳定"租金"直到 truncation。
3. **着陆信号始终无法在最终 policy 中存活**：iter_6 乘法 landing_bonus 太窄（四因子乘积），iter_7 加法 contact_reward 门控太严（dist<0.3 AND speed<0.6 AND contact）。两者都在 cp10% 被随机探索到，但 cp100% 完全消失。

**iter_7 的具体行为**：terminated=8/20（35% crash），len=948。agent 失去了 iter_6 的 proximity 引导后，approach_reward 成了唯一正信号（+12.05），fuel_penalty（-28.45）和 velocity_penalty（-9.04）压倒性地占据负向。agent 学到的策略是随机移动、偶尔靠近目标获得微弱的 approach_reward，但无法稳定靠近，更无法着陆。8/20 的 terminated 说明 agent 频繁飞出视口或 crash。

**根本原因**：着陆奖励需要 agent 同时满足多个条件（近 + 慢 + 正 + 接触），而"悬停收租"只需满足一个条件（近）。在 RL 探索中，后者必然先被发现。一旦发现，agent 的策略梯度就锁死在"悬停"局部最优，永远不会探索到需要精确控制的着陆。

### 干预策略

**Level 4：结构性重构**。不是参数微调，而是重写奖励的梯度结构。

核心思想：**「漏斗式着陆梯度」**——从远到近创建平滑、无局部最优的奖励梯度，让 agent 像滑滑梯一样自然滑向着陆：

1. **极小 proximity 引导**（0.015 系数，从 iter_6 的 0.05 削减 70%）——仅够维持"靠近目标"的基本方向，不足以成为租金来源
2. **「着陆就绪」预奖励**——当 agent 进入极近距（dist<0.2）且低速（speed<0.35）时给微小正向信号，搭建从"靠近"到"接触"的桥梁
3. **加法接触奖励，极大放宽初始条件**——只要 dist<0.5 且有任意腿接触就给奖励，质量因子作为**乘数而非门控**（最低保留 30% 基础值），确保高速粗接触也有信号
4. **完美着陆叠加**——双腿 + 极低速 + 姿态正 → 每步大幅额外奖励

关键比率：着陆奖励（~4-12/step）>> proximity 引导（~0.01/step），约 **400-1200 倍**差距，agent 不可能满足于悬停。

---

## 奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化 v8 — 漏斗式着陆梯度。

    核心创新：消除「租金陷阱 + 着陆信号不可达」的矛盾。
    
    设计原则：
    - 不存在可被 exploit 的中等距离租金源（iter_3/iter_6 的教训）
    - 着陆奖励从粗到精连续可微分（iter_6/iter_7 的教训：门控太窄）
    - 微小 proximity 仅维持靠近目标的动力，但量级远不足以支撑悬停策略
    - 「着陆就绪」预奖励桥接 proximity 和 contact 之间的梯度断层
    
    梯度漏斗结构：
      远距 → approach_reward (delta, 路径无关, 不可 exploit)
      中距 → 极微 proximity_warmth (仅方向指引, ~0.01/step)
      近距 → landing_readiness (dist<0.2, slow, upright → ~0.05/step)
      接触 → contact_reward (additive, 宽松初始条件 → ~3-8/step)
      完美 → perfect_landing (双腿+极低速+正姿 → +15/step 叠加)
      
    关键比率：contact_reward / proximity_warmth ≈ 400-1200x
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
    # 1. approach_reward — 路径无关 delta 主干（不可 exploit）
    #
    #    总 episode 贡献 = 20 * (initial_dist - final_dist)，完全路径无关。
    #    靠近目标 → 正，远离 → 负，悬停振荡 → 净≈0。
    #    从 iter_6 的 15 提升到 20，使靠近目标的收益更显著。
    # ================================================================
    dist_improvement = current_dist - next_dist
    approach_reward = 20.0 * dist_improvement

    # ================================================================
    # 2. proximity_warmth — 极微方向指引（不可 exploit）
    #
    #    从 iter_6 的 0.05*(1-dist/3) 大幅削减：
    #    - 系数：0.05 → 0.015（削减 70%）
    #    - 范围：3.0 → 2.0（收缩三分之一）
    #    在 dist=0.5 处：0.015*0.75=0.011/step → 1000 步 ≈ 11 点
    #    vs iter_6 在 dist=0.5 处：0.05*0.833=0.042/step → 1000 步 ≈ 42 点
    #    仅够提供"朝目标走"的微弱动机，不可能成为租金来源。
    # ================================================================
    proximity_warmth = 0.015 * max(0.0, 1.0 - next_dist / 2.0)

    # ================================================================
    # 3. landing_readiness — 「着陆就绪」预奖励
    #
    #    解决 iter_6/iter_7 中 "proximity→contact 梯度断层" 问题。
    #    当 agent 进入极近距（dist<0.2）、低速（speed<0.35）、姿态不极端时，
    #    给予微小正向信号，引导 agent 进入"预备着陆"姿态。
    #    这个奖励非常小（max ~0.06/step），但创建了从"靠近"到"接触"
    #    之间的连续梯度，使 agent 不会在 proximity 区域停滞。
    # ================================================================
    landing_readiness = 0.0
    if next_dist < 0.2 and next_speed < 0.35:
        d_factor = 1.0 - next_dist / 0.2       # 1 at dist=0
        s_factor = 1.0 - next_speed / 0.35     # 1 at speed=0
        a_factor = max(0.0, 1.0 - abs(n_angle) / 0.3)  # 1 at angle=0
        landing_readiness = 0.06 * d_factor * s_factor * a_factor
        # 典型值 dist=0.1, speed=0.15, angle=0.1:
        #   0.06 * 0.5 * 0.571 * 0.667 = 0.011/step — 极微，纯梯度信号

    # ================================================================
    # 4. contact_reward — 加法独立着陆奖励（核心创新）
    #
    #    对比 iter_7 contact_reward 的关键改进：
    #    (a) 放宽距离门控：0.5 vs 0.3
    #    (b) 移除速度硬门控：不再要求 speed<0.6，改为质量乘数
    #    (c) 质量因子作为乘数而非门控：保留 30% 基础值，高速粗接触也有信号
    #    (d) 每条腿独立奖励 + 叠加
    #
    #    设计保证：即使在 dist=0.4, speed=0.8（高速俯冲）, angle=0.2 时，
    #    单腿接触仍能获得：1*6.0*0.2*(0.3+0.7*0*0.5) = 0.36/step
    #    虽小但非零——agent 能感知到"接触是好事"。
    #    随着条件改善，奖励平滑增长到 ~8-15/step。
    # ================================================================
    contact_reward = 0.0
    if next_dist < 0.5:
        d_quality = 1.0 - next_dist / 0.5            # 1 at dist=0
        s_quality = max(0.0, 1.0 - next_speed / 0.6) # 1 at speed=0
        a_quality = max(0.0, 1.0 - abs(n_angle) / 0.4)  # 1 at angle=0
        
        contact_count = n_left + n_right  # 0.0, 1.0, or 2.0
        
        if contact_count > 0:
            # 每腿基础奖励 × 质量乘数（最低保底 30%）
            quality_mult = 0.3 + 0.7 * s_quality * a_quality
            contact_reward = contact_count * 6.0 * d_quality * quality_mult
            
            # 双腿稳定接触叠加奖励
            if contact_count >= 2.0:
                contact_reward += 8.0 * d_quality * s_quality * a_quality

    # ================================================================
    # 5. perfect_landing — 完美着陆超级奖励
    #
    #    当所有条件完美满足时叠加：双腿接触 + 极低速 + 极正姿态 + 极近。
    #    这是一个"锦上添花"的信号，帮助 agent 从"粗糙着陆"
    #    精炼到"完美软着陆"。
    # ================================================================
    perfect_landing = 0.0
    if (n_left > 0.5 and n_right > 0.5 
            and next_speed < 0.08 
            and abs(n_angle) < 0.08 
            and next_dist < 0.1):
        perfect_landing = 15.0

    # ================================================================
    # 6. velocity_penalty — 距离门控速度抑制
    #
    #    gate = 1/(1+3*dist)：dist=0→1.0, dist=0.5→0.4, dist=2→0.14
    #    远处几乎不干预，近处温和引导减速。
    #    系数 0.5 适度，不会像 iter_7 的 0.8 那样强势。
    # ================================================================
    vel_gate = 1.0 / (1.0 + 3.0 * next_dist)
    velocity_penalty = -0.5 * vel_gate * (nvx**2 + nvy**2)

    # ================================================================
    # 7. angle_penalty — 距离门控姿态约束
    #
    #    与 velocity_penalty 使用相同的门控逻辑。
    #    系数温和，避免阻碍中距离机动。
    # ================================================================
    angle_gate = 1.0 / (1.0 + 3.0 * next_dist)
    angle_penalty = -angle_gate * (1.0 * n_angle**2 + 0.5 * n_ang_vel**2)

    # ================================================================
    # 8. fuel_penalty — 引擎使用惩罚
    #
    #    保持 0.03 不变，与 iter_6/iter_7 一致。
    #    足够轻量以不阻碍必要探索。
    # ================================================================
    fuel_penalty = -0.03 * float(action != 0)

    # ================================================================
    # 9. boundary_penalty — 水平越界安全网
    #
    #    保持 iter_7 的参数不变（阈值 1.3，二次惩罚）。
    #    历史数据显示 boundary_penalty 始终为 0，agent 已经学会不越界。
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
        proximity_warmth +
        landing_readiness +
        contact_reward +
        perfect_landing +
        velocity_penalty +
        angle_penalty +
        fuel_penalty +
        boundary_penalty
    )

    components = {
        "approach_reward": approach_reward,
        "proximity_warmth": proximity_warmth,
        "landing_readiness": landing_readiness,
        "contact_reward": contact_reward,
        "perfect_landing": perfect_landing,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "fuel_penalty": fuel_penalty,
        "boundary_penalty": boundary_penalty,
    }

    return float(total_reward), components
```
