现在我已掌握完整信息。让我综合所有数据进行诊断。

---

## 诊断报告

### 1. Evidence

| 来源 | 关键数据 |
|---|---|
| **跨迭代演化** | iter1(-2.98, len=813) → iter2(67.23, len=561) → iter3(128.44, len=574) → iter4(135.22, len=285)。v1→v2 引入 transition-based landing bonus，分数大涨但被 exploit；v3→v4 改为连续 per-step，长度减半但训练剧烈震荡 |
| **iter4 checkpoint score 趋势** | cp10%=-63.6 → cp40%=-6.4 → cp60%=97.4 → cp70%=-32.6 → cp80%=-79.8 → cp90%=157.9 → cp100%=89.6。**剧烈震荡**，五次穿越零轴，说明策略不稳定 |
| **iter4 soft_landing_bonus 趋势** | cp10%=1.4 → cp40%=126.7 → cp50%=1.7 → cp60%=278.9 → cp70%=17.5 → cp80%=14.1 → cp90%=87.2 → cp100%=431.5。大幅摆动，agent 反复"找到→失去→再找到"landing bonus |
| **iter4 final eval** | soft_landing_bonus magnitude_share=**76.2%**（仍主导），active_rate=23.5%；proximity_progress=22.5%，active=97.4%（稳定但占比低） |
| **iter3 vs iter4 对比** | iter3 soft_landing cp60%=**1592**（one-shot 爆炸），iter4 cp100%=431.5（连续版有所收敛但仍波动大）。连续化改善了爆炸但未解决震荡 |
| **score_range** | [-43.26, 287.31]，max 已超 200 目标，但 mean=135。部分 episode 完美（287），部分崩盘（-43），**方差是主要拖累** |

### 2. Behavior Diagnosis

当前策略存在「悬崖式奖励地形」导致的不稳定学习：

- **机制**：soft_landing_bonus 使用三个因子**相乘**（both_contact × velocity_quality × angle_quality），形成 AND 逻辑。当 agent 接触但姿态稍偏（如 angle=0.5rad），angle_quality=0.286，奖励被压缩到完美的 1/3.5。这种陡峭梯度使 agent 在「刚好完美」和「差一点就几乎为零」之间剧烈摆动。
- **后果**：agent 在训练中反复经历「找到完美着陆→获得高奖励→策略过拟合→稍微偏离→奖励崩塌→策略退化→重新探索」的循环，checkpoint score 五次穿越零轴就是直接证据。
- **根因**：quality gates 温度参数 5.0 过于严厉 + 系数 4.0 使 landing bonus 占比 76%，导致奖励地形过于尖锐。
- **gap 分析**：目标 200，当前 135，差距 65。不是方向错误（方向对，score 持续上升），而是**稳定性不足**导致 mean 被少数崩盘 episode 拉低。max 已达 287 说明最佳 episode 已超目标。

### 3. 干预方案 — L2

**单一语义修改**：软化 landing quality gates（温度 5→3）+ 降低 landing 系数（4.0→2.5）+ 提升 proximity 系数（40→50）以重新平衡。

- 温度降低使 quality 函数从陡峭变为平缓，在 angle=0.5 时 quality 从 0.286 升至 0.40，保持梯度但不制造悬崖
- 系数降低使 landing bonus 从 dominant（76%）降至与 proximity 更平衡的水平
- proximity 增强补偿 landing 减弱，维持总奖励量级，同时加强过程引导

这保持了乘法 AND 逻辑（接触+低速+水平缺一不可的正确语义），只改变了「多严厉地惩罚不完美」。

---

## 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v5 — Softens landing quality gates and rebalances components.

    Changes from v4:
    - soft_landing_bonus: quality gate temperature 5.0 → 3.0 (softer gradient).
      The multiplicative AND-logic is preserved (contact + low speed + level
      attitude all required), but the penalty for imperfect conditions is less
      severe, reducing the "reward cliff" that caused training oscillation.
    - soft_landing_bonus coefficient: 4.0 → 2.5 to reduce dominance (was 76%
      magnitude share), shifting balance toward process-oriented guidance.
    - proximity_progress: 40.0 → 50.0 to strengthen approach signal now that
      landing bonus is less dominant. This provides more consistent per-step
      guidance throughout the descent.
    - attitude_penalty unchanged.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 50.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: continuous per-step reward for safe contact ----------
    # Only when both legs are firmly on the pad (binary gate)
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # Quality gates — temperature reduced 5.0→3.0 for softer gradient
    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 3.0 * v_mag)       # → 1.0 at v=0, → 0.25 at v=1
    angle_quality    = 1.0 / (1.0 + 3.0 * abs(angle))  # → 1.0 at θ=0, → 0.25 at |θ|=1

    # Coefficient 2.5 (was 4.0) — softer max per-step, reduced dominance
    soft_landing_bonus = 2.5 * both_contact * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```