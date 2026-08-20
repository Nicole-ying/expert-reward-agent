现在我对整个迭代历史有了清晰的了解。让我整理最终诊断。

---

## 诊断报告

### Evidence

| 来源 | 关键数据 |
|---|---|
| **跨迭代演化** | v1(-2.98)→v2(67.23)→v3(128.44)→**v4(135.22, BEST)**→v5(-113.47)→v6(-24.58)→v7(-116.44) |
| **v4 成功画像** | score=135.22, len=285, 19/20 terminated, soft_landing_bonus=161.4 (76.2% mag), proximity=44.4 (22.5%) — 着陆奖励主导但健康 |
| **v7 当前画像** | score=-116.44, len=68, 20/20 early terminated, soft_landing_bonus=1.5 (2.9% mag, active 1.8%), proximity=44.7 (**91.6% mag**) — 着陆奖励几乎为零 |
| **soft_landing_bonus 历史** | v1:8.0→v2:300.6→v3:535.4→v4:161.4→**v5:1.0→v6:0.0→v7:1.5** — v5之后断崖式崩溃 |
| **proximity_progress 历史** | v2:32.3→v3:32.6→v4:44.4→**v5:55.9→v6:45.1→v7:44.7** — v5之后一直维持在44+ |
| **v7 checkpoint** | score 全程平坦在 -115~-120，所有组件在所有 CP 几乎不变 — **训练从未起步** |

### Behavior Diagnosis

**当前策略：高速俯冲坠毁（crash diving）。**

v7 的策略从训练一开始就锁死在 crash diving：proximity_progress 在所有 checkpoint 都恒定在 ~44.6，soft_landing_bonus 始终 ~1.0-1.5（active 仅 1-2%）。agent 学会了一条捷径——关闭所有引擎（fuel_penalty 惩罚引擎使用），让重力自由下落，最大化 proximity_progress（每步约 0.64 的欧氏距离缩减），在 ~68 步后因高速撞击地面而提前终止。从未探索到软着陆。

**根因分析**：v7 在 v4 骨架（已验证可行）上添加了 `fuel_penalty=-0.15`（对 action≠0）。这个惩罚虽然微小（magnitude_share 仅 0.9%），但它改变了早期探索的梯度方向：agent 在探索阶段发现"不用引擎 = 没惩罚 + 重力白送 proximity_progress"，这条捷径的 reward 远高于任何需要引擎参与的着陆尝试。一旦这条捷径在早期被强化，agent 永远跳不出这个局部最优。

**教训**：v4→v5（改系数比例）崩溃，v4→v6（加速度惩罚）崩溃，v4→v7（加燃料惩罚）崩溃。v4 骨架对任何附加惩罚都极其敏感。修复策略应该是：**回到 v4 骨架，仅做系数微调**。

### 修改方案

**L1 — 系数调整。** 回到 v4 骨架（删除 fuel_penalty），将 `proximity_progress` 从 40.0 降至 30.0。

理由：
- v4 的 proximity=40 配合 continuous landing bonus=4.0 能工作，但 proximity 偏高（一度在 v5/v7 中驱动 crash diving）
- v2/v3 使用 proximity=30 已验证可稳定引导接近目标（score 67→128）
- 将 proximity 降回 30 可以降低"俯冲捷径"的相对吸引力，同时保持足够的接近引导
- soft_landing_bonus=4.0（continuous）保持不变 — 这是 v4 成功的关键信号
- 删除 fuel_penalty — 它破坏了早期探索

---

## 修改后代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v8 — Rollback to v4 skeleton, reduce proximity_progress 40→30.

    Diagnosis from v5-v7 history:
    - v4 (135.22, 19/20 landed) is the only working framework. Its skeleton:
      continuous per-step soft_landing_bonus + moderate proximity_progress.
    - v5 (proximity↑50, landing↓2.5) → crash diving: proximity dominates gradient.
    - v6 (+velocity_penalty) → permanent hovering: global speed penalty kills descent.
    - v7 (+fuel_penalty on v4) → crash diving: even tiny fuel penalty makes "no engine"
      the dominant early strategy, agent never explores soft landing.

    Single semantic change (L1):
    - Remove fuel_penalty (back to v4 skeleton).
    - Reduce proximity_progress 40.0 → 30.0 (v2/v3 level, proven stable).
      This reduces the crash-diving gradient while keeping approach guidance intact.
    - soft_landing_bonus kept at 4.0 (continuous per-step) — the key signal that
      made v4 achieve 19/20 landings.
    - attitude_penalty unchanged.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: continuous per-step reward for safe contact ----------
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 4.0 * both_contact * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```