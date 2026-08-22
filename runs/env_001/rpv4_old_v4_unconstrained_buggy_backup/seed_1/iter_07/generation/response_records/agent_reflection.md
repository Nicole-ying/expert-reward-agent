# Response Record

## 诊断报告

### Evidence

| 来源 | 关键数据 |
|---|---|
| **跨迭代演化** | v1(-2.98)→v2(67.23)→v3(128.44)→**v4(135.22, BEST)**→v5(-113.47)→v6(-24.58) |
| **v4 最终评估** | score=135.22, len=285, terminated=**19/20**, soft_landing_bonus=161.4 (76.2% mag), proximity=44.4 (22.5%) |
| **v4 checkpoint 趋势** | soft_landing_bonus: cp40%=126.7 → cp60%=278.9 → **cp100%=431.5** — 晚期被 exploit；score: cp90%=157.9 → cp100%=89.6，训练末期反而退化 |
| **v5 崩溃** | proximity_progress 50.0 + soft_landing coeff 2.5 → 全部 20 集早期终止(len=68)，高速俯冲撞击 |
| **v6 悬停** | velocity_penalty 导致 soft_landing_bonus **完全死亡**(0% active)，0/20 终止，1000 步全程悬停 |
| **soft_landing_bonus 历史** | v2:300.6(3.9% active, 85%) → v3:535.4(5.8%, 92%) → v4:161.4(23.5%, 76%) → v5:1.0(1.5%) → **v6:0.0(0%)** |
| **被拒草稿** | 仅是去掉 velocity_penalty 回到 v4，零新语义 |

### Behavior Diagnosis

v4 骨架是目前唯一可工作的框架（19/20 成功着陆，best=135.22），但它有两个明确问题：

1. **soft_landing_bonus 晚期 exploit**：cp100% 时 bonus 膨胀到 431.5，agent 学会了弹跳-重复着陆来刷分，导致最终 score 从 cp90% 的 157.9 退化到 89.6。
2. **缺少节油/快速停靠压力**：没有任何信号鼓励 agent 着陆后保持静止。弹跳需要引擎重新点火，但如果引擎使用没有代价，agent 就没有动力停止。

v5（加强 proximity）和 v6（加速度惩罚）都破坏了 v4 的脆弱平衡：v5 导致高速撞击，v6 导致永不休止的悬停。因此不能动 proximity_progress 或给速度加全局惩罚。

### 修改方案

**单一语义修改：引入 fuel_penalty（燃料效率惩罚）** — 对引擎动作施加小额惩罚。这是全新的组件，历史上从未出现过。

**机制**：
- action 0（无引擎）= 惩罚 0
- action 1/2/3（任意引擎点火）= 惩罚 -0.15/步
- 落地后 agent 只需 action 0 即可保持静止 → 天然鼓励快速停靠
- 弹跳需要引擎重新点火 → 弹跳 exploit 会被持续惩罚
- 节油是任务声明的次要目标 → 直接对齐

**为什么不是 L1 微调**：当前 best=135.22，距目标 200 差距 > 10%（差距 32%），且 v4 存在结构化缺陷（无燃料效率信号），需要 L2 新组件。

**预期效果**：v4 场景中 len=285、引擎使用率 ~50%，fuel_penalty 贡献约 -21，占总 magnitude 约 10%，与 attitude_penalty 和 proximity_progress 形成更均衡的四组件结构，同时抑制弹跳 exploit。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v7 — Adds fuel_penalty to v4 skeleton, encouraging efficient settling.

    Diagnosis from v4-v6 history:
    - v4 (135.22, 19/20 landed) is the only working framework, but soft_landing_bonus
      explodes late in training (cp100%=431.5). Agent bounces and re-lands to farm bonus.
    - v5 (+proximity, -landing) → high-speed crash diving (len=68, all terminated early).
    - v6 (+velocity_penalty) → permanent hovering (len=1000, 0/20 terminated,
      soft_landing_bonus 0% active). Global speed penalty kills landing incentive.
    - The v4 skeleton must be preserved; only v4 achieved reliable landings.

    Single semantic change: NEW fuel_penalty component.
    - Penalizes engine actions (1/2/3) with -0.15 per step.
    - Action 0 (no engine) costs nothing — gravity alone can descend.
    - Once landed and settled, agent can stay on action 0 indefinitely → natural settling incentive.
    - Bouncing requires engine re-fire → bouncing exploit is now taxed per re-fire.
    - Aligns with stated secondary objective: fuel efficiency.

    All other components unchanged from v4:
    - proximity_progress=40.0, soft_landing_bonus=4.0, attitude_penalty same.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 40.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: continuous per-step reward for safe contact ----------
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at theta=0, → 0.17 at |theta|=1

    soft_landing_bonus = 4.0 * both_contact * velocity_quality * angle_quality

    # ---------- fuel_penalty: discourage unnecessary engine use (NEW) ----------
    # action 0 = no engine (free), actions 1/2/3 = engine fire (cost -0.15)
    fuel_penalty = 0.0 if action == 0 else -0.15

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus + fuel_penalty

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "fuel_penalty": fuel_penalty
    }
    return float(total_reward), components
```
