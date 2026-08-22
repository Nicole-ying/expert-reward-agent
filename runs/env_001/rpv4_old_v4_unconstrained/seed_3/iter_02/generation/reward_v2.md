## 诊断报告

### evidence

**关键数据点：**
- Score = -32.35，全部 20 集 truncated（1000步），terminated=0——agent 从未着陆
- contact_bonus：active_rate=0%，ep_sum=0——完全死亡。cp10% 时曾短暂激活（5.162），随后立即归零
- goal_proximity 在所有 checkpoint 中几乎不变（3.38~3.96），说明 agent 停在某处不再移动
- orientation_penalty 与 goal_proximity 量级相等、符号相反（-3.96 vs +3.76），互相抵消
- velocity_penalty 极小（-0.49），对行为无实质影响
- score 在 cp10% 达到 38.8 后急剧下跌并持续为负

**根因分析：**

1. **Delta-only 距离奖励导致梯度消失**：`x_cur² - x_nxt²` 在 agent 靠近目标后趋近于零。一旦 agent 进入某个"舒适区"（悬停在某处），每步净奖励 ≈ 0，没有驱动力让它继续靠近并着陆。

2. **contact_bonus 过于稀疏**：同时要求 `both_legs=1 AND |x|<0.2 AND |y|<0.1`。初始探索侥幸触发一次后（cp10%），agent 再无法复现，该信号完全消失。

3. **正负信号互相抵消**：goal_proximity ≈ +3.76，orientation_penalty ≈ -3.96，两者量级几乎完全对消。agent 学到的策略是"悬停不动"——既不靠近也不远离，保持姿态稳定，在零和博弈中苟活 1000 步。

4. **没有"着陆紧迫感"**：缺乏促使 agent 尽早结束 episode 的机制。当前奖励结构下，悬停 1000 步和着陆都是差不多的净收益（甚至悬停更安全）。

### behavior_diagnosis

Agent 当前行为模式：**惰性悬停（lazy hovering）**。从高空出发后，agent 学会了下降到某个中间高度并保持静止，既不接近着陆平台也不飞走。这是一种 reward hacking —— agent 发现只要维持姿态稳定、不剧烈移动，就能让 goal_proximity（delta≈0）和 orientation_penalty（角度≈0）互相对消，以 ~0 的净收益度过全部 1000 步。这是不健康的——它完全绕过了着陆这一核心任务。

### 干预策略

1. **引入直接距离惩罚** `-w * distance`：确保"远离目标 = 持续扣分"，消除悬停的零和博弈
2. **保留 approach_bonus**（距离改善奖励）：在直接惩罚之上叠加正向引导
3. **新增 landing_approach 连续塑形奖励**：在靠近平台 + 低速 + 姿态正直时给予正向激励，填补"接近"与"着陆"之间的奖励空白
4. **contact_bonus 使用软门控**（指数衰减而非硬阈值）：显著降低稀疏性
5. **降低 orientation_penalty 权重**：避免它与 proximity 信号对消
6. **新增极小的 fuel_penalty**：鼓励引擎效率## 诊断报告

### evidence

| 指标 | 数值 | 含义 |
|---|---|---|
| Score | **-32.35** | 严重负分 |
| Terminated | **0/20** | 无一着陆，全部超时截断 |
| Episode Length | **1000** (上限) | agent 活满每一步，拒绝着陆 |
| contact_bonus ep_sum | **0.00**, active=0% | 完全死亡组件 |
| contact_bonus cp10% | 5.16 → 立即归零 | 初始侥幸触发后无法复现 |
| goal_proximity 全 checkpoint | 3.38~3.96，几乎平坦 | agent 停止向目标移动 |
| orientation_penalty vs goal_proximity | -3.96 vs +3.76 | **几乎完全对消** |
| velocity_penalty | -0.49 | 量级太小，无行为影响 |

### behavior_diagnosis

**当前学到的行为：惰性悬停（Lazy Hovering）。**

三个致命机制叠加导致了这一结果：

1. **梯度消失**：v1 的 `goal_proximity = x_cur² - x_nxt²` 是纯 delta 信号。一旦 agent 进入某个"舒适区"悬停不动，每步 delta ≈ 0，goal_proximity ≈ 0。不再有任何力把它拉向目标。

2. **正负信号对消**：orientation_penalty（~-3.96）与 goal_proximity（~+3.76）量级几乎完全相等且符号相反。agent 发现：只要保持姿态稳定、不移动，orientation_penalty 接近 0，goal_proximity 也接近 0，净奖励 ≈ 0。这就是它在 1000 步中做的事——原地悬停苟活。

3. **着陆信号过于稀疏**：contact_bonus 要求 `both_legs=1 AND |x|<0.2 AND |y|<0.1` 同时满足。cp10% 随机探索触发一次（5.16）后再也无法复现。agent 学到的结论是"着陆不可达"，转而选择更安全的悬停策略。

**本质问题：奖励函数缺少"远离目标就要付出代价"的持续压力，也缺少从"接近"到"着陆"之间的平滑奖励桥梁。**

---

## v2 代码

核心改动：

| 改动 | v1 | v2 | 原因 |
|---|---|---|---|
| 距离信号 | 纯 delta（梯度消失） | 直接距离惩罚 + approach_bonus | 保证全局梯度 |
| contact_bonus | 硬阈值 AND 门 | 指数软门控 | 降低稀疏性，扩大发现面 |
| 着陆桥梁 | 无 | landing_approach 塑形奖励 | 填补接近→着陆的空白 |
| 姿态惩罚 | w=0.1（过重） | w=0.03 | 避免与 proximity 对消 |
| 燃料 | 无 | 极小额惩罚 | 鼓励效率，不影响主目标 |

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: direct distance penalty + approach bonus + soft-gated contact + landing shaping

    Core idea: "being far from target always costs something" — no more lazy hovering.
    A smooth reward bridge connects "near the pad" → "actually landing".
    """

    # --- unpack ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- helpers ---
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2

    # ================================================================
    # 1. goal_proximity — direct penalty + approach incentive
    #    "You pay rent for being away. You earn credit for getting closer."
    # ================================================================
    w_dist = 0.08                     # per-unit-distance penalty
    proximity_penalty = -w_dist * dist_nxt

    w_approach = 0.4                  # bonus per unit of distance reduced
    delta_dist = dist_cur - dist_nxt  # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ================================================================
    # 2. velocity_damping — penalise speed, stronger near the pad
    # ================================================================
    w_v = 0.15
    k_gate = 2.0
    gate = 1.0 / (1.0 + k_gate * dist_nxt)   # ~1 when close, ~0 when far
    velocity_penalty = -w_v * gate * speed_sq

    # ================================================================
    # 3. orientation_penalty — reduced weight (was 0.1, now 0.03)
    #    Goal: stay upright but don't let this dominate proximity
    # ================================================================
    w_ang = 0.03
    w_angvel = 0.005
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ================================================================
    # 4. landing_approach — continuous shaping bridge
    #    Rewards being in a "ready to land" configuration near the pad:
    #    low altitude, centered, slow, upright.
    #    Uses exponential decay — smooth, always > 0, no hard cliffs.
    # ================================================================
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)    # decays as you go higher
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)    # decays with lateral offset
    slow     = 2.718281828 ** (-speed_sq / 0.8)        # decays with speed
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.3)  # decays with tilt
    landing_approach = 0.3 * near_y * near_x * slow * upright

    # ================================================================
    # 5. contact_bonus — soft-gated, no longer a needle in a haystack
    #    Uses exponential spatial gate instead of hard |x|<0.2 & |y|<0.1.
    #    Contact anywhere gives *some* reward; contact near (0,0) gives full.
    # ================================================================
    w_contact = 25.0
    both_legs = left_cont_nxt * right_cont_nxt          # 0.0 or 1.0
    spatial_gate = 2.718281828 ** (-(x_nxt ** 2 + y_nxt ** 2) / 0.5)
    contact_bonus = w_contact * both_legs * spatial_gate

    # ================================================================
    # 6. fuel_efficiency — tiny nudge toward engine-off when possible
    # ================================================================
    w_fuel = 0.003
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ================================================================
    # assemble
    # ================================================================
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + contact_bonus
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'contact_bonus': contact_bonus,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components
```