# Response Record

## 诊断报告

### Evidence

| 指标 | 数值 | 含义 |
|---|---|---|
| Score | -32.35 | 远低于目标 200，且为负 |
| Len | 1000.0 | 所有 episode 跑满步数 |
| Terminated | 0/20 | 无一次自然终止，从未着陆成功 |
| goal_proximity ep_sum | +3.76 | 1000 步仅积累 ~0.0038/步，信号极度微弱 |
| orientation_penalty ep_sum | -3.96 | 与 goal_proximity 几乎完全抵消 |
| contact_bonus ep_sum | 0.00 (active 0%) | **彻底死亡** |
| cp10% score | **+38.8** | 早期探索曾找到正向行为！ |
| cp10% contact_bonus | **5.16** (active 0.8%) | 早期曾触发过接触，随后完全灭绝 |

### Behavior Diagnosis

**当前策略行为：围绕目标附近悬停/振荡，永不落地。**

证据链：
1. **cp10% 分数 +38.8 → cp100% 分数 -27.8**：早期探索阶段 agent 偶然触发了着陆接触（contact_bonus=5.16），获得了正向总分。但随着训练推进，agent **主动遗忘**了着陆行为——因为 contact_bonus 的条件过于严苛（x_th=0.2, y_th=0.1 且必须双腿同时接触），且奖励量级不足以对抗 orientation_penalty 的持续消耗。

2. **goal_proximity 是 delta-based（`x_cur² - x_nxt²`）**：只在"靠近"时给正奖励，"远离"时给负奖励。一旦 agent 到达目标附近，继续靠近的边际收益趋近于零，振荡时正负相抵，1000 步净积累仅 +3.76。这是一个**信号塌缩**问题——离目标越近，梯度越弱。

3. **orientation_penalty 与 goal_proximity 量级持平**（+3.76 vs -3.96）：两者对消后净信号 ≈ 0，agent 处于"奖励荒漠"——没有足够梯度指引它完成最后一步着陆。

4. **contact_bonus 死亡是结果而非原因**：agent 从未被充分激励去尝试着陆，因为靠近目标后 net reward ≈ 0，没有梯度推动它跨越"悬停→着陆"的鸿沟。

**核心问题：奖励结构创造了一个"靠近但不落地"的局部最优陷阱。**

### 干预方案（L2 — 单一语义修改）

**语义方向：将着陆（contact）重建为清晰的全局最优解。**

两项联动修改服务于同一语义——"让着陆成为不可抗拒的最优选择"：

1. **goal_proximity：delta-based → direct distance penalty**。改为 `-w * sqrt(x² + y²)`，确保 agent 在任何距离都有持续、不衰减的梯度指向原点。越靠近，惩罚越小，但梯度不会消失。

2. **contact_bonus：连续弱奖励 → 稀疏大额奖励**。放宽阈值（x_th=0.3→0.5, y_th=0.1→0.3），改为离散的 100 点奖励。早期探索中 agent 已证明能找到接触（cp10% active=0.8%），现在只需让这个发现被强力强化。

辅助微调（同一语义内的权重配合）：降低 orientation_penalty 权重以避免淹没正向信号，提高 velocity_penalty 近地权重以鼓励软着陆。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: landing-as-global-optimum — direct distance gradient + large sparse contact bonus
    
    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: direct distance penalty ---
    # Continuous, non-decaying gradient toward (0,0).
    # Using sqrt (Euclidean) gives linear falloff; Manhattan would also work.
    # Weight chosen so that at typical start distance ~8, penalty ≈ -0.4/step,
    # which over ~200-step approach accumulates ~-80, clearly beaten by +100 contact.
    w_dist = 0.05
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = -w_dist * dist_nxt

    # --- 2. velocity_damping: stronger near ground ---
    # Quadratic penalty on speed, gated by proximity to pad.
    # Increased weight (0.05→0.15) to meaningfully discourage high-speed landing.
    w_v = 0.15
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: softened to not dominate ---
    # Reduced angle weight (0.1→0.05) so positive signals aren't drowned.
    # Keeps angular velocity damping to suppress wobble.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus: large sparse reward for successful landing ---
    # Relaxed thresholds (x<0.5, y<0.3) vs. previous (x<0.2, y<0.1).
    # Discrete +100 bonus makes landing the unambiguous global optimum.
    # Early training (cp10%) showed agent CAN find contact — now it will be reinforced.
    w_contact = 100.0
    near_target = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    contact_bonus = w_contact if (near_target and both_legs) else 0.0

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components
```
