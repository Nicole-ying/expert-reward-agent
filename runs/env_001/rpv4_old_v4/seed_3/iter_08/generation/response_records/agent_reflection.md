# Response Record

Now I have the full picture. Let me compile the diagnosis.

---

## 1. 诊断报告

### evidence

**全局演进 — iter 7 突破了「发现-遗忘」循环：**

| iter | score | contact_bonus | contact_active | descent_incentive | term |
|------|-------|---------------|----------------|-------------------|------|
| 1 | -32.35 | 0.00 (dead) | 0.0% | — | 0/20 |
| 2 | 107.13 | 154.47 (92.5% mag) | 2.8% | — | 11/20 |
| 3 | -26.17 | 0.00 (dead) | 0.0% | — | 0/20 |
| 4 | 183.52 | 114.08 (86.3% mag) | 4.9% | 0.78 (0.6%) | 19/20 |
| 5 | -15.01 | 0.00 (dead) | 0.0% | 0.62 (3.2%) | 0/20 |
| 6 | -15.73 | 0.00 (dead) | 0.0% | 3.37 (15.8%) | 0/20 |
| **7** | **227.64** | **36.60 (55.8%)** | **20.3%** | **11.55 (17.6%)** | **19/20** |

iter 7 的 L2 修改（descent_incentive 从线性门控切换为势能式 shaping `φ(y)=1/(y+0.1)`）是转折点：descent_incentive ep_sum 从 3.37 → 11.55，contact_bonus 首次在与 descent_incentive 共存时成功激活。得分 227.64 > 200，目标已解决。

**iter 7 checkpoint 趋势揭示的残留不稳定：**
- cp60%: 246.1（骤升），cp70%: 157.2（回落），cp80%: 223.1，cp90%: 248.3，cp100%: 236.3
- **contact_bonus 在 cp70% 飙升至 412.079**（cp60%: 61.0, cp80%: 64.5），表明训练中期存在 exploit 式 spike
- descent_incentive 全程稳定：cp40%: 12.6 → cp100%: 11.5，是可靠的连续信号

**当前组件健康度：**
- `contact_bonus`: 55.8% mag share，20.3% active — 非 exploit（<90%），但仍是最大单组件
- `descent_incentive`: 17.6% mag share，87.9% active — 健康，连续梯度
- `goal_proximity`: 12.1% mag share，96.4% active — 稳定
- `velocity_penalty`: 10.4% mag share，97.0% active — 正常
- `orientation_penalty`: **3.7% mag share，100% active，始终为负** — 全覆盖惩罚，梯度微弱

### behavior_diagnosis

策略已学会着陆（19/20 成功），但 training 中期 contact_bonus 的 exploit spike（cp70%: 412）表明稀疏 bonus 仍在某些阶段被过度放大，可能导致训练不稳定。orientation_penalty 是全覆盖惩罚 — 每步都触发但幅度极小（每步均值 -0.0064），相比其他组件几乎不提供有效梯度，本质上是噪声税。

### signal_completeness

五个信号覆盖了完整着陆 pipeline：进度引导（goal_proximity + descent_incentive）、速度约束（velocity_penalty）、姿态约束（orientation_penalty）、着陆确认（contact_bonus）。不缺信号，但 orientation_penalty 的均匀施加（不区分高度）使它成为弱信号。

### selected_level

**L1** — 目标已达成（227.64 > 200），差距为负。根据"接近目标时优先微调"原则，只调系数。iter 7 的 L2 势能式 descent_incentive 已被证明是有效的连续引导信号，加强它是最安全的改进方向。

### selected_intervention

**`descent_incentive` 权重 w_desc: 1.5 → 2.5**。单一系数修改，服务于同一语义：强化连续下降引导，降低对稀疏 contact_bonus 的相对依赖。

### falsifiable_hypothesis

descent_incentive 是当前最可靠的连续梯度来源（势能式 shaping 在 y→0 时梯度自然增强）。将其权重从 1.5 提升到 2.5 将：
1. 增强每步下降的奖励力度，加速 agent 接近地面
2. 提高 descent_incentive 的 magnitude_share（预计从 17.6% → ~27%），间接压缩 contact_bonus 的相对份额（预计从 55.8% → ~45%）
3. 降低 cp70% 式 exploit spike 对总奖励的冲击

### expected_next_round

- score: 维持在 220-260 区间，或略有提升
- descent_incentive ep_sum: 从 11.55 → ~18-20
- contact_bonus magnitude_share: 从 55.8% → ~40-50%
- orientation_penalty: 不变（未修改）
- terminated: 保持 19-20/20
- 训练稳定性（checkpoint 曲线）应更平滑，cp60%-cp80% 之间的波动减小

### main_risk

descent_incentive 过强可能导致 agent 过度优先下降而忽略水平位置修正。但 near_center 门控（|x|<0.5 内线性衰减）提供了保护——偏离中心时下降奖励自动削弱，agent 仍需先修正水平位置。此风险可控。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v8 reward: L1 — amplify descent_incentive weight (1.5→2.5).

    Context (v7): L2 potential-based descent_incentive resolved the "hover near pad"
    problem and achieved 227.64 (target 200). However, contact_bonus still shows
    exploit spikes during training (cp70%: 412 ep_sum), indicating the sparse bonus
    can destabilize learning.

    Change:
      - w_desc: 1.5 → 2.5. Strengthens the continuous descent gradient to reduce
        relative dependence on the sparse contact_bonus. The potential φ(y)=1/(y+0.1)
        already provides natural gradient intensification near y=0; higher weight
        amplifies this well-shaped signal.
      - All other components unchanged.

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

    # --- 1. goal_proximity (improvement_delta, squared distance) ---
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (L2 potential-based shaping, L1 weight boost) ---
    # Potential φ(y) = 1/(y+eps): increases as y decreases, with derivative
    # dφ/dy = -1/(y+eps)^2 that grows strongly near y=0. This creates a
    # natural funnel: the same descent step yields MORE reward when closer
    # to the ground.
    #
    # Weight boosted 1.5→2.5 to strengthen continuous descent gradient
    # and reduce relative dependence on sparse contact_bonus.
    #
    # Gated by near_center so agent only gets descent credit when aligned
    # with the landing pad.
    w_desc = 2.5
    eps = 0.1
    potential_cur = 1.0 / (y_cur + eps)
    potential_nxt = 1.0 / (y_nxt + eps)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descent_incentive = w_desc * near_center * max(0.0, potential_nxt - potential_cur)

    # --- 5. contact_bonus (sparse landing confirmation) ---
    w_contact = 2.0
    y_th = 0.1
    x_th = 0.2
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + descent_incentive + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'descent_incentive': descent_incentive,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```
