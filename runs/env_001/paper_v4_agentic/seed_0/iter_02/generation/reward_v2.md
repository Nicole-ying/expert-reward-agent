# 设计理由

## 信号覆盖审计发现

**终止模式分析**：所有20个episode都是terminated（没有truncated），平均len=211步，说明agent不是快速死亡，而是徘徊后触发某种终止。score范围12-90差异极大——高分的可能偶然成功着陆，低分的可能是crash或出界。agent已经能存活的够久，但**缺少信号告诉它"如何成功结束"而非"如何避免失败"**。

**观测使用扫描**：代码声明了所有8个维度，但实际使用：
- `obs[6]`（left_contact）和`obs[7]`（right_contact）：**声明但未使用**，在组件中完全没出现
- `obs[4]`（body_angle）：仅在`landing_safety_penalty`中作为`tilt_pen`使用
- 其他维度都有使用

**信号缺口**：接触标志`left_support_contact`和`right_support_contact`是区分成功着陆（双腿着地）与crash（单腿或身体其他部位着地）的**唯一可观测信号**。环境事实§5明确说crash_or_body_contact可能是成功着陆也可能是失败，需要通过"双腿是否都接触"来区分。当前奖励函数完全没有利用这个区分信号——agent无法从reward中学习到"双腿接触是好的，单腿或不正常的身体接触是坏的"。

## Level 2 结构变换：新增`landing_contact_bonus`组件

**为什么不能用landing_safety_penalty修补？** landing_safety_penalty只惩罚速度和角度越界，但没有正向引导agent用双腿着地。一个agent可以摔倒（单腿触地、身体触地）但仍然满足低速低角度的条件，这样的episode会被判定为低罚但实际失败了。

**新增组件设计**：
- 使用`next_obs[6]`（left_contact）和`next_obs[7]`（right_contact）
- 形式：连续reward，用distance-based gate激活，reward = `(left_contact + right_contact) * gate`
- gate：`1.0 / (1.0 + 5.0 * dist_next)`，与landing_safety_penalty相同的gate函数，确保只在接近目标时才鼓励接触
- 权重：0.3，使per-step量级在接近目标时约为0.15（当双腿都接触时），不超过主信号2x

**校准检查**：
- 主信号progress_reward per-step ≈ 1.34/211 ≈ 0.0064
- 新组件per-step（在接近目标、双腿接触时）≈ 0.3 * 1.0 * 0.5 ≈ 0.15，这是gate≈1场景下的值，实际agent在接近目标时才激活，平均per-step远小于0.15
- 所有penalty per-step合计：landing_safety_penalty 0.32/211≈0.0015 + x_boundary_penalty 0.0 = 0.0015，不超过主信号0.3x ✓

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation variables
    x, y = obs[0], obs[1]
    x_v, y_v = obs[2], obs[3]
    angle = obs[4]
    ang_v = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ---------- 1. Progress reward: moving toward the landing pad (0,0) ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next          # positive when getting closer
    progress_reward = 1.0 * progress          # weight = 1.0

    # ---------- 2. Horizontal boundary penalty (crash prevention) ----------
    x_limit = 1.2
    x_boundary_penalty = 0.5 * max(0.0, abs(nx) - x_limit)

    # ---------- 3. Landing softness / safety penalty ----------
    # Velocity and angular velocity limits
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)                  # ideal angle is 0

    # Distance‑based activation gate: only enforce strict softness near the pad
    gate = 1.0 / (1.0 + 5.0 * dist_next)     # increases when close to target

    landing_safety_penalty = (0.1 * vel_pen + 0.05 * ang_pen + 0.1 * tilt_pen) * gate

    # ---------- 4. Landing contact bonus: positive signal for proper touchdown ----------
    # Reward both legs contacting the ground, gated to only activate near the target
    landing_contact_bonus = 0.3 * (n_left + n_right) * gate

    # ---------- Total reward ----------
    total_reward = progress_reward - x_boundary_penalty - landing_safety_penalty + landing_contact_bonus

    components = {
        "progress_reward": float(progress_reward),
        "x_boundary_penalty": float(x_boundary_penalty),
        "landing_safety_penalty": float(landing_safety_penalty),
        "landing_contact_bonus": float(landing_contact_bonus)
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 接触标志`obs[6]`和`obs[7]`声明但未使用——这是区分成功着陆与crash的关键信号，当前agent无法从reward中学到"双腿着地是好着陆"。
- **behavior**: agent在徘徊（len=211步），progress微弱（per-step 0.0064），没有明确的成功着陆引导信号。
- **signal**: 缺少正向成功着陆信号——需要告诉agent"在接近目标时，双腿接触是好的"。
- **level**: Level 2
- **hypothesis**: 新增接触奖励让agent有动力在接近目标时保持双腿着地姿态，配合landing_safety_penalty抑制速度，使agent学会"接近→减速→双腿着地→静止"的完整着陆序列。当前成功着陆可能偶然发生但无reinforcement，此组件让偶然成功被capture并强化。
- **risk**: 0.3的权重在gate≈1时给予约0.3 per-step reward，可能让agent过早的降低高度以提前获得接触奖励（双腿接触地面但未到平台），导致提前在非平台区域crash。后续轮次需监控此组件是否与目标位置充分对齐（即gate中dist_next的权重5.0是否足够）。