# Response Record

现在我有了完整的诊断图景。让我综合所有证据，撰写诊断报告和 v5 奖励函数。

---

## 诊断报告

### Evidence

| 证据来源 | 关键数据 |
|---|---|
| **iter_4 score** | 120.84（目标 200），terminated=20/20，len=291.9 |
| **iter_4 score_range** | [-104.5, 267.5]，高方差 — 2/20 early_terminal crash |
| **iter_4 checkpoint 趋势** | cp10%=-46.8 → **cp90%=64.5（峰值）→ cp100%=-1.8（退化）** |
| **landing_bonus 趋势** | cp90%=5387.5 → **cp100%=665**（暴跌 8×），训练末期灾难性遗忘 |
| **vel_damping 趋势** | cp70-90%: active 15-17%（着陆期低）→ **cp100%: active 70.5%（末期飙升）** |
| **landing_bonus magnitude_share** | **97.4%** — 所有其他组件在量级上可忽略 |
| **orientation_penalty** | ep_sum=-0.80，active=100%，但 magnitude_share=0.0% — **完全不起作用** |
| **vel_damping proximity_gate** | sigma=0.5 → dist=1.0 时 gate≈0.135，**激活范围过窄** |

### 逐组件分析

**(1) landing_bonus — 97.4% magnitude_share，但并非 exploit**

当前 50/step per-step 结构是成功的核心驱动力。问题不在 magnitude_share 高（稀疏终端奖励本该如此），而在于：
- checkpoint 趋势显示 cp100% 时 bonus 从 5387 暴跌到 665，说明 agent 在训练末期「遗忘」了着陆技能
- 原因：奖励景观过于「悬崖式」——着陆前几乎无正反馈，一旦着陆则巨量奖励。缺乏平滑梯度导致训练不稳定
- **策略**：保留 50/step，但增加 `near_landing_bonus`（10/step）作为中间阶梯

**(2) progress_reward — 有效但过弱**

ep_sum=1.72，magnitude=0.1%。delta-distance 是正确的导航信号（v4 的核心突破），但系数 1.5 产生的梯度太弱，无法可靠引导 agent 穿越复杂姿态空间。
- **策略**：提升至 2.5

**(3) orientation_penalty — 完全无效**

ep_sum=-0.80，100% active 但 magnitude=0.0%。系数 0.1/0.04 太小，agent 完全忽略姿态约束。这直接导致部分 episode 翻滚 crash（2/20 early terminal）。
- **策略**：提升至 0.25/0.10

**(4) vel_damping — 门控过窄**

sigma=0.5 意味着只在 dist<0.7 时才显著激活。当 agent 从远处高速接近时，阻尼来不及起作用。cp100% 时 vel_damping 突然飙升说明 agent 在末期恢复了高速撞击模式。
- **策略**：sigma 0.5→1.5，coefficient 0.5→4.0，使阻尼从 dist≈2 开始逐步激活

**(5) contact_reward — 适度但可加强**

ep_sum=59.3，22.1% active。作为地面探索信号表现良好。略微提升可增强发现着陆垫的动机。
- **策略**：0.5→1.0

**(6) action_reward — 正常运作**

ep_sum=-5.52，52.1% active。量级合理，不需要大改。略微放宽以减少对必要机动的不当惩罚。

### Behavior Diagnosis

Agent 在 iter_4 学到的行为模式：**部分 episode 成功着陆，但存在 crash 和不稳定**。具体表现为：
- 18/20 episode 成功着陆（terminated 而非 crash），2/20 crash
- 着陆质量不稳定：score_range 跨度 372 点
- 训练末期存在灾难性遗忘（cp90%→cp100% 分数暴跌）

根本原因：奖励景观在着陆点附近过于「悬崖化」——着陆前后奖励差异巨大（0 → 50/step），缺乏平滑的中间梯度。加上 shaping 分量过弱，导致策略梯度信号噪声大、训练不稳定。

### 干预策略

| 变更 | 从 | 到 | 理由 |
|---|---|---|---|
| progress_reward 系数 | 1.5 | 2.5 | 强化导航梯度 |
| orientation_penalty | -0.1/-0.04 | -0.25/-0.10 | 使姿态约束生效 |
| vel_damping sigma | 0.5 | 1.5 | 扩大减速门控范围 |
| vel_damping 系数 | 0.5 | 4.0 | 使阻尼显著 |
| contact_reward | 0.5 | 1.0 | 强化地面探索 |
| **新增 near_landing_bonus** | 无 | 10/step | 架起着陆奖励的中间阶梯 |
| action_reward | -0.04/-0.02 | -0.03/-0.01 | 略微放宽 |

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5: Smoothed landing gradient + strengthened shaping.

    Core changes from v4:
    - near_landing_bonus (10/step) bridges the gap between contact and full landing
    - Wider vel_damping gate (sigma=1.5) for earlier deceleration
    - Stronger orientation penalty so it actually matters
    - Stronger progress_reward for clearer navigation gradient
    - Landing bonus (50/step) remains the dominant terminal incentive
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    x_prev = obs[0]
    y_prev = obs[1]

    # ── 1. Progress: delta-distance (strengthened navigation signal) ──
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    progress_reward = 2.5 * (dist_prev - dist_curr)

    # ── 2. Orientation penalty (strengthened — now actually matters) ──
    orientation_penalty = -0.25 * (angle ** 2) - 0.1 * (angvel ** 2)

    # ── 3. Velocity damping near pad (widened gate + stronger coefficient) ──
    # sigma=1.5 means gate ≈ 0.64 at dist=1.0, ≈ 0.90 at dist=0.5
    # Agent feels deceleration pressure from much farther out
    speed = (vx ** 2 + vy ** 2) ** 0.5
    proximity_gate = 2.718281828 ** (- (dist_curr ** 2) / 1.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -4.0 * proximity_gate * excess_speed

    # ── 4. Contact reward (strengthened ground-exploration incentive) ──
    contact_sum = left_contact + right_contact
    contact_reward = 1.0 * contact_sum

    # ── 5. Landing bonuses: two-tier for smooth gradient ──
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist_curr < 0.6)
    upright = (abs(angle) < 0.35)
    slow = (speed < 0.5)
    is_good_landing = both_on_ground and near_pad and upright and slow

    # near_landing: on the pad but not yet perfectly settled
    # This creates a stepping stone — agent gets rewarded for just being on the pad
    is_near_landing = both_on_ground and near_pad and not is_good_landing

    if is_good_landing:
        landing_bonus = 50.0
    elif is_near_landing:
        landing_bonus = 10.0
    else:
        landing_bonus = 0.0

    # ── 6. Action efficiency (slightly relaxed) ──
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.03
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.01

    # ── Aggregate ──
    total_reward = (
        progress_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
    }
    return float(total_reward), components
```
