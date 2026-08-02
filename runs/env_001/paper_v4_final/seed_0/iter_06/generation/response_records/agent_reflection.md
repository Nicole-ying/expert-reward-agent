# Response Record

# 设计理由
**改了什么组件**：`success_bonus`（推断成功奖励）  
**为什么改**：第 5 轮在 `iter4` 有效的 `progress + soft_landing` 基础上新增了 `success_bonus` 和 `contact_stability`，导致 episode length 从 956 断崖式下跌到 68，score 从 142 跌回 -111。组件表中 `success_bonus` 的 `episode_sum_mean=22.5` 但 `active_rate=0.7%`，表明它是一个极度稀疏的二值奖励（触发时有 50 点奖励，但不触发时全为零）。这个稀疏信号严重干扰了价值估计，使 agent 抛弃已经学到的渐进飞行策略，转而追逐几乎不可能稳定复现的“成功瞬间”，从而造成训练崩溃。  
**数学形式变更**：将原来的二值判定 `if is_success: 50 else: 0` 改造为**连续乘积因子**：  
$$
\text{success\_bonus} = w \cdot \prod_{g\in G} \max\!\big(0,\, 1 - \frac{\text{error}_g}{\text{threshold}_g}\big)
$$
其中 $G$ 包含距离、速度、角度、接触四个条件。每个因子在“完全满足阈值”时为 1，否则线性衰减到 0。这样：
- 取消了稀疏跳变，每一步都提供可微的局部梯度；
- 系数从 `50.0` 降至 `1.0`，使该组件在成功边缘时每步贡献 ≤ 1.0，不到主信号（progress ~1.1/episode）的 2 倍，且单步远小于 0.3×主信号，避免排挤其他目标；
- 保留了对**终态完美着陆**的选择性高压引导（当所有因子都接近 1 时该奖励集中爆发），但不会在早期训练中制造噪声。

**系数校准**：`SUCCESS_BONUS = 1.0`，threshold 沿用原有合理的经验值（距离 0.3，速度 0.3，角度 0.2）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander goal-reaching task.
    v6: transform sparse success_bonus into continuous dense factor to restore training stability.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.05
    ANGLE_PENALTY_WEIGHT = 0.01
    CONTACT_WEIGHT = 0.1
    PROXIMITY_THRESHOLD = 0.5
    SUCCESS_DIST_THRESHOLD = 0.3
    SUCCESS_SPEED_THRESHOLD = 0.3
    SUCCESS_ANGLE_THRESHOLD = 0.2
    SUCCESS_SCALE = 1.0          # was 50.0 for sparse bonus; now continuous

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, angvel_o, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, angvel_n, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)

    # ---------- 2) soft landing incentive (continuous slowdown) ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    speed = (x_v_n ** 2 + y_v_n ** 2) ** 0.5
    speed_bonus = 1.0 / (1.0 + speed)
    soft_landing = LANDING_WEIGHT * proximity * speed_bonus

    # ---------- 3) light angular penalty ----------
    angle_penalty = -ANGLE_PENALTY_WEIGHT * (angle_n ** 2)

    # ---------- 4) contact stability reward ----------
    contact_flag = max(left_n, right_n)
    angle_bonus = 1.0 / (1.0 + abs(angle_n))
    contact_stability = (
        CONTACT_WEIGHT * proximity * contact_flag * speed_bonus * angle_bonus
    )

    # ---------- 5) success bonus (dense continuous factor) ----------
    proximity_factor = max(0.0, 1.0 - R_next / SUCCESS_DIST_THRESHOLD)
    speed_factor = max(0.0, 1.0 - speed / SUCCESS_SPEED_THRESHOLD)
    angle_factor = max(0.0, 1.0 - abs(angle_n) / SUCCESS_ANGLE_THRESHOLD)
    contact_factor = max(left_n, right_n)
    success_factor = (proximity_factor * speed_factor *
                      angle_factor * contact_factor)
    success_bonus = SUCCESS_SCALE * success_factor

    # ---------- aggregate ----------
    total_reward = (
        progress_reward + soft_landing + angle_penalty +
        contact_stability + success_bonus
    )

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angle_penalty": angle_penalty,
        "contact_stability": contact_stability,
        "success_bonus": success_bonus,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 观测使用已完整，崩溃源于 `success_bonus` 的极端稀疏二值结构（active_rate 0.7%），不是信号缺失。  
- **behavior**: agent 在短暂的 68 步内快速终止（crash），被偶尔触发的巨大成功奖励破坏了对 `progress_reward` 的依赖。  
- **signal**: `success_bonus` 的二值稀疏性是核心缺陷；改为连续因子后，可提供平滑梯度且不排挤现有渐进信号。  
- **level**: Level 2（稀疏二值 → 连续乘积因子）  
- **hypothesis**: 连续化后 agent 会恢复飞行和减速策略（如同 iter4 的效果），并在接近理想着陆时获得逐步增强的额外激励，从而在稳定后缩短着陆时间。  
- **risk**: 如果 proximity_factor 在靠近目标时过早饱和，可能使 agent 在目标周围徘徊以获得连续奖励；但 `progress_reward` 的逼近驱动力和 speed_factor 对低速的正反馈足以抵消这一局部停滞趋势。
