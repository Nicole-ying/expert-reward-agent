## 诊断

### 0. 任务理解
2D 着陆器：从视口顶部出发，降落到 (0,0) 平台。obs 含位置/速度/角度/腿接触，action 为 4 个离散引擎选择。成功 = 双腿着地 + 低速 + 稳定姿态。

### 1. 发生了什么？
- **Score 120**，best 是 iter 3 的 **146**（5 轮前的峰值）。
- Episode length **966**（近 1000 上限），说明 agent 没有 crash 而是在**徘徊/悬停**。
- **contact_reward nonzero_rate = 64%**：agent 频繁触腿但分数反而从 139（iter 6 无 contact）降到 120——这正是知识库中 `contact_reward_hacking` 模式：agent 在平台上方"点腿"赚取 contact reward 而非真正降落。
- **approach_quality_reward ratio_to_progress = 35.75**：乘积形态的接近品质信号以 35:1 的优势压倒 progress，agent 主要优化"靠近目标 + 低速 + 正姿"——悬停完美满足这三个条件，**不需要着陆**。

### 2. 根因
`approach_quality = prox * speed * angle` 的乘积形态在"悬停于 (0,0) 上方"处形成一个**局部最优陷阱**：三个因子均接近 1.0，乘积奖励极高，但没有一项要求 `y≈0` 或双腿接触。progress 信号太弱（ratio 0.88）无法将其拉出陷阱。contact_reward 不仅没解决悬停，反而被 agent exploit 了——64% 触腿率 + 更低的外部得分说明 agent 学会了"反复点腿"而非"稳定着陆"。

### 3. 我之前做了什么？
上轮（iter 7→8）在 best skeleton 上加 contact_reward → score 从 139 跌到 120。**contact 方向已证伪，本轮不能继续在这个方向修补。**

### 决策：换骨架（层次 3）
同一骨架家族（approach_quality 乘积 + α）已迭代 6 轮（iter 2-8，除 iter 7），最佳得分仍是 5 轮前的 iter 3。满足层次 3 条件。

**新骨架：progress + descent_incentive + landing_settle + attitude_penalty**

核心思路：从"奖励处于好状态"（→悬停）转为"奖励向平台下降的动作"（→着陆）：
- `descent_incentive`：水平对齐时奖励**受控下降**（`max(0, -y_vel)`），悬停时 y_vel≈0 得零奖励，直接打破悬停最优
- `landing_settle`：双腿同时接触 + 低速 + 正姿时给予持续奖励，提供"完成着陆"的终端引导

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Skeleton change: replace approach_quality product with descent_incentive +
    landing_settle to break the hovering local optimum.

    Old skeleton (approach_quality = prox * speed * angle product) created a
    local maximum at "hover near (0,0) with low speed and upright attitude" —
    no requirement to actually land.  Contact_reward (iter 8) was exploited
    via leg-dipping (64% contact rate, score dropped 139→120).

    New skeleton rewards the *action* of controlled descent (not the state of
    being near) and provides a settlement bonus gated behind both-legs contact.
    """

    # ----- Unpack observations -------------------------------------------------
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # ----- Component 1: Progress towards target (0, 0) -------------------------
    dist_current = (x ** 2.0 + y ** 2.0) ** 0.5
    dist_next = (next_x ** 2.0 + next_y ** 2.0) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # ----- Component 2: Descent incentive (anti-hover) -------------------------
    # Rewards controlled downward movement when horizontally aligned with target.
    # Hovering (y_vel ≈ 0) gives zero — directly breaks the local optimum.
    horiz_prox = 1.0 / (1.0 + 5.0 * abs(next_x))
    speed = (next_x_vel ** 2.0 + next_y_vel ** 2.0) ** 0.5
    speed_safety = 1.0 / (1.0 + speed)  # dampens high-speed descent
    descent_incentive = 0.3 * horiz_prox * max(0.0, -next_y_vel) * speed_safety

    # ----- Component 3: Landing settle (terminal-quality proxy) ----------------
    # Rewards being settled with BOTH legs in contact, low speed, and upright.
    # Uses min(left, right) so single-leg contact gives zero — prevents dipping.
    both_legs = min(next_left, next_right)  # 1.0 only if both legs contact
    land_speed_factor = 1.0 / (1.0 + speed)
    land_angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    landing_settle = 0.15 * both_legs * land_speed_factor * land_angle_factor

    # ----- Component 4: Attitude penalty (gentle uprighting) -------------------
    attitude_penalty = -0.01 * abs(next_angle)

    # ----- Total ----------------------------------------------------------------
    total_reward = progress + descent_incentive + landing_settle + attitude_penalty

    components = {
        "progress_reward": progress,
        "descent_incentive": descent_incentive,
        "landing_settle": landing_settle,
        "attitude_penalty": attitude_penalty,
    }

    return float(total_reward), components
```