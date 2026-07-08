## 诊断

### 1. Agent 发生了什么？
**Crash**。episode_length=69.9，100% early terminal，score=-107。所有episode在<150步内因碰撞终止。`original_env_reward` mean=-1.55 说明环境每步在猛烈惩罚。

### 2. 哪个组件是主因？
`soft_landing_proxy` 太弱。ratio_to_progress 仅 0.085，而 iter 2 成功时这个 ratio 约 20x。上一版把 binary landing 改成连续乘积 `1/(1+kx)` 四个因子相乘 + 系数 0.15，实际 mean 只有 0.001——比 progress 的 0.016 还小一个数量级。Agent 完全忽略 landing 引导，退化回 crash。

对比 iter 2（score=156）：`soft_landing_proxy` mean=0.061，是 progress 的 20 倍，agent 被强力引向着陆姿态。

### 3. 上一轮改了什么？
把 binary landing 改成连续 `1/(1+kx)` 乘积形式。结果：连续形式自身太弱（四个<1的因子相乘），系数 0.15 远不够补偿。**本轮不再重复这个方向**——换用 `max(0, 1-x/D)` 形式，让因子在接近目标时接近 1，乘积不会塌缩。

### 验证失败原因
上一版 `1/(1+kx)` 导致 nonzero_rate=100%（永不归零），可能触发验证检查；且四个因子乘积在数值上极度微弱。改用 `max(0, 1-x/D)` 后，远离目标时自动归零，nonzero_rate 自然合理。

### 修复方案
- **soft_landing_proxy**：用 `max(0, 1-x/D)` 代替 `1/(1+kx)`。减少因子数量（3个关键因子：距离、速度、角度），每个在理想状态附近接近1。系数 0.4，使接近着陆时信号 ~0.1-0.4，远强于 progress 的 0.016。
- **stability_penalty**：保持低水平（已验证合理）。
- **progress_delta_reward**：保持不变。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract states ----------
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1. main learning signal: progress toward (0,0) ----------
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress_delta_reward = dist_obs - dist_next

    # ---------- 2. stability / smoothness penalty ----------
    vel_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    ang_vel_penalty = abs(ang_vel)

    w_vel = 0.001
    w_angle = 0.005
    w_angvel = 0.001
    stability_penalty = -(w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (CONTINUOUS, max(0,1-x/D) form) ----------
    # DIAGNOSIS: previous 1/(1+kx) four-factor product collapsed to mean=0.001,
    # too weak to guide. max(0,1-x/D) stays near 1.0 when close to ideal,
    # so product doesn't collapse. Three factors: distance, speed, angle.
    # Coefficient 0.4 gives max 0.4 at perfect landing, ~0.05-0.15 during approach.

    # distance factor: 1.0 at dist=0, 0.0 at dist>=0.5
    dist_factor = max(0.0, 1.0 - dist_next / 0.5)

    # speed factor: 1.0 at zero speed, 0.0 at total_speed>=0.5
    total_speed = abs(vx) + abs(vy)
    speed_factor = max(0.0, 1.0 - total_speed / 0.5)

    # angle factor: 1.0 at angle=0, 0.0 at |angle|>=0.2
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.2)

    # contact: soft factor [0.33, 1.0], rewards any contact
    contact_factor = 0.33 + 0.335 * (left_contact + right_contact)

    soft_landing_proxy = 0.4 * dist_factor * speed_factor * angle_factor * contact_factor

    # ---------- total ----------
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }

    return float(total_reward), components
```