## 诊断

### 1. 这个 agent 发生了什么？
- **score=244.95, episode_length=1385.75** — 这个环境是 HalfCheetah 或类似 locomotion 环境，正常 episode 长度是 1000。agent 存活了 1385 步（超过了 1000），说明它没有 crash，而是稳定地走完了全程。
- 但 **original_env_reward = 0.117** — 非常低。说明 agent 虽然存活了，但行走质量很差（几乎没前进）。
- **结论：agent 在"原地徘徊"或"小步慢走"** — 它学会了保持站立（alive_bonus 几乎总是触发，nonzero_rate=99.4%），但没有学会有效前进。

### 2. 哪个组件是主要原因？
- **alive_bonus ratio = 0.869** — 存活奖励的强度是前进速度奖励的 87%！agent 只要站着就能拿到几乎和前进一样多的奖励。
- **progress_reward ratio = 1.0**（基准）— 但它的绝对值只有 0.572，而 alive_bonus 是 0.497。
- **stability_penalty ratio = -0.027** — 几乎可以忽略，惩罚太弱了。
- **根因：alive_bonus 太强，progress_reward 太弱。** agent 发现"站着不动拿 0.5"比"冒险前进拿 0.57"更划算，因为前进需要承受摔倒风险（虽然惩罚很小，但风险本身存在）。

### 3. 我之前改了什么？
这是第一轮（iter=1），没有历史修改。这是初始设计。

### 对齐检查
`original_env_reward` ratio = 0.205（正），与主信号 progress_reward（正）同向 → 没有 misalignment。

---

## 修改方案

**层次：层次 1（改系数）+ 层次 2（改数学形式）**

核心问题：alive_bonus 是二值条件（0 或 0.5），agent 只要站着就拿 0.5，这太容易了。需要：
1. **削弱 alive_bonus** — 从 0.5 降到 0.2，降低"站着不动"的吸引力。
2. **增强 progress_reward** — 从 scale=2.0 提高到 3.0，让前进更有价值。
3. **把 alive_bonus 从二值改成连续** — 根据角度和角速度的"健康程度"给出渐变奖励，而不是全有全无。这样 agent 不仅要站着，还要站得稳才能拿到全额存活奖励。
4. **稍微增强 stability_penalty** — 让摔倒的代价更明显，但保持轻量。

具体改动：
- `fwd_scale`: 2.0 → 3.0
- `alive_bonus`: 二值 0.5 → 连续值 `0.2 * (1 - (hull_angle/0.5)^2) * (1 - (hull_angular_vel/2.0)^2)`，当角度和角速度都为 0 时拿满 0.2，接近边界时趋近 0
- `angle_penalty_scale`: 0.5 → 1.0
- `angular_vel_penalty_scale`: 0.3 → 0.5

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：前进速度奖励 ==========
    # 使用水平速度(obs[2])作为前进驱动力
    forward_velocity = next_obs[2]
    fwd_scale = 3.0  # 从2.0提高到3.0，增强前进驱动力
    progress_reward = fwd_scale * forward_velocity
    
    # ========== 存活奖励：连续化，鼓励稳定站立 ==========
    hull_angle = abs(next_obs[0])
    hull_angular_vel = abs(next_obs[1])
    
    # 连续存活奖励：角度和角速度都接近0时拿满，越接近边界越小
    # 使用二次衰减，当角度=0且角速度=0时 reward=0.2
    # 当角度接近0.5或角速度接近2.0时 reward→0
    angle_factor = 1.0 - (hull_angle / 0.5) ** 2
    vel_factor = 1.0 - (hull_angular_vel / 2.0) ** 2
    # 限制在[0,1]范围，避免负值
    angle_factor = max(0.0, angle_factor)
    vel_factor = max(0.0, vel_factor)
    
    alive_bonus = 0.2 * angle_factor * vel_factor  # 连续值，最大0.2
    
    # ========== 稳定性约束：适度惩罚 ==========
    # 惩罚过大的主体角度和角速度
    angle_penalty_scale = 1.0    # 从0.5提高到1.0
    angular_vel_penalty_scale = 0.5  # 从0.3提高到0.5
    
    angle_penalty = angle_penalty_scale * (hull_angle ** 2)
    angular_vel_penalty = angular_vel_penalty_scale * (hull_angular_vel ** 2)
    stability_penalty = -(angle_penalty + angular_vel_penalty)
    
    # ========== 总奖励 ==========
    total_reward = progress_reward + alive_bonus + stability_penalty
    
    # ========== 组件记录 ==========
    components = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components
```