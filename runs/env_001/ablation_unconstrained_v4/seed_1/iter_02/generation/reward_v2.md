## 分析

### 1. Agent发生了什么？

**核心问题：所有20集都在约91步提前终止（terminated=20/20），score=-107.52极负。** 

关键证据链：
- `soft_landing_bonus`的`active_rate=0.8%`，意味着在91步×20集=1820个step中，只有约15次激活。这是一个**极端稀疏的产品型奖励**——需要`contact_flag`、接近目标、低速、小角度、小角速度这5个条件同时近似满足才有值。对于尚在探索阶段的agent，几乎永远得不到landing引导信号。
- `distance_progress`全程激活但episode_sum_mean仅1.2，在极度负分的整体score面前完全不足以改变行为。
- `posture_penalty`虽然绝对值不大(-0.45/episode)，但作为全局持续惩罚，在agent已经因其他原因崩溃时没有帮助。

**结论：agent从未学到着陆动作，因为landing bonus过于稀疏（0.8%触发率），progress信号太弱，agent在所有episode中提前撞毁。**

### 2. 哪个组件最值得干预？

**soft_landing_bonus**。0.8% active_rate触发了检索知识中的`sparse_completion_proxy`模式——这种乘积形式本质上是隐式的二值条件组合，在agent学会粗略接近目标之前几乎没有梯度。必须将其分解为可加的连续组件。

### 3. 上一轮改了什么？

Agent Memory显示iter 1是唯一一次尝试（new_best），使用了product-form landing bonus + weak progress + global posture penalty。当前被拒绝的重复版本与iter 1语义完全一致，不是有效干预。

### 修改方案

**根本性结构变更**：
1. **移除产品型landing bonus**（5个指数项乘积×contact_flag），替换为分解的加性组件
2. **新增连续proximity shaping**：`1/(1+dist)`全程激活，提供从任何位置到目标的密集梯度
3. **新增加性contact reward**：`contact_count * near_pad`，不需要所有条件同时完美
4. **姿态/速度惩罚改为近垫门控**：使用`near_pad = exp(-dist²/0.5)`作为调制器，远距离时≈0让agent自由移动，接近时才施加稳定化压力
5. **progress权重从1.0提升到3.0**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract current and next state ----
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- distances ----
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # proximity gate: ~1.0 near pad, decays to ~0 far away
    near_pad = 2.718281828 ** (-next_dist**2 / 0.5)

    # Component A: distance progress (strengthened main signal)
    progress = prev_dist - next_dist
    c_progress = 3.0 * progress

    # Component B: continuous proximity shaping (dense gradient everywhere)
    c_proximity = 1.0 / (1.0 + next_dist)

    # Component C: decomposed contact reward (additive, not product)
    contact_count = n_left_contact + n_right_contact  # [0.0, 1.0, 2.0]
    c_contact = 3.0 * contact_count * near_pad

    # Component D: speed penalty gated by proximity
    speed_sq = nvx**2 + nvy**2
    c_speed = -0.5 * speed_sq * near_pad

    # Component E: posture shaping (mild global + focused near-pad)
    c_angle_global = -0.03 * (n_angle**2)
    c_angle_near   = -0.5 * (n_angle**2) * near_pad
    c_angvel_global = -0.003 * (n_angvel**2)
    c_angvel_near   = -0.05 * (n_angvel**2) * near_pad

    total_reward = (
        c_progress
        + c_proximity
        + c_contact
        + c_speed
        + c_angle_global
        + c_angle_near
        + c_angvel_global
        + c_angvel_near
    )

    components = {
        'distance_progress': c_progress,
        'proximity_shaping': c_proximity,
        'contact_reward': c_contact,
        'speed_penalty': c_speed,
        'angle_global': c_angle_global,
        'angle_near': c_angle_near,
        'angvel_global': c_angvel_global,
        'angvel_near': c_angvel_near,
    }

    return (float(total_reward), components)
```