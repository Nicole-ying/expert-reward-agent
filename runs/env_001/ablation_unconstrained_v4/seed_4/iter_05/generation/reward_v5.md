### 分析总结

1. **Agent行为**：训练后策略在20个评估episode中成功着陆18次（90%成功率），平均episode长度约464步。奖励结构中，`contact_reward`（双腿同时接触垫板时的常数奖励）占比高达99%，而引导朝向目标的`progress`仅占0.6%，速度惩罚（`vel_penalty`）完全未触发。说明agent已学会抵达并保持接触，但奖励几乎全部来自“接触后持续停留”这一阶段，对到达效率、着陆质量（速度、姿态）缺乏区分。这导致不同episode得分差异大，整体平均得分未达目标。

2. **最值得干预的组件**：`contact_reward`。其数值占主导，但仅为二值条件（接触=1，否则=0），不对接触瞬间的速度和姿态质量做出响应，浪费了大量信号容量。应将其中一部分权重转化为**与着陆质量相关的连续奖励**，引导agent在接触时保持低速、直立姿态，从而在不破坏已有成功率的前提下提升优质着陆episode的得分，缩小高低分差距。

3. **历史修改回顾**：上一轮在Iter3的`contact_reward + progress`基础上增加了`ang_penalty`和`vel_penalty`，得分从-0.24跃升至140.27，表明增加约束对成功着陆至关重要。但速度惩罚阈值过高（1.2）导致从未激活，未能进一步区分着陆质量。本次修改需降低这些阈值、调整系数，并把接触奖励从“常数”改为“常数+质量因子”的混合形态。

### 修改方案

- **降低基础接触奖励**系数 `w_contact_base`，减少对纯接触时长的无差别奖励。
- **新增接触质量奖励**：在双腿接触时，根据速度与角度的hinge指标（`1 - speed/v_soft`、`1 - angle/a_soft`），用几何平均计算质量因子，乘上较大的权重，使高质量着陆获得更高奖励。
- **强化速度和角度惩罚**：降低惩罚阈值并提高系数，使其在接近阶段就能约束高速、大倾角行为。
- **增加轻量燃料惩罚**：鼓励减少引擎使用，缩短无效漂移，提升效率。
- **提高进度奖励权重**，加强向目标移动的梯度信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox = 25.0               # distance progress weight (increased)
    w_contact_base = 2.0        # low base contact reward
    w_contact_quality = 30.0    # quality bonus for good landing
    v_soft = 0.5                # target speed for quality (soft limit)
    a_soft = 0.25               # target angle for quality (soft limit)
    k_vel_pen = 2.0             # velocity penalty coefficient
    k_ang_pen = 2.0             # angle penalty coefficient
    v_threshold = 0.6           # safe speed threshold (lowered)
    a_threshold = 0.2           # safe angle threshold (rad)
    fuel_cost = 0.5             # fuel penalty per engine use

    # ---------- 1. distance progress ----------
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_dist - next_dist
    progress_reward = w_prox * progress

    # ---------- 2. contact reward (base + quality) ----------
    contact = next_obs[6] * next_obs[7]   # 1.0 when both legs touch
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])
    quality_speed = max(0.0, 1.0 - speed / v_soft)
    quality_angle = max(0.0, 1.0 - angle / a_soft)
    quality = (quality_speed * quality_angle) ** 0.5
    contact_reward = w_contact_base * contact + w_contact_quality * contact * quality

    # ---------- 3. speed and angle penalties (hinge) ----------
    vel_penalty = k_vel_pen * max(0.0, speed - v_threshold)
    ang_penalty = k_ang_pen * max(0.0, angle - a_threshold)

    # ---------- 4. fuel penalty ----------
    fuel_penalty = fuel_cost * float(action != 0)

    # ---------- total ----------
    total_reward = progress_reward + contact_reward - vel_penalty - ang_penalty - fuel_penalty

    components = {
        'progress': progress_reward,
        'contact_reward': contact_reward,
        'vel_penalty': vel_penalty,
        'ang_penalty': ang_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components
```