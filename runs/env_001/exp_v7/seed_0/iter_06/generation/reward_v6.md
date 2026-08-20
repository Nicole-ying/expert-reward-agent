## 诊断

### 1. Agent 发生了什么？
Score 从 -111 跃升到 **216**，episode_length 从 68 → 527，early_terminal=0%。表面上看很好，但有一个红灯：

| 信号 | mean |
|------|------|
| generated_reward | **+0.337** |
| original_env_reward | **-0.146** |

**符号相反** → misalignment。环境认为 agent 行为平均是负面的，但我们的奖励函数给了正面评价。agent 在"高得分"但可能没有真正完成任务。

### 2. 哪个组件是主要原因？
- **soft_landing_reward**：mean=0.336, nonzero_rate=33.6%, abs_contrib_%=11.85%。**这是总奖励的唯一驱动力**。二值事件型（1.0 or 0），agent 在 1/3 的步骤中触发它。
- **progress_reward**：mean=0.002, abs_contrib_%=**0.09%**。学习信号几乎为零。progress_delta 骨架推荐的 scale 是 5~20，我们用的是 0.5，差了 10-40 倍。

核心问题：二值 soft_landing 在边界处**无梯度**，agent 可能凑巧满足条件后就不再优化。且每步 1.0 的固定值容易被 exploit（徘徊在条件边界附近薅奖励），这解释了为什么 original_env_reward 仍为负。

### 3. 我之前改了什么？
上一轮把 distance_reward 换成 progress_reward（形态改变），得分暴涨。这次不需要换骨架。需要修复 soft_landing 的数学形态——从**二值事件**改为**连续乘积**，提供平滑梯度，让 agent 真正学会精确着陆而不是薅二值边界。

### 修改方案
**层次 2：改数学形式** — 将 `soft_landing_reward` 从二值条件改为连续乘积形式。四个因子（距离、速度、角度、接触）各自用 bounded 函数映射到 [0,1]，乘积确保"同时满足"才有高奖励。保持 progress_reward 和 stability_penalty 不变，让下一轮单独归因。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ======== 提取观测 ========
    x_t, y_t = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    w_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ======== 1. 主学习信号：进步奖励 (progress_delta) ========
    dist_t = (x_t ** 2 + y_t ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_t - dist_next   # 正值为接近目标
    progress_reward = 0.5 * progress

    # ======== 2. 连续软着陆奖励 (soft_landing_continuous) ========
    # 从二值事件改为连续乘积——每个因子在 [0,1]，提供平滑梯度
    # 距离因子：dist_next=0 时为 1，dist_next>=0.3 时为 0
    proximity = max(0.0, 1.0 - dist_next / 0.3)
    # 速度因子：总速度=0 时为 1，总速度>=0.3 时为 0
    speed_norm = abs(vx_next) + abs(vy_next)
    speed_factor = max(0.0, 1.0 - speed_norm / 0.3)
    # 角度因子：angle=0 时为 1，|angle|>=0.2 时为 0
    angle_factor = max(0.0, 1.0 - abs(angle_next) / 0.2)
    # 接触因子：鼓励双腿同时接触，单腿=0.5，双腿=1.0
    contact_factor = (left_contact + right_contact) / 2.0

    soft_landing_reward = proximity * speed_factor * angle_factor * contact_factor

    # ======== 3. 轻量稳定性惩罚 (stability_penalty) ========
    stability_penalty = -0.002 * (abs(vx_next) + abs(vy_next)) \
                        - 0.002 * abs(angle_next) \
                        - 0.001 * abs(w_next)

    # ======== 总奖励 ========
    total_reward = progress_reward + soft_landing_reward + stability_penalty

    # ======== 组件字典 ========
    components = {
        'progress_reward': progress_reward,
        'soft_landing_reward': soft_landing_reward,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

### 改动说明

| 维度 | 旧（二值事件） |