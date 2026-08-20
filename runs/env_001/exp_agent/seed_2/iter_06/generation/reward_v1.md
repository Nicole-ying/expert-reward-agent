# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 使用 next_obs 计算所有奖励组件，确保奖励反映动作后的状态
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. 主学习信号：密集距离奖励（负欧氏距离）
    distance = (x**2 + y**2) ** 0.5
    dist_reward = -distance

    # 2. 稳定约束：惩罚速度、姿态角和角速度
    stability_penalty = -0.1 * (abs(vx) + abs(vy)) \
                        - 0.2 * abs(angle) \
                        - 0.05 * abs(angular_vel)

    # 3. 任务完成近似信号：软着陆奖励（仅当多条件同时满足时激活）
    landing_proxy = 0.0
    if (abs(x) < 0.1 and abs(y) < 0.1 and 
        abs(vx) < 0.2 and abs(vy) < 0.2 and 
        abs(angle) < 0.1 and 
        left_contact == 1.0 and right_contact == 1.0):
        landing_proxy = 1.0

    total_reward = dist_reward + stability_penalty + landing_proxy

    components = {
        "dist_reward": dist_reward,
        "stability_penalty": stability_penalty,
        "landing_proxy": landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件与角色

- **dist_reward（主学习信号）**：以负欧氏距离作为密集引导，每一步奖励为 `-distance(next_obs, target)`。越靠近目标平台惩罚越小，鼓励智能体减少与目标之间的相对距离。这是与 `progress_delta_reward` 完全不同的骨架，不依赖相邻两步的距离差，避免因目标附近震荡导致的 reward 噪声。

- **stability_penalty（稳定/安全约束）**：对水平速度、垂直速度、机体角度和角速度施加轻量连续惩罚，引导智能体以一种平缓、稳健的姿态接近目标，减少剧烈翻滚或高速撞击的风险。权重较小，不至于让智能体因惧怕惩罚而不敢移动。

- **landing_proxy（任务完成近似信号，小权重）**：当智能体同时满足位置很近、速度很小、姿态很平且双支撑腿均接触时，给出一个小的正向激励。它不伪造 `success` 标志，只是对“看起来像安全着陆”的状态提供额外鼓励，帮助智能体学会稳定停在平台上。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

`environment_card` 明确声明 **explicit_success_flag_available = false** 且 **explicit_failure_flag_available = false**。没有显式成功/失败信号可用，`info` 字典为空。任何试图读取 `info['success']` 或用终止原因推断成败都可能误判（例如 `body_not_awake_or_settled` 既可能是成功稳定着陆，也可能是意外摔落）。因此在 v1 中完全禁用这些终点奖励，避免引入错误的学习信号。

## 留到后续迭代的组件

- **energy_penalty / time_penalty**：当前版本优先让智能体学会安全到达并稳定着陆，过早加入引擎使用惩罚或时间惩罚可能导致不想移动或冒险快速失败。后续当接近成功率较高后再引入能耗优化。
- **gated_reward**：复杂的安全门控结构留到后面迭代使用，v1 先用简单的恒常惩罚项控制稳定性。
- **terminal_success_reward / terminal_failure_penalty**：只有当 wrapper 明确暴露出可信的 success/failure 标志时，才会在 v2 引入。
- **progress_delta_reward**：尽管它是推荐的导航主信号，但因前序以 progress 为主骨架的尝试失败，本次特意避开该路径，改用 `distance` 骨架。若 `distance` 效果不佳，后续可考虑用 `potential_based_shaping` 替换。

## 训练后应观察的 failure mode

- **high_reward_without_success**：智能体可能学会在目标附近缓慢振荡，利用 `landing_proxy` 反复触发小奖励而不真正稳定，或者长期停留在 `distance` 很小但 `contacts` 未满足的状态。需要检查成功着陆的 episode 是否真的达到了双接触稳定。
- **fast_crash_near_goal**：`stability_penalty` 权重过低时