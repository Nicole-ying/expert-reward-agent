# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：每一步向目标靠近的 progress（potential-based shaping）
    dist_old = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_new = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_old - dist_new

    # 稳定着陆 proxy：多条件连续乘积，引导低速、竖直、双脚接触且对准着陆垫中心
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 温度参数（根据环境大致范围设定，可在后续迭代中调整）
    sigma_p = 1.0    # 位置衰减尺度
    sigma_v = 1.0    # 速度衰减尺度
    sigma_a = 0.3    # 角度衰减尺度（弧度）

    # 距离、速度、角度项的指数和
    exponent = -(
        (x ** 2 + y ** 2) / (2.0 * sigma_p ** 2) +
        (vx ** 2 + vy ** 2) / (2.0 * sigma_v ** 2) +
        (angle ** 2) / (2.0 * sigma_a ** 2)
    )

    contact_factor = (left_contact + right_contact) / 2.0
    # 使用 2.718281828**exponent 替代 exp，避免 numpy 依赖
    stable_landing_reward = 10.0 * contact_factor * (2.718281828 ** exponent)

    total_reward = progress_reward + stable_landing_reward

    components = {
        "progress_reward": progress_reward,
        "stable_landing_reward": stable_landing_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 组件与角色

- **progress_reward（主学习信号）**  
  基于 potential-based shaping，奖励每一步缩短到着陆垫中心 `(0,0)` 的欧氏距离。  
  每一步提供稠密梯度，无稀疏事件依赖，直接告诉飞行器“靠近目标就有分”。

- **stable_landing_reward（任务完成近似 proxy）**  
  通过高斯型平滑乘积，鼓励飞行器在靠近着陆垫中心的同时，保持低速、竖直姿态，并且双脚接触垫面。  
  接触因子为 0 时该项为 0，避免了悬停而不着陆的漏洞；各项采用连续指数衰减，梯度连续，不会在阈值边界消失。  
  最高奖励约 10，与典型单步 progress 积累相当，确保最终着陆的吸引力，同时不干扰早期远距离探索。

## 未使用 terminal_success / terminal_failure 的原因

环境未提供显式成功/失败标志（`explicit_success_flag_available=false`，`info` 为空）。  
因此无法设计 `terminal_success_reward` 或 `terminal_failure_penalty`，所有信号只能从观测中推断。  
`stable_landing_reward` 替代了这一角色：它在成功着陆的最后一刻给出最大奖励，同时保持每一步可微。

## 留到后续迭代的组件

- **动作/能量效率惩罚**：当前 v1 先让飞行器学会完成着陆，后续可以加入轻量引擎使用惩罚以优化燃料消耗。  
- **姿态平滑约束**：若后期发现剧烈抖动，可加入角速度或姿态变化惩罚。  
- **显式安全约束**：若飞行器撞击或出界频繁，可加入速度上限惩罚或边界 soft margin。

## 应观察的失败模式

- **目标附近震荡**：若 `progress_reward` 驱动过度追逐，飞行器可能在中心周围快速移动而不减速；`stable_landing_reward` 应迫使速度归零，观察最终是否能稳定着陆。  
- **单脚着陆**：`contact_factor` 对部分接触也给分，可能导致飞行器仅用一只脚接触糊弄奖励；需要确认最终是否强制双脚触垫，后续可调整接触阈值或权重。  
- **悬停在远离垫面的高空**：若温度参数 `sigma_p` 过大，接近奖励过早饱和，飞行器可能在高空减速而不下降；后续可调小 `sigma_p` 或引入高度相关项。  
- **提前休眠**：`body_not_awake_or_settled` 可能在非理想位置触发并结束 episode；需要监控是否因不满足双脚接触而提前终止，后续可结合终止时观测添加针对性引导。