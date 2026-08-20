# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置为原点 (0,0)
    # 计算到目标的欧氏距离（使用 next_obs）
    dist_to_target = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5
    
    # 主学习信号：负距离，引导飞行器持续靠近目标
    distance_reward = -dist_to_target

    # 稳定性惩罚：轻量抑制高速、大角度和角速度，促进安全减速和姿态稳定
    speed_norm = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])
    
    lambda_vel = 0.02
    lambda_angle = 0.02
    lambda_av   = 0.02
    
    stability_penalty = -(lambda_vel * speed_norm + lambda_angle * angle_abs + lambda_av * angvel_abs)

    # 软着陆近似奖励：多条件组合，只有当飞行器非常接近目标、速度极低、姿态稳定且双支撑脚均接触时才触发
    threshold_dist   = 0.3
    threshold_speed  = 0.1
    threshold_angle  = 0.05
    contact_left     = next_obs[6]
    contact_right    = next_obs[7]
    
    if (dist_to_target < threshold_dist and 
        speed_norm < threshold_speed and 
        angle_abs < threshold_angle and 
        contact_left == 1.0 and 
        contact_right == 1.0):
        soft_landing_proxy = 10.0
    else:
        soft_landing_proxy = 0.0

    # 总奖励
    total_reward = distance_reward + stability_penalty + soft_landing_proxy

    # 记录各组件
    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'soft_landing_proxy': soft_landing_proxy,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 组件角色

- **distance_reward**（主学习信号）：  
  每一时刻对飞行器距目标垫的欧氏距离取负值，提供稠密的连续梯度，引导飞行器持续向目标垫移动。在数值上它是奖励的主要正向贡献部分（负值中逐步收窄的负奖励）。

- **stability_penalty**（轻量稳定约束）：  
  对速度、机体倾斜角和角速度进行弱惩罚，防止 agent 为了更快接近目标而保持高速、大倾角或旋转，从而无法稳定着陆。权重设置很小（各 0.02），避免抑制探索，仅作为“方向盘”式的背景约束。

- **soft_landing_proxy**（任务完成近似信号）：  
  当飞行器同时满足“足够接近目标、极低速度、姿态接近水平、双支撑脚均接触”这四个条件时，给予一次性大额奖励（+10）。这个多条件组合不能由单一接触信号轻易 exploit，能够有效鼓励 agent 学会正确且稳定地完成软着陆。

## 为何未使用 terminal_success_reward / terminal_failure_penalty

`explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 false，`info` 字典为空，无法从环境获取可靠的成功或失败标志。任何依赖这些标志的奖励组件都会导致 agent 利用未被定义的信息，因此 v1 中完全不引入。我们通过 soft_landing_proxy 近似替代了成功信号，未来若 wrapper 暴露明确的完成标志，可再加入 terminal_success_reward。

## 留到后续迭代的组件

- **energy_penalty**：任务要求最小化燃料消耗，v1 先让 agent 学会安全着陆，v2 再引入对动作推力（action 1,2,3）的消耗惩罚，实现多目标权衡。
- **time_penalty**：同样为效率组件，后续在 agent 能稳定完成任务后再加入，促使 agent 更快着陆。
- **progress_delta_reward**：当前已用 distance_reward 提供靠近信号，未来若出现目标附近震荡，可考虑替换为带 clip 的 progress_delta_reward，但在 v1 阶段避免重复梯度。
- **terminal_failure_penalty**：待环境能可靠区分失败终止时加入，用于预防坠毁或出界。

## 训练后应观察的 failure mode

1. **目标附近震荡而不完成软着陆**：agent 因 distance_reward 在接近零时梯度变弱，可能盘旋而不满足 soft_landing_proxy 的条件。需检查高频 pendulum 式行为，必要时可收紧 soft_landing_proxy 的条件或增加其权重。
2. **高速俯冲触地获取 bonus 后 crash**：若 soft_landing_proxy 的条件过于宽松（如距离阈值过大），agent 可能高速冲进目标触发 bonus 但随即因姿态不稳 crash。当前通过严格的速度和角度阈值避免。
3. **过度保守／不敢移动**：若 stability_penalty 过强，agent 可能选择几乎不点火，以极低速度缓慢下降，导致无法在规定时间内完成任务。v1 中权重已设置较小，若发现此问题可进一步降低 lambda 值。
4. **忽视燃料经济性**：v1 未约束推力使用，agent 可能频繁开关主引擎以调整位置。此问题是预期内的，将在加入 energy_penalty 后优化。
