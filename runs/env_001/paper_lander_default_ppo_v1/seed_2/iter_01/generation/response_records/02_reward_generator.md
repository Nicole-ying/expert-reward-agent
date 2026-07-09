# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract next observation variables
    x = next_obs[0]      # x relative to pad center
    y = next_obs[1]      # y relative to pad height
    vx = next_obs[2]     # horizontal velocity
    vy = next_obs[3]     # vertical velocity
    angle = next_obs[4]  # body angle
    omega = next_obs[5]  # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Main drive: distance to target (dense negative signal)
    dist_to_target = (x**2 + y**2) ** 0.5
    reward_dist = -2.0 * dist_to_target

    # 2. Stability constraint: penalise high speeds and large angles
    reward_stability = (
        -0.1 * abs(vx) -
        0.1 * abs(vy) -
        0.1 * abs(angle) -
        0.1 * abs(omega)
    )

    # 3. Soft landing proxy: reward simultaneous near-target, low-speed,
    #    upright attitude and both legs contacting the pad.
    prox_dist_thresh = 0.3
    prox_vel_thresh = 0.2
    prox_angle_thresh = 0.1

    condition = (
        dist_to_target < prox_dist_thresh and
        abs(vx) < prox_vel_thresh and
        abs(vy) < prox_vel_thresh and
        abs(angle) < prox_angle_thresh and
        left_contact > 0.5 and
        right_contact > 0.5
    )
    reward_landing = 1.0 if condition else 0.0

    total_reward = reward_dist + reward_stability + reward_landing

    components = {
        "distance_reward": reward_dist,
        "stability_penalty": reward_stability,
        "soft_landing_proxy": reward_landing
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **distance_reward**  
  角色：主学习信号。每一步根据飞行器到目标着陆垫中心的欧氏距离给出负奖励，驱动 agent 尽可能靠近目标位置。它是稠密的连续梯度源。

- **stability_penalty**  
  角色：轻量稳定约束。对速度（水平/垂直）、机体角度和角速度施加惩罚，抑制过快的飞行和剧烈翻滚，帮助 agent 学会平稳接近并避免 crash。

- **soft_landing_proxy**  
  角色：任务完成近似信号。当飞行器同时满足“接近目标、低速、姿态接近水平、双腿接触”时给予正向奖励，弥补环境中缺少显式成功标志的问题，引导 agent 完成最终着陆。

- **为何未使用 terminal_success_reward / terminal_failure_penalty**  
  环境未提供显式的 success/failure 标志（`explicit_success_flag_available: false`，`info` 为空），因此不能依赖终端成功奖励或失败惩罚。

- **未加入的组件与后续迭代方向**  
  - `energy_penalty` / `time_penalty`：v1 阶段 agent 应优先学会到达并稳定着陆，效率优化留至后续版本。  
  - 更平滑的 proxy（如连续函数代替离散阈值）或动态阈值，用于提升训练稳定性和克服稀疏性。  
  - 若 agent 学会任务后速度仍过快，可加强稳定性权重或引入速度上限约束。

- **训练后应观察的 failure mode**  
  - **high_reward_on_pad_without_stabilising**：agent 可能学会短暂触碰垫面获得 proxy 后立刻抬起，形成“弹跳”式 exploit。需检查 proxy 条件是否需要更严格的停留时长（后续可改为累积奖励）。  
  - **oscillation_near_goal**：距离信号可能导致 agent 在目标附近来回摆动而不完成着陆，需关注是否过度依赖距离奖励而忽视 proxy。  
  - **afraid_to_move**：稳定性惩罚若过强，agent 可能选择不启动引擎，仅靠惯性缓慢飘浮，导致学习缓慢。当前系数保守（0.1），一般不会出现此问题。
