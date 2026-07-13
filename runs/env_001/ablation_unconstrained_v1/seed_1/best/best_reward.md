# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 主学习信号：接近目标的进度 ---
    # 使用当前和下一步到目标 (0,0) 的欧氏距离差
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_obs - dist_next  # 靠近为正，远离为负

    # --- 稳定和平滑约束 ---
    # 惩罚过大的速度、偏角和角速度，促使平稳着陆
    vel_penalty = 0.05 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.1 * abs(next_obs[4])
    angvel_penalty = 0.01 * abs(next_obs[5])
    stability_penalty = -(vel_penalty + angle_penalty + angvel_penalty)

    # --- 任务完成近似信号：软着陆奖励 (proxy) ---
    # 当双腿同时接触平台、位置靠近中心、速度足够小、姿态接近水平时给予一次性奖励
    landing_bonus = 0.0
    both_legs_down = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    near_center = (abs(next_obs[0]) < 0.2) and (abs(next_obs[1]) < 0.2)
    slow_enough = (abs(next_obs[2]) < 0.2) and (abs(next_obs[3]) < 0.2)
    upright = abs(next_obs[4]) < 0.1

    if both_legs_down and near_center and slow_enough and upright:
        landing_bonus = 10.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus

    # 组件字典（只包含被加到 total_reward 的项）
    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_bonus': landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 奖励组件与角色

- **progress_reward**（主学习信号）  
  用当前步到下一步的欧氏距离差驱动 agent 尽快靠近目标平台。每步都有梯度，是任务的主要驱动力。

- **stability_penalty**（稳定约束，0-2个）  
  同时惩罚较大的水平/垂直速度、机体倾角和角速度，引导 agent 在接近目标时减速并保持姿态水平，为成功着陆创造条件。权重很小，不会压制向目标移动的趋势。

- **landing_bonus**（任务完成近似信号，0-1个）  
  当两条支撑腿同时着地、位置接近平台中心、速度很低且姿态接近水平时，给予一次性大额奖励。这个多条件组合的 proxy 在 episode 终止（稳定着陆）时触发，不能重复利用，规避了接触 exploit 风险。

## 未使用的骨架及原因

- **terminal_success_reward / terminal_failure_penalty**  
  explicit_success_flag_available 和 explicit_failure_flag_available 均为 false，info 为空，无法获取成功/失败标签，跳过。

- **energy_penalty / time_penalty**  
  v1 阶段的首要目标是让 agent 学会到达并稳定着陆，效率类约束会后续迭代加入，避免 agent 因担心代价而不敢行动。

- **gated_reward / dynamic_curriculum**  
  复杂门控不适合 v1；当前设计已经能够提供连续的形状引导和稀疏的着陆信号。

## 后续迭代方向

- 当 agent 能够较稳定地完成任务后，可引入**轻量能源惩罚**（基于 action 是否使用引擎），进一步优化燃料消耗。
- 可收紧 soft landing proxy 的条件或将其转换为连续信号（例如双腿接触时给予小幅度持续奖励），减少学习难度。
- 如果发现 agent 在目标附近长时间震荡不降落，可提高 **landing_bonus** 或加入**姿态/速度退火**以鼓励最终着陆。

## 预期 failure mode 及观察

- **长时间不降落**：agent 到达目标附近但迟迟不减速或不平整，proxy 触发率极低。观察 episode 长度与着陆速度分布，必要时放宽 proxy 的速度/角度阈值或增加持续着陆引导。
- **单腿着地 exploit**：仅一条腿接触可能获得部分位置奖励，但缺少 landing_bonus。应监控左/右腿接触频率，若发现单腿着陆能获得足够累积 reward 而不完成双腿着陆，后期可加强双腿同时接触的引导。
- **过度使用主引擎**：追求快速下降可能产生过大的垂直速度，导致冲击过大。稳定惩罚虽能抑制，但若仍需强化，可在后续增加更针对性的垂直速度惩罚或冲击模拟。