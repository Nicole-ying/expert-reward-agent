# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Calculate distance to target (0,0) for obs and next_obs
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Primary learning signal: progress towards the landing pad
    progress_reward = 1.0 * (dist_obs - dist_next)

    # Stability penalty: discourage high speeds, large angle, and angular velocity in the new state
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    stability_penalty = -0.01 * speed_next - 0.01 * abs(next_obs[4]) - 0.005 * abs(next_obs[5])

    # Soft landing proxy: small bonus when close, slow, upright and both supports in contact
    landing_proxy = 0.0
    if (dist_next < 0.3 and speed_next < 0.3 and abs(next_obs[4]) < 0.2 and
        next_obs[6] == 1.0 and next_obs[7] == 1.0):
        landing_proxy = 1.0

    total_reward = progress_reward + stability_penalty + landing_proxy

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_proxy': landing_proxy,
        'total_reward': total_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 组件与角色
- **progress_reward**（主学习信号）：  
  每一步计算到目标点 `(0,0)` 的距离变化，鼓励智能体持续向目标垫移动。这是密集的引导信号，帮助探索阶段形成有效行为。
- **stability_penalty**（轻量稳定约束）：  
  对执行动作后的水平速度 + 垂直速度、身体角度和角速度施加小惩罚。引导着陆器在接近目标时降低速度、摆正姿态，为安全着陆创造条件。
- **landing_proxy**（任务完成 proxy）：  
  当智能体处于“近、慢、稳、双足接触”状态时给予一次性的正向奖励（1.0）。这并非真实的成功信号，但可作为近似完成标志，帮助强化最终稳定行为。

## 2. 未使用 terminal_success_reward / terminal_failure_penalty 的原因
环境**未提供显式成功/失败标志**（`explicit_success_flag_available=false`，`info` 为空）。若在 v1 中引入这类奖励，模型会自行猜测或冒用不存在的字段，极易产生幻觉奖励并导致训练崩溃。因此终端奖励留待后续 wrapper 明确暴露 `success` 或 `failure` 信号后再添加。

## 3. 留到后续迭代的组件
- **energy_penalty / time_penalty**：当前版本优先让智能体学会接近并稳定，暂不加入动作代价或时间惩罚，避免出现“不敢动”（agent afraid to move）。待接近成功率稳定后，再引入轻量能耗优化。
- **terminal_success_reward / terminal_failure_penalty**：如上所述，等环境接口提供明确标志后补充。
- **gated_reward / dynamic curriculum**：更复杂的多阶段或安全门控逻辑，适合在基础行为稳定后迭代。

## 4. 训练后需关注的 failure mode
- **高速撞击**：若 `progress_reward` 完全主导，智能体可能以高速冲向目标垫，导致硬着陆甚至崩溃。此时需观察是否出现高速撞地，适当增大 `stability_penalty` 的权重或收紧 `landing_proxy` 的条件。
- **reward hacking via minimal movement**：若 `landing_proxy` 奖励过于丰厚（例如阈值过宽），智能体可能在早期就停留在边界附近获取奖励，而不完成精确着陆。监控 `landing_proxy` 的触发频率，必要时收紧距离/速度/角度阈值。
- **姿态振荡**：若仅靠 `progress_reward` 引导，智能体可能在目标上空来回摇摆以保持距离奖励。应检查角速度惩罚是否足够抑制晃荡。