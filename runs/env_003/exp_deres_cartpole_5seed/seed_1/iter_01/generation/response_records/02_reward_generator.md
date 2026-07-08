# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward.
    Uses:
    - progress_reward: dense improvement towards stable upright and center.
    - stability_penalty: light penalty on linear/angular speeds.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.1

    w_vel = 0.01
    w_angvel = 0.01

    # -- cost helper --
    def cost(o):
        return w_pos * abs(o[0]) + w_angle * abs(o[2])

    # -- main progress reward --
    progress_reward = scale_progress * (cost(obs) - cost(next_obs))

    # -- light stability penalty (only on next step) --
    stability_penalty = -w_vel * abs(next_obs[1]) - w_angvel * abs(next_obs[3])

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward** – 主学习信号（密集，每步可导）  
   - 角色：生存‑平衡任务的核心驱动力。  
   - 形式：\( \text{cost}(obs) - \text{cost}(next_obs) \)，再缩放。  
   - cost 定义：\( |obs[0]| \cdot 1.0 + |obs[2]| \cdot 10.0 \)，即底座偏离中心 + 杆倾斜的线性组合，角度权重更高。  
   - 含义：agent 朝着“底座居中、杆竖直”的方向移动时获得正奖励，恶化时获得负奖励。  
   - 从专家骨架 `progress_delta_reward` 异步适配而来，目标从“到达”改为“稳态”。

2. **stability_penalty** – 轻量稳定约束  
   - 角色：抑制大幅振荡，使运动更平滑。  
   - 形式：\( -0.01 \cdot (|next\_obs[1]| + |next\_obs[3]|) \)。  
   - 权重很小，仅作为方向盘，不会压制主学习信号。  
   - 从专家骨架 `stability_penalty` 精简而来，去掉了对角度和位置的惩罚（已包含在 progress_reward 中），只保留速度和角速度。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡明确告知 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，同时 `info` 字典为空。  
- 因此无法也不应发明不存在的 `info['success']` 或终局失败标记。  
- 终局惩罚已隐式包含在序列终止后的零奖励中；同时 `progress_reward` 会在越界/倾倒的前一步产生较大负值（因为 cost(next_obs) 会迅速增大），这提供了额外的梯度。

## 未在本阶段使用的骨架及其原因

- **terminal_success_reward / terminal_failure_penalty** – 缺少显式标记。  
- **distance_reward** – 本任务无固定目标位置，不适用。  
- **potential_based_shaping** – 可用但 v1 优先使用更直观的 delta‑cost 形式，势能塑造留待后续需要更严格理论保证时再加入。  
- **soft_landing_proxy** – 任务不是“着陆”，没有离散完成事件，不需要近似成功信号。  
- **energy_penalty / time_penalty / action_penalty** – 效率类约束默认 v3+ 才引入，避免 agent 在学会稳定前就因害怕代价而不敢行动。  
- **gated_reward** – 没有不安全 vs 正常的分区需求，v1 不采用复杂门控。

## 训练后应关注的 failure mode

1. **high_reward_without_stability**  
   agent 可能学会在中心附近缓慢摇摆但始终不终止，因为 progress_reward 鼓励中心化和低角度，但若权重不协调可能出现小幅振荡换取奖励。  
   → 检查奖励曲线与杆角度、位置轨迹，必要时微调 `w_pos` / `w_angle` 比例。

2. **survival without centering**  
   如果 progress_reward 中位置权重太低，agent 可能停在边界附近但勉强不倒杆。  
   → 观察底座位置分布，若偏向一侧可适当提高 `w_pos` 或缩放。

3. **oscillation exploit**  
   由于稳定性惩罚很轻，agent 可能高频切换推力以持续获得 progress_reward 的正负波动。  
   → 若观测到高频率动作，后续迭代可加入 action_smoothness（但 v1 离散动作暂不适用），或微调惩罚系数。

4. **too-cautious agent**  
   如果 stability_penalty 过强，agent 可能停在中间不敢做任何动作（但离散动作无法完全不动，可能反复推同一方向导致漂移）。  
   → 正常情形下 `w_vel/w_angvel = 0.01` 足够弱，通常不会导致此问题。
