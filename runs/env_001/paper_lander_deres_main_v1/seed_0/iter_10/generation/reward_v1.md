# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next distance to the landing target (target is at (0,0) in relative coordinates)
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d = (x**2 + y**2) ** 0.5
    nd = (nx**2 + ny**2) ** 0.5
    
    # Main learning signal: dense progress towards the target
    progress_reward = d - nd  # positive when moving closer
    
    # Constraint: penalty for tilting away from upright
    body_angle = abs(next_obs[4])
    attitude_weight = 0.1
    attitude_penalty = -attitude_weight * body_angle
    
    # Total reward
    total_reward = progress_reward + attitude_penalty
    
    # Components dictionary (only the terms directly summed)
    components = {
        'progress_reward': progress_reward,
        'attitude_penalty': attitude_penalty
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用了哪些奖励组件
1. **progress_reward**（主学习信号）：使用 `d(obs) - d(next_obs)`，每一步接近目标时获得正奖励，远离目标时受到惩罚。此信号稠密且与任务目标（飞向目标点）直接相关，是奖励中最主要的正向驱动。
2. **attitude_penalty**（稳定约束）：惩罚机身偏离竖直的角度 `|body_angle|`，权重为 0.1，属于轻量约束。它引导飞行器在靠近目标的过程中保持竖直姿态，为最终着陆时的“竖直、平稳”条件提供早期信号，但数值上远小于 progress_reward，避免压制探索。

## 每个组件的角色
- **progress_reward** 是核心学习信号，告诉 agent “靠近目标有好结果”。它每步可计算，梯度连续，符合导航任务的需要。
- **attitude_penalty** 是辅助约束，角色相当于“方向盘”——提示 agent 不能以任意姿态接近，必须保持竖直。它不是“刹车”，因此权重较小，确保主信号仍占主导。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- 环境卡片明确声明 **explicit_success_flag_available = false** 且 info 字典为空，不存在任何显式成功或失败标志。
- 当前无法可靠判断终止原因来给予 success/failure 终端奖励，因此不引入这两类组件，避免发明不存在的信息字段。

## 哪些组件留到后续迭代
- **速度惩罚**（对水平/垂直速度的轻量惩罚）：当前未加入，因为 agent 首先需要学会到达目标区域。若后续出现高速撞地、不回降速的情况，再以小权重加入。
- **接触/着陆近似奖励**（双腿触地 + 低速 + 竖直的组合奖励）：作为任务完成的软性 proxy，可用于强化真正“停靠”行为。在 v1 阶段先不引入，以降低组合复杂性。
- **效率代价**（引擎推力惩罚或时间惩罚）：v1 阶段应优先完成任务，后续再加入以鼓励省油、快速。

## 训练后应观察哪些 failure mode
- **目标附近震荡不完成**：progress_reward 可能让 agent 在接近目标时来回摆动以持续获得小进步奖励，而不执行最终着陆动作。
- **频繁 crash**：当机身倾斜或快速下降时可能直接撞击地面终止，若该模式占优，需引入更直接的 crash 惩罚或速度上限。
- **姿态过大**：若 attitude_penalty 权重不够，agent 可能保持侧倾飞行，难以满足着陆的竖直要求。
- **推力滥用**：agent 可能持续使用主引擎维持高度而并非滑翔接近，导致效率低下；该项留待效率组件加入后改善。