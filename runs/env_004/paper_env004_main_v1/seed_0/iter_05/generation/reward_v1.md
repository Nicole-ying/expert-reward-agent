# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Main learning signal: reward forward velocity only when positive.
    # This drives locomotion, while termination from falling or extreme tilt
    # provides implicit negative feedback against instability.
    forward_vel = obs[5]
    forward_reward = max(0.0, forward_vel)
    total_reward = forward_reward
    components = {
        'forward_reward': forward_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

- `forward_reward`：主学习信号，奖励正向瞬时速度 `max(0, obs[5])`。这是驱动前进的唯一正向激励。

## 每个组件的角色

- `forward_reward` 作为 **主学习信号**：告诉 agent 向前移动才能得分，且速度越快得分越高。由于是稠密连续信号，每步都提供梯度，能高效引导 locomotion 策略。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确声明 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，且 `info` 字典为空。因此无法获得成功/失败标志，**禁止使用** terminal_success_reward 和 terminal_failure_penalty。

## 为什么舍弃稳定性惩罚、垂直活动、动作代价等组件

- 此前迭代中 `stability_penalty` 与 `forward_stability_reward` 组合达到 2663 分，但加入垂直活动组件后分数急剧下降，表明手动施加的姿态/速度约束可能压制了跳跃所需的动态探索。
- 本版提出一个 **截然不同的设计假设**：完全依赖环境自身的终止条件（torso_height、torso_angle 越界即结束）来提供隐式稳定性惩罚。agent 为避免回合过早终止，必须自发学习保持直立和控制高度，从而将“稳定”转化为内在行为，不再需要在奖励中线性叠加惩罚项。
- 动作代价（energy_penalty）和效率类组件按照原则 5 留到后续迭代再加入。

## 后续迭代考虑

- 若 agent 出现 `hover_or_stand_still`（原地站立只靠存活奖励），可加入小额 `alive_bonus` 并配合前向奖励引导。
- 若 agent 出现 `fast_then_fail`（高速前进但频繁摔倒），可考虑用轻量、平滑的稳定性约束（如低权重的 `stability_penalty`）进行微调。
- 若动作过于抖动，可引入 `action_smoothness_penalty`。

## 训练后应关注的 failure mode

1. **静止不前**：agent 为了不触发终止条件完全不动，`forward_reward=0`，需要观察是否因环境长期生存收益高于移动收益。
2. **极低速度的“爬行”步态**：agent 用小幅度关节运动产生微小前进速度，但仍可能摔倒，需要查看奖励曲线能否甄别这种低效策略。
3. **高速摔倒**：正向速度奖励强但缺少稳定性引导，可能短期内分数剧烈波动；如果长期难以收敛，再考虑添加轻微的直立倾向信号。