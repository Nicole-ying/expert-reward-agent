## 诊断

### 1. 发生了什么？
- **得分：** 250.40（比 best 244.95 略高，但提升很小）
- **episode_length：** 1077（比 best 的 1385 明显下降）
- **结论：** agent 在更短的时间内拿到了差不多的分数 → 它跑得更快了（progress_reward 从 0.572→0.921），但摔倒得更早了（长度缩短 300 步）。

### 2. 哪个组件是主要原因？
- **progress_reward** ratio=1.0（主信号），mean=0.921，比上一轮的 0.572 大幅提升 ✅ 前进速度确实提高了
- **alive_bonus** ratio=0.19，mean=0.175，nonzero_rate=99% → 从二值 0.5 改成连续 0.2*乘积后，均值从 0.497 降到 0.175，但 nonzero_rate 很高（99%），说明 agent 大部分时间都在站立
- **stability_penalty** ratio=-0.039，绝对值很小，不是问题
- **original_env_reward** ratio=0.143（正号），与主信号同向 → 对齐没问题

**核心问题：** 上一轮我把 `fwd_scale` 从 2.0 提高到 3.0，同时把 `alive_bonus` 从二值 0.5 改成连续 0.2*乘积。结果是 progress 变强了（agent 跑更快），但 alive_bonus 变弱了（从 0.5→最大 0.2，且乘积形式使实际值更低）。agent 为了追求速度奖励而牺牲了稳定性，导致更早摔倒。

**根因：** 连续 alive_bonus 的最大值 0.2 太弱了，相对于 progress_reward（均值 0.92）来说，存活奖励的引导力不足。agent 倾向于"跑快然后摔倒"而不是"稳定地跑更久"。

### 3. 我之前改了什么？
- 上一轮（iter 2）：fwd_scale 2.0→3.0，alive_bonus 从二值 0.5 改成连续 0.2*angle_factor*vel_factor
- 效果：progress 提升，但 episode_length 下降，总分几乎没变

### 修改方案

**层次 1：改系数。** 不改变数学形式，只调整系数。

1. **增强 alive_bonus：** 从 0.2 提高到 0.5，让存活信号更有分量。连续形式保留（有梯度），但幅度恢复到与原来二值版本相当的水平。
2. **降低 fwd_scale：** 从 3.0 回到 2.5（介于 2.0 和 3.0 之间），稍微抑制过度追求速度的倾向。

这样 agent 会得到更强的"保持站立"信号，同时前进速度的驱动力适中，不会为了速度牺牲稳定性。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：前进速度奖励 ==========
    forward_velocity = next_obs[2]
    fwd_scale = 2.5  # 从3.0降到2.5，平衡速度与稳定性
    progress_reward = fwd_scale * forward_velocity
    
    # ========== 存活奖励：连续化，鼓励稳定站立 ==========
    hull_angle = abs(next_obs[0])
    hull_angular_vel = abs(next_obs[1])
    
    # 连续存活奖励：角度和角速度都接近0时拿满
    angle_factor = 1.0 - (hull_angle / 0.5) ** 2
    vel_factor = 1.0 - (hull_angular_vel / 2.0) ** 2
    angle_factor = max(0.0, angle_factor)
    vel_factor = max(0.0, vel_factor)
    
    alive_bonus = 0.5 * angle_factor * vel_factor  # 从0.2提高到0.5，增强存活引导
    
    # ========== 稳定性约束：适度惩罚 ==========
    angle_penalty_scale = 1.0
    angular_vel_penalty_scale = 0.5
    
    angle_penalty = angle_penalty_scale * (hull_angle ** 2)
    angular_vel_penalty = angular_vel_penalty_scale * (hull_angular_vel ** 2)
    stability_penalty = -(angle_penalty + angular_vel_penalty)
    
    # ========== 总奖励 ==========
    total_reward = progress_reward + alive_bonus + stability_penalty
    
    # ========== 组件记录 ==========
    components = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components
```