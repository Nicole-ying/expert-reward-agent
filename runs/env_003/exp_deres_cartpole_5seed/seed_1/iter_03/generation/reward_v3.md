## 诊断

### 0. 任务理解
这是一个生存-平衡任务：保持杆竖直、底座在轨道内，存活 500 步即满分。观测 4 维，动作 2 离散（左/右推力）。info 为空。

### 1. Agent 发生了什么？
**满分 500，完全存活。** episode_length=500，score=500，无 early termination。任务已解决。

### 2. 哪个组件是主要原因？
两个组件都接近零（稳态时 pole 几乎不动、底座几乎静止），这是完美平衡的自然结果：
- `progress_reward` mean=-0.00037，略负（小幅度振荡导致 cost 微增）
- `stability_penalty` mean=-0.00032，ratio=-0.32，在合理范围（<0.5）
- 无组件异常，无需削弱任何组件

### 3. 之前改了什么？
Iter 1→2：stability_penalty 权重降 10x（0.01→0.001），解决了惩罚主导问题，达到满分。**该方向已证明正确。**

### v3 修改方向
Iter 2 代码已满分，但 v3 验证失败（可能因自动生成代码有语法/约束违规）。按照 revert 规则，回到 best 代码并做一个新修改：**添加小量生存奖励**，使稳态时总奖励为正值，更贴合"存活即好"的任务本质。其他组件不变。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward — v3.

    Based on iter 2 (score=500, solved).  One change: added a small per-step
    survival_bonus so that the total reward stays slightly positive even when
    the agent is perfectly balanced (previously mean ~ -0.0007).

    Signals:
      - progress_reward: encourages moving towards upright + center.
      - stability_penalty: light damping on velocities to suppress jitter.
      - survival_bonus: constant small positive per step.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.1

    # kept low from iter 2 (0.001) — ratio_to_progress ~ -0.32, healthy
    w_vel = 0.001
    w_angvel = 0.001

    # new in v3: tiny constant to keep total reward positive at steady state
    survival_bonus = 0.005

    # -- cost helper --
    def cost(o):
        return w_pos * abs(o[0]) + w_angle * abs(o[2])

    # -- main progress reward (potential-based shaping) --
    progress_reward = scale_progress * (cost(obs) - cost(next_obs))

    # -- light stability penalty --
    stability_penalty = -w_vel * abs(next_obs[1]) - w_angvel * abs(next_obs[3])

    # -- total --
    total_reward = progress_reward + stability_penalty + survival_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "survival_bonus": survival_bonus,
    }

    return float(total_reward), components
```