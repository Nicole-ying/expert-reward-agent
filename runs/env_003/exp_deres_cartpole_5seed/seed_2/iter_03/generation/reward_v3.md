## 诊断

### 0. 环境理解
Env_003 是一个 CartPole 式的生存平衡任务：obs = [cart_pos, cart_vel, pole_angle, pole_angvel]。目标是在 500 步内保持 pole 直立、cart 不越界。original_env_reward = 每步 +1。

### 1. Agent 发生了什么？
**任务已完美解决。** score=500（满分），episode_length=500（满时长），early_terminal=0，errors=0。agent 在每个回合中稳定存活全部 500 步。

### 2. 哪个组件是主要原因？
所有组件表现健康：
- `progress_reward`：mean=0.981，nonzero_rate=100%，作为主学习信号稳定工作
- `stability_penalty`：mean=-0.0037，ratio_to_progress=-0.0037，远低于 0.5 的警戒线，是极弱的背景信号，不干扰学习
- `original_env_reward` 与 `generated_reward` 同向（ratio 均为正），无 misalignment

### 3. 我之前改了什么？
Memory 显示 iter 1 和 iter 2 使用完全相同代码，两次都得 500 分。标记为 "target_solved_no_improvement"——即上限已触达，无改进空间。

### 根因判断
奖励函数未失败。但已连续两轮无变化且得分封顶。当前唯一可优化的点是 `stability_penalty` 使用 `abs()` 函数，其梯度在零点不连续（subgradient 从 -1 跳变到 +1），对优化不够友好。虽然当前影响微乎其微（ratio 仅 -0.0037），但改为平方形式可提供更平滑的梯度，且保持同等量级。

**修改方案：层次 2（改数学形式）**——将 `stability_penalty` 从 `abs()` 改为平方形式，系数从 0.01 调整为 0.03 以维持相似量级。这是唯一修改的组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for Env_003 (CartPole-like survival balance task).

    DIAGNOSIS: Task already solved (500/500, full survival). The only refinement
    is changing stability_penalty from abs() to squared form for smoother gradients
    (abs has discontinuous subgradient at zero). Coefficient adjusted 0.01→0.03
    to maintain similar magnitude.

    - progress_reward: baseline - (w_angle * angle^2 + w_pos * pos^2)
      Encourages keeping the pole near upright and the cart near the center.
    - stability_penalty: light squared penalty on velocities for smooth gradients.
    """
    w_angle = 5.0
    w_pos = 0.1
    baseline = 1.0

    lambda_vel = 0.03
    lambda_angvel = 0.03

    pos = next_obs[0]
    vel = next_obs[1]
    angle = next_obs[2]
    angvel = next_obs[3]

    progress_reward = baseline - (w_angle * (angle ** 2) + w_pos * (pos ** 2))
    stability_penalty = -lambda_vel * (vel ** 2) - lambda_angvel * (angvel ** 2)

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
    }

    return float(total_reward), components
```