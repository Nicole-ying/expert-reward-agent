# Response Record

## 诊断

**0. 环境理解：** 观察空间包含 `base_pos`（底座水平位置）、`pole_angle`（杆偏角）、`pole_angular_vel`（杆角速度）。`original_env_reward` 恒定 +1.0/step，是纯生存信号。这是一个需要在轨道中心保持杆直立并存活 500 步的平衡任务。

**1. Agent 发生了什么？** 
- 两轮迭代都拿到满分 500（500 步 × 1.0 生存奖励），survival 已到天花板。
- 但 `progress_reward` 从 v1 的 -0.059 恶化到 v2 的 -0.189——绝对值惩罚（v2）的平衡质量反而不如二次惩罚（v1）。
- **核心问题：纯负惩罚始终与 `original_env_reward` (+1.0) 符号相反。** ratio=+5.29 但两者方向相反（progress 为负、original 为正），这是典型的 misalignment：环境说"活着就好"，我们的奖励说"你永远不够好"。

**2. 哪个组件是主因？**
- 只有一个组件 `progress_reward`。它的数学形态从 v1 的二次改为 v2 的绝对值后平衡质量下降（-0.059 → -0.189），说明绝对值在零点附近的恒定梯度虽然"不会消失"，但在本任务中反而让精细平衡变难——二次惩罚在接近零时梯度自然变小，恰好适合微调。
- 更深层问题：纯惩罚永远为负，agent 学到的"最优"只是"最不差"（reward≈0），缺少正向强化来巩固良好行为。

**3. 我之前改了什么？**
- v1→v2：二次→绝对值，声称"梯度不消失便于精细平衡"。实际效果相反：平衡质量恶化。**不应再往绝对值方向调整。**

**修复方案：** 回到 v1 的二次惩罚形态（已验证平衡质量更好），并在数学形式上增加正基线（层次 2），使 reward 在良好平衡时为正，与生存奖励方向对齐。这是一个组件的形式修改，不涉及其他组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 生存平衡任务 reward_v3。

    诊断：v1/v2 的纯负惩罚始终与 original_env_reward (+1.0) 符号相反，
    agent 学到"最不差"而非"最好"。v1 二次形态的平衡质量 (-0.059) 优于 v2 绝对值 (-0.189)。

    改动 (v2→v3)：回到 v1 二次惩罚系数，并增加正基线 +1.0。
    - 良好平衡时 reward≈+1.0，与生存奖励同向对齐，提供正向强化。
    - 偏离增大时 reward 平滑下降，极端情况可转负，梯度连续。
    - 保持 v1 验证过的系数：w_angle=5.0, w_angular_vel=0.5, w_pos=0.1。
    """
    base_pos = next_obs[0]
    pole_angle = next_obs[2]
    pole_angular_vel = next_obs[3]

    w_angle = 5.0
    w_angular_vel = 0.5
    w_pos = 0.1

    # 偏离惩罚：二次形态（v1 验证，精细平衡优于绝对值）
    penalty = (
        w_angle * (pole_angle ** 2) +
        w_angular_vel * (pole_angular_vel ** 2) +
        w_pos * (base_pos ** 2)
    )

    # 正基线奖励：完美平衡时≈1.0，对齐生存奖励方向
    progress_reward = 1.0 - penalty

    total_reward = progress_reward
    components = {"progress_reward": progress_reward}

    return float(total_reward), components
```
