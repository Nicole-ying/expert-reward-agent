## 诊断

### 0. 环境与任务理解
生存平衡任务。观察 `[base_pos, base_vel, pole_angle, pole_angular_vel]`，动作是离散二值推力。终止条件：`|base_pos| > 2.4` 或 `|pole_angle| > 0.20944`。最大 500 步截断。

### 1. Agent 发生了什么？
**score=500, len=500, errors=0** — agent 在所有 episode 中都完美存活到截断上限，任务已经解决。`progress_reward` 均值 -0.059，非常接近零，说明 agent 维持了极好的平衡。

### 2. 哪个组件是主要原因？
只有一个组件 `progress_reward`（负惩罚），它驱动 agent 最小化角度/位置/角速度的偏差。`original_env_reward` ratio=16.96（正向）与 progress_reward 反向，但这只是因为 progress_reward 是惩罚（负值），不存在 misalignment。

### 3. 我之前改了什么？
Memory 显示 iter 1 使用 progress_reward 骨架，得分 500，action=target_solved_new_best。**v2 验证失败**——但我看到的"上一轮代码"是 iter 1 的成功版本。v2 是系统自动生成的或之前提交的版本，其代码未知。我需要基于 iter 1 的成功经验生成新的 v2。

**根因判断：** v2 验证失败的具体错误未知，但 iter 1 代码在逻辑和约束层面都是正确的。最安全的策略是保持核心方法（负惩罚引导平衡），做一个单一的数学形式改进：**将二次惩罚改为绝对值惩罚**。理由：
- 二次惩罚在接近零时梯度消失（梯度 ∝ x），agent 在接近完美时学习信号变弱
- 绝对值惩罚在接近零时梯度恒定（梯度 ∝ sign(x)），提供持续的精细化驱动力
- 量级通过系数调整保持与 v1 一致，不会引入新的量级失衡问题
- 这是层次 2 的改动（数学形式），符合"每次改一个信号"原则

系数调整以匹配 v1 在终止阈值附近的惩罚量级：
- `w_angle`: v1 在 0.20944 处惩罚=5.0×0.04386=0.219 → 绝对值版 w=0.219/0.20944≈1.05，取 **1.0**
- `w_angular_vel`: 保持 **0.5**
- `w_pos`: v1 在 2.4 处惩罚=0.1×5.76=0.576 → 绝对值版 w=0.576/2.4=0.24，取 **0.25**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 生存平衡任务 reward_v2。
    
    改动 (v1→v2)：将二次惩罚改为绝对值惩罚。
    - 二次惩罚在接近零时梯度消失，绝对值提供恒定梯度，便于精细平衡。
    - 系数重新标定以在终止阈值附近保持与 v1 相同的惩罚量级。
    """
    # 提取下一步观察
    base_pos = next_obs[0]            # 底座水平位置
    pole_angle = next_obs[2]          # 杆偏角 (rad)
    pole_angular_vel = next_obs[3]    # 杆角速度

    # 惩罚系数（已针对绝对值形态重新标定）
    w_angle = 1.0
    w_angular_vel = 0.5
    w_pos = 0.25

    # 核心学习信号：偏离直立和中心的绝对值惩罚
    progress_reward = -(
        w_angle * abs(pole_angle) +
        w_angular_vel * abs(pole_angular_vel) +
        w_pos * abs(base_pos)
    )

    total_reward = progress_reward
    components = {"progress_reward": progress_reward}

    return float(total_reward), components
```